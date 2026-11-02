"""拼接搜索（不镜像）：允许部分贴合（一块的边可以只贴住另一块边的一段，即 T 型接缝）。

规则:
    * 只用边的几何：新块的一条边贴到已放块某条边的共线位置（反向、贴另一侧）；
      端点锚定有两种：新块 a 端贴目标 b 端 / 新块 b 端贴目标 a 端（后者对应"只贴一段"）；
    * 公共长度 ≥ cov × 较短边（默认 0.35）；
    * 不镜像/不翻面；不重叠（用 shapely 精确算，支持凹多边形）；
    * 逐层扩展 + beam 保留缝最小的若干状态。

用法:
    python src/beam_search.py --pieces out4/pieces.json --out out4 --beam 300 --cov 0.35
输出:
    out4/beam_solutions.json
    out4/beam_progress.png   搜索过程中每层刷新的「当前最优拼装」预览
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from assemble import Piece, world, place_edge  # noqa: E402
import geometry as G  # noqa: E402


def gap_pct(polys):
    pts = np.vstack(polys).astype(np.float32)
    hull = abs(float(cv2.contourArea(cv2.convexHull(pts))))
    s = sum(G.polygon_area(p) for p in polys)
    return max(0.0, hull - s) / s * 100.0


def render_progress(polys, level, n_total, gap, path) -> None:
    """把当前最优的「已拼部分」画成一张小图，每层覆写一次。

    搜索是纯 CPU 的，中间几十秒没有任何输出，网页端只剩一根进度条很乏味；
    这里每层落一张图，前端轮询就能看到拼装过程一层层长出来。只画轮廓，开销可忽略。
    """
    try:
        pts = np.vstack(polys).astype(np.float32)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        pad = max(x1 - x0, y1 - y0) * 0.08 + 20.0
        sx = 900.0 / max(x1 - x0 + 2 * pad, 1.0)
        sy = 620.0 / max(y1 - y0 + 2 * pad, 1.0)
        scale = max(min(sx, sy), 0.05)
        w = max(2, int((x1 - x0 + 2 * pad) * scale))
        h = max(2, int((y1 - y0 + 2 * pad) * scale))
        canvas = np.full((h, w, 3), (26, 26, 38), np.uint8)
        palette = [(235, 130, 60), (240, 160, 60), (120, 200, 90), (50, 160, 230),
                   (220, 90, 200), (230, 220, 60), (180, 70, 180), (90, 150, 250)]
        for i, poly in enumerate(polys):
            p = np.asarray(poly, np.float32).reshape(-1, 2)
            p = np.column_stack([(p[:, 0] - x0 + pad) * scale,
                                 (p[:, 1] - y0 + pad) * scale])
            cv2.polylines(canvas, [np.round(p).astype(np.int32)], True,
                          palette[i % len(palette)], 3, cv2.LINE_AA)
        cv2.putText(canvas, f"level {level}/{n_total}   gap {gap:.2f}%", (14, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.imwrite(str(path), canvas)
    except Exception:
        pass


def mirror_piece(p):
    """返回 p 的镜像副本（沿 y 轴反射：x → -x）。面积/边长不变，边的朝向翻转。"""

    def flip(pt):
        return [-float(pt[0]), float(pt[1])]

    edges = []
    for e in p.edges:
        ne = dict(e)
        ne["a"] = flip(e["a"])
        ne["b"] = flip(e["b"])
        edges.append(ne)
    return Piece({
        "id": p.id,
        "polygon": [flip(pt) for pt in p.poly],
        "area": p.area,
        "centroid": flip(p.centroid),
        "edges": edges,
    })


def _bbox_gap(e1a, e1b, e2a, e2b) -> float:
    """两条线段包围盒之间的空隙（用于快速预筛，不做精确判定）。"""
    ax0, ax1 = (e1a[0], e1b[0]) if e1a[0] <= e1b[0] else (e1b[0], e1a[0])
    ay0, ay1 = (e1a[1], e1b[1]) if e1a[1] <= e1b[1] else (e1b[1], e1a[1])
    bx0, bx1 = (e2a[0], e2b[0]) if e2a[0] <= e2b[0] else (e2b[0], e2a[0])
    by0, by1 = (e2a[1], e2b[1]) if e2a[1] <= e2b[1] else (e2b[1], e2a[1])
    dx = max(0.0, max(ax0, bx0) - min(ax1, bx1))
    dy = max(0.0, max(ay0, by0) - min(ay1, by1))
    return (dx * dx + dy * dy) ** 0.5


def _seg_overlap(e1a, e1b, e2a, e2b) -> float:
    """两条线段若近共线，返回重叠长度（否则 0）。"""
    v1 = e1b - e1a
    L1 = float(np.linalg.norm(v1))
    v2 = e2b - e2a
    L2 = float(np.linalg.norm(v2))
    if L1 < 1e-6 or L2 < 1e-6:
        return 0.0
    u1 = v1 / L1
    u2 = v2 / L2
    if abs(u1[0] * u2[1] - u1[1] * u2[0]) > 0.35:      # 夹角 > ~20° 不算共线
        return 0.0
    # 端点离对方直线的距离
    n1 = np.array([-u1[1], u1[0]])
    d2a = abs(float((e2a - e1a) @ n1))
    d2b = abs(float((e2b - e1a) @ n1))
    if min(d2a, d2b) > 6.0:
        return 0.0
    t1 = float((e2a - e1a) @ u1)
    t2 = float((e2b - e1a) @ u1)
    lo, hi = min(t1, t2), max(t1, t2)
    return max(0.0, min(hi, L1) - max(lo, 0.0))


def joint_badness(joints, pieces) -> float:
    """已做接缝两侧边长差的平均值（%）。joints: [(pid, ei, qid, fi)]"""
    if not joints:
        return 0.0
    bad = 0.0
    for pid, ei, qid, fi in joints:
        la = float(pieces[pid].edges[ei]["length"])
        lb = float(pieces[qid].edges[fi]["length"])
        m = min(la, lb)
        if m > 1e-6:
            bad += abs(la - lb) / m * 100.0
    return bad / len(joints)


def rect_ratio(polys) -> float:
    """凸包面积 / 外接矩形面积：越接近 1 越像矩形/方形。"""
    import cv2
    pts = np.vstack([np.asarray(p, np.float32) for p in polys])
    hull = cv2.convexHull(pts.reshape(-1, 1, 2))
    minx, miny = pts.min(axis=0)
    maxx, maxy = pts.max(axis=0)
    box = float((maxx - minx) * (maxy - miny))
    if box <= 1e-6:
        return 0.0
    return float(abs(cv2.contourArea(hull)) / box)


def expand(pieces, placed, polys, used, qid, cov, overlap_tol, mirror_map=None):
    """列出所有贴合方式；同时做「边的区间占用」检查。

    返回 (已放块id, 已放边号, 新块边号, 角度, tx, ty, 世界多边形, 是否镜像,
          占用段起点, 占用段终点, 新块边长)
    """
    variants = [(pieces[qid], False)]
    if mirror_map and qid in mirror_map:
        variants.append((mirror_map[qid], True))
    out = []
    for k, item in enumerate(placed):
        pid = int(item[0])
        pw = polys[k]
        p = pieces[pid]
        occ_map = used.get(pid) or {}
        for ei in range(p.n):
            aw, bw = pw[ei], pw[(ei + 1) % p.n]
            La = float(np.linalg.norm(bw - aw))
            if La < 1e-6:
                continue
            u = (bw - aw) / La
            occ = occ_map.get(ei, [])
            for q, q_mirror in variants:
                for fi in range(q.n):
                    eb = q.edges[fi]
                    Lb = float(eb["length"])
                    for anchor in (True, False):
                        ang, tr = place_edge(q, fi, aw, bw, anchor)
                        wq = world(q, ang, tr)
                        qaw = (G.rotation(ang) @ np.asarray(eb["a"], float)) + np.asarray(tr, float)
                        qbw = (G.rotation(ang) @ np.asarray(eb["b"], float)) + np.asarray(tr, float)
                        p0 = float((qaw - aw) @ u)
                        p1 = float((qbw - aw) @ u)
                        seg0, seg1 = (p0, p1) if p0 <= p1 else (p1, p0)
                        seg0, seg1 = max(0.0, seg0), min(La, seg1)
                        shared = max(0.0, seg1 - seg0)
                        if shared < cov * min(La, Lb):
                            continue
                        # ★ 同一段已被别的块占用 → 冲突，剪掉
                        if shared > 1e-6 and occ:
                            used_len = sum(max(0.0, min(seg1, e1) - max(seg0, e0))
                                           for e0, e1 in occ)
                            if used_len > 0.25 * shared:
                                continue
                        ca, cq = p.poly.mean(axis=0), wq.mean(axis=0)
                        sa = u[0] * (ca[1] - aw[1]) - u[1] * (ca[0] - aw[0])
                        sq = u[0] * (cq[1] - aw[1]) - u[1] * (cq[0] - aw[0])
                        if sa * sq >= 0:
                            continue
                        ov = sum(G.poly_overlap_area(wq, pp) for pp in polys)
                        if ov > overlap_tol * q.area:
                            continue
                        # ★ 几何去重：新块任何一条边，不能在空间上和"其它已贴边"重复占线
                        bad = False
                        for kk, pw_other in enumerate(polys):
                            other_pid = int(placed[kk][0])
                            o_n = len(pw_other)
                            for ej in range(o_n):
                                oa, ob = pw_other[ej], pw_other[(ej + 1) % o_n]
                                for fj in range(q.n):
                                    wa, wb = wq[fj], wq[(fj + 1) % q.n]
                                    if _bbox_gap(wa, wb, oa, ob) > 8.0:
                                        continue                      # 包围盒都离得远，跳过精确判定
                                    ovl = _seg_overlap(wa, wb, oa, ob)
                                    if ovl <= 1e-6:
                                        continue
                                    shorter = min(float(np.linalg.norm(wb - wa)),
                                                  float(np.linalg.norm(ob - oa)))
                                    if shorter < 1e-6:
                                        continue
                                    if ovl > 0.20 * shorter:
                                        # 允许「这条边正是本次贴合的那条」的重叠
                                        if other_pid == pid and ej == ei and fj == fi:
                                            continue
                                        bad = True
                                        break
                                if bad:
                                    break
                            if bad:
                                break
                        if bad:
                            continue
                        out.append((pid, ei, fi, ang, float(tr[0]), float(tr[1]), wq,
                                    q_mirror, seg0, seg1, Lb))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pieces", default="out4/pieces.json")
    ap.add_argument("--out", default="out4")
    ap.add_argument("--beam", type=int, default=300)
    ap.add_argument("--cov", type=float, default=0.35)
    ap.add_argument("--overlap", type=float, default=0.004)
    ap.add_argument("--time", type=float, default=180.0)
    ap.add_argument("--save", type=int, default=60,
                    help="最多保存多少个完整解到 beam_solutions.json")
    ap.add_argument("--no-mirror", action="store_true",
                    help="关闭逐块镜像（默认每个组件都会尝试镜像）")
    args = ap.parse_args()

    outdir = pathlib.Path(args.out)
    data = json.loads(pathlib.Path(args.pieces).read_text(encoding="utf-8"))
    pieces = {int(d["id"]): Piece(d) for d in data["pieces"]}
    mirror_map = {} if args.no_mirror else {pid: mirror_piece(p) for pid, p in pieces.items()}
    deadline = time.time() + args.time

    seed = max(pieces, key=lambda i: pieces[i].area)
    # used[pid][edge] = [(t0,t1)] 已占用区间
    states = [([(seed, 0.0, 0.0, 0.0, False)],
               [world(pieces[seed], 0.0, (0.0, 0.0))],
               {pid: {} for pid in pieces},
               [])]
    print(f"碎片 {len(pieces)}，起始块 #{seed}，镜像={'开' if mirror_map else '关'}")
    render_progress(states[0][1], 1, len(pieces), gap_pct(states[0][1]),
                    outdir / "beam_progress.png")
    for level in range(2, len(pieces) + 1):
        cand = []
        for placed, polys, used, joints in states:
            done = {r[0] for r in placed}
            for qid in pieces:
                if qid in done:
                    continue
                for (pid, ei, fi, ang, tx, ty, wq, q_mirror, seg0, seg1, Lb) in expand(
                        pieces, placed, polys, used, qid, args.cov, args.overlap, mirror_map):
                    nu = {kk: {e: list(v) for e, v in vv.items()} for kk, vv in used.items()}
                    nu[pid].setdefault(ei, []).append((seg0, seg1))
                    nu[qid] = {fi: [(0.0, Lb)]}
                    nj = joints + [(pid, ei, qid, fi)]
                    nps = polys + [wq]
                    g = gap_pct(nps)
                    # ★ 三指标：空隙小 + 接缝边长吻合 + 轮廓越方越好
                    key = (g + 1.5 * joint_badness(nj, pieces)
                           + 60.0 * (1.0 - rect_ratio(nps)))
                    cand.append((placed + [(qid, ang, tx, ty, q_mirror)],
                                 nps, nu, nj, key, g))
            if time.time() > deadline:
                break
        if not cand:
            print(f"第 {level} 层：无可行扩展")
            break
        cand.sort(key=lambda s: s[4])
        kept, seen = [], set()
        for a, b, c, d, _k, _g in cand:
            sig = tuple(sorted((int(r[0]), round(float(r[1]) / 5.0),
                                round(float(r[2]) / 10.0), round(float(r[3]) / 10.0),
                                bool(r[4]) if len(r) > 4 else False) for r in a))
            if sig in seen:
                continue                     # 位姿几乎相同的重复状态，只留一个
            seen.add(sig)
            kept.append((a, b, c, d))
            if len(kept) >= args.beam:
                break
        states = kept
        n_mir = sum(1 for a, _, _, _ in states if any((len(r) > 4 and r[4]) for r in a))
        print(f"第 {level} 层：候选 {len(cand)}，最好 缝 {cand[0][5]:.2f}%（键 {cand[0][4]:.2f}）/ "
              f"保留 {len(states)}，其中含镜像块 {n_mir}")
        render_progress(states[0][1], level, len(pieces), cand[0][5],
                        outdir / "beam_progress.png")
        if time.time() > deadline:
            print("超时，停在当前层")
            break

    full = sorted([(gap_pct(polys), placed, polys) for placed, polys, used, joints in states
                   if len(placed) == len(pieces)], key=lambda x: x[0])
    if full:
        n_mirror_sol = sum(1 for _, pl, _ in full if any(len(r) > 4 and r[4] for r in pl))
        print(f"完整拼法 {len(full)} 个，最好 缝 {full[0][0]:.2f}%（含镜像块 {n_mirror_sol} 个）")
    else:
        print("没有完整拼法")
    (outdir / "beam_solutions.json").write_text(json.dumps(
        {"n": len(full), "solutions": [{"gap_pct": g, "placed": pl,
                                        "polys": [p.tolist() for p in ps]}
                                       for g, pl, ps in full[:args.save]]}, ensure_ascii=False),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
if __name__ == "__main__":
    raise SystemExit(main())
