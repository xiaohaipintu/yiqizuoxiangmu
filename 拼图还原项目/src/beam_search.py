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


def expand(pieces, placed, polys, used, qid, cov, overlap_tol, mirror_map=None):
    """把所有「已放置碎片的边 × 待放碎片的边」的贴合方式列出来。

    mirror_map 存在时，待放碎片会**同时尝试原版和镜像版**——每个组件都试镜像。
    返回 (已放块id, 已放边号, 新块边号, 角度, tx, ty, 世界多边形, 是否镜像)
    """
    variants = [(pieces[qid], False)]
    if mirror_map and qid in mirror_map:
        variants.append((mirror_map[qid], True))
    out = []
    for k, item in enumerate(placed):
        pid = int(item[0])
        pw = polys[k]
        p = pieces[pid]
        for ei in range(p.n):
            if ei in used[pid]:
                continue
            aw, bw = pw[ei], pw[(ei + 1) % p.n]
            La = float(np.linalg.norm(bw - aw))
            if La < 1e-6:
                continue
            u = (bw - aw) / La
            for q, q_mirror in variants:
                for fi in range(q.n):
                    eb = q.edges[fi]
                    Lb = float(eb["length"])
                    for anchor in (True, False):
                        ang, tr = place_edge(q, fi, aw, bw, anchor)
                        wq = world(q, ang, tr)
                        qaw = (G.rotation(ang) @ np.asarray(eb["a"], float)) + np.asarray(tr, float)
                        qbw = (G.rotation(ang) @ np.asarray(eb["b"], float)) + np.asarray(tr, float)
                        proj = np.clip([((qaw - aw) @ u), ((qbw - aw) @ u)], 0, La)
                        shared = float(max(0.0, proj.max() - proj.min()))
                        if shared < cov * min(La, Lb):
                            continue
                        ca, cq = p.poly.mean(axis=0), wq.mean(axis=0)
                        sa = u[0] * (ca[1] - aw[1]) - u[1] * (ca[0] - aw[0])
                        sq = u[0] * (cq[1] - aw[1]) - u[1] * (cq[0] - aw[0])
                        if sa * sq >= 0:
                            continue
                        ov = sum(G.poly_overlap_area(wq, pp) for pp in polys)
                        if ov > overlap_tol * q.area:
                            continue
                        out.append((pid, ei, fi, ang, float(tr[0]), float(tr[1]), wq, q_mirror))
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
                    help="最多保存多少个完整解到 beam_solutions.json（上层重排要看更多候选）")
    ap.add_argument("--no-mirror", action="store_true",
                    help="关闭逐块镜像（默认每个组件都会尝试镜像）")
    args = ap.parse_args()

    outdir = pathlib.Path(args.out)
    data = json.loads(pathlib.Path(args.pieces).read_text(encoding="utf-8"))
    pieces = {int(d["id"]): Piece(d) for d in data["pieces"]}
    mirror_map = {} if args.no_mirror else {pid: mirror_piece(p) for pid, p in pieces.items()}
    deadline = time.time() + args.time

    seed = max(pieces, key=lambda i: pieces[i].area)
    states = [([(seed, 0.0, 0.0, 0.0, False)],
               [world(pieces[seed], 0.0, (0.0, 0.0))],
               {pid: set() for pid in pieces})]
    print(f"碎片 {len(pieces)}，起始块 #{seed}，镜像={'开' if mirror_map else '关'}")
    render_progress(states[0][1], 1, len(pieces), gap_pct(states[0][1]),
                    outdir / "beam_progress.png")
    for level in range(2, len(pieces) + 1):
        cand = []
        for placed, polys, used in states:
            done = {r[0] for r in placed}
            for qid in pieces:
                if qid in done:
                    continue
                for (pid, ei, fi, ang, tx, ty, wq, q_mirror) in expand(
                        pieces, placed, polys, used, qid, args.cov, args.overlap, mirror_map):
                    nu = {kk: set(v) for kk, v in used.items()}
                    nu[pid].add(ei)
                    nu[qid] = {fi}
                    cand.append((placed + [(qid, ang, tx, ty, q_mirror)],
                                 polys + [wq], nu, gap_pct(polys + [wq])))
            if time.time() > deadline:
                break
        if not cand:
            print(f"第 {level} 层：无可行扩展")
            break
        cand.sort(key=lambda s: s[3])
        states = [(a, b, c) for a, b, c, _ in cand[: args.beam]]
        n_mir = sum(1 for a, _, _ in states
                    if any((len(r) > 4 and r[4]) for r in a))
        print(f"第 {level} 层：候选 {len(cand)}，缝最大 {cand[0][3]:.2f}% / "
              f"保留 {len(states)}（最差 {cand[min(len(cand) - 1, args.beam - 1)][3]:.2f}%）"
              f"，其中含镜像块 {n_mir}")
        render_progress(states[0][1], level, len(pieces), cand[0][3],
                        outdir / "beam_progress.png")
        if time.time() > deadline:
            print("超时，停在当前层")
            break

    full = sorted([(gap_pct(polys), placed, polys) for placed, polys, used in states
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
