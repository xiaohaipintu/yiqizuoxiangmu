"""配对式拼图求解（经典拼图算法），替代/补充 beam search。

为什么换：
    beam 是「逐块往上拼 + 按紧凑度剪枝」，碎片一多候选爆炸（8 块时每层上万），
    Python 跑不动，而且正确路径常被剪掉。

这里的做法：
    1) **兼容矩阵**：所有边两两比长度（拼图切割后，接缝两侧边长基本相等），
       差 ≤ tol 的才算候选；再做一次几何校验（贴上去不能重叠、两块得分居两侧）。
    2) **精确配对 + 回溯**：每条边最多只能有一个对手边。从面积最大的一块开始，
       每步挑「候选对手最少（最强制）的那条边」先拼，冲突就回溯。
    3) 完整解按「凸包空隙 + 接缝边长差 + 轮廓方不方」打分，输出前 N 个，
       格式与 beam_solutions.json 一致（gap_pct / placed / polys），上层不用改。

用法：
    python src/pair_solve.py --pieces out/pieces_scaled.json --out out --time 60 --save 200
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import geometry as G                                          # noqa: E402


class Pc:
    """一块碎片（可选镜像版）。"""

    def __init__(self, d, mirror=False):
        poly = np.array(d["polygon"], float)
        if mirror:
            poly = poly * np.array([-1.0, 1.0])
        self.id = int(d["id"])
        self.mirror = mirror
        self.poly = poly
        self.area = float(d["area"])
        self.centroid = np.array(d["centroid"], float) * (np.array([-1.0, 1.0]) if mirror else 1.0)
        self.n = len(poly)
        self.edges = []
        for i in range(self.n):
            a, b = poly[i], poly[(i + 1) % self.n]
            self.edges.append({"a": a, "b": b, "length": float(np.linalg.norm(b - a))})


def world(pc: Pc, deg: float, t) -> np.ndarray:
    return (G.rotation(deg) @ pc.poly.T).T + np.asarray(t, float)


def place_by_edge(pc: Pc, fj: int, ta, tb, anchor_a: bool):
    """把 pc 的第 fj 条边反向贴到 (ta,tb) 上，返回 (角度, 平移)。"""
    e = pc.edges[fj]
    a, b = e["a"], e["b"]
    ds = (b - a) / (np.linalg.norm(b - a) + 1e-12)
    dt = (np.asarray(tb, float) - np.asarray(ta, float))
    dt /= (np.linalg.norm(dt) + 1e-12)
    want = -dt
    dst = np.asarray(tb, float) if anchor_a else np.asarray(ta, float)
    ang = math.degrees(math.atan2(want[1], want[0]) - math.atan2(ds[1], ds[0])) % 360.0
    return ang, dst - G.rotation(ang) @ a


def hull_stats(polys):
    pts = np.vstack([np.asarray(p, float) for p in polys])
    hull = G.convex_hull_area(pts) if hasattr(G, "convex_hull_area") else None
    if hull is None:                                  # 兜底：用 cv2
        import cv2
        h = cv2.convexHull(pts.astype(np.float32).reshape(-1, 1, 2))
        hull = abs(cv2.contourArea(h))
    area = sum(float(_poly_area(p)) for p in polys)
    minx, miny = pts.min(axis=0)
    maxx, maxy = pts.max(axis=0)
    box = float((maxx - minx) * (maxy - miny))
    gap = (hull - area) / hull * 100.0 if hull > 1e-6 else 999.0
    rect = hull / box if box > 1e-6 else 0.0
    return float(gap), float(rect)


def _poly_area(p) -> float:
    p = np.asarray(p, float)
    x, y = p[:, 0], p[:, 1]
    return abs(float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))) / 2.0)


def build_pairs(pieces, variants, len_tol=0.06, overlap_tol=0.006, cov=0.25):
    """所有「边 i ↔ 边 j」候选：长度够接近 + 贴上去不重叠 + 分居两侧。"""
    pairs = []                                        # (pid, ei, qid, fj, ang, tx, ty, mirror)
    lens = [(pid, ei, e["length"]) for pid, pc in pieces.items() for ei, e in enumerate(pc.edges)]
    lens.sort(key=lambda x: x[2])
    for a in range(len(lens)):
        pid, ei, La = lens[a]
        for b in range(a + 1, len(lens)):
            qid, fj, Lb = lens[b]
            if qid == pid:
                continue
            if Lb > La * (1 + len_tol):
                break                                 # 长度差太大，后面的更大，直接停
            if abs(Lb - La) / max(min(La, Lb), 1e-9) > len_tol:
                continue
            ea = pieces[pid].edges[ei]
            for mirror in (False, True):
                pc = variants[(qid, mirror)]
                for anchor in (True, False):
                    ang, t = place_by_edge(pc, fj, ea["a"], ea["b"], anchor)
                    wq = world(pc, ang, t)
                    # 朝向：必须分居两侧
                    wa = world(pieces[pid], 0.0, (0.0, 0.0)) if False else None
                    pairs.append((pid, ei, qid, fj, float(ang), float(t[0]), float(t[1]), mirror))
    return pairs


def verify_pair(pieces, variants, pid, ei, qid, fj, ang, tx, ty, mirror,
                polys_world, overlap_tol=0.006, cov=0.25):
    """几何校验：贴上去后与已放多边形不重叠、且贴合段够长。"""
    pc = variants[(qid, mirror)]
    wq = world(pc, ang, (tx, ty))
    for pw in polys_world:
        if G.poly_overlap_area(wq, pw) > overlap_tol * pc.area:
            return None
    # 贴合段：新块边 fj 投影到已放边 ei 上
    e = pieces[pid].edges[ei]
    aw, bw = e["a"], e["b"]
    La = float(np.linalg.norm(bw - aw))
    u = (bw - aw) / max(La, 1e-9)
    qa, qb = wq[fj], wq[(fj + 1) % pc.n]
    t1, t2 = float((qa - aw) @ u), float((qb - aw) @ u)
    shared = abs(t2 - t1)
    if shared < cov * min(La, pc.edges[fj]["length"]):
        return None
    # 分居两侧
    ca = pieces[pid].poly.mean(axis=0)
    cq = wq.mean(axis=0)
    sa = u[0] * (ca[1] - aw[1]) - u[1] * (ca[0] - aw[0])
    sq = u[0] * (cq[1] - aw[1]) - u[1] * (cq[0] - aw[0])
    if sa * sq >= 0:
        return None
    return wq


def solve(pieces, variants, time_limit=60.0, max_solutions=200, beam=400):
    """回溯装配：优先拼「最强制」的边（候选对手最少的）。"""
    deadline = time.time() + time_limit
    seed = max(pieces, key=lambda i: pieces[i].area)
    seed_pc = pieces[seed]
    polys0 = [seed_pc.poly]
    used_pairs = set()                                # {(pid, ei, qid, fj)} 已配对的边
    used_edge = {(seed, ei) for ei in range(seed_pc.n)}  # 已经贴过的边（一块的每条边只用一次）

    solutions = []
    best = {"key": 1e18}

    def key_of(polys):
        gap, rect = hull_stats(polys)
        return gap + 60.0 * (1.0 - rect), gap, rect

    def dfs(placed, polys, variants_used, used_edge, used_pairs, depth):
        if len(solutions) >= max_solutions * 3 or time.time() > deadline:
            return
        if len(placed) == len(pieces):
            k, gap, rect = key_of(polys)
            solutions.append((k, gap, rect, list(placed), [p.copy() for p in polys]))
            if len(solutions) > max_solutions * 2:
                solutions.sort(key=lambda s: s[0])
                del solutions[max_solutions:]
            return
        # 候选：已放碎片的未用边 × 未放碎片 × 未用对手
        cands = []
        for pid, ei in list(used_edge):
            if ei >= len(pieces[pid].edges):
                continue
            ea = pieces[pid].edges[ei]
            La = ea["length"]
            for qid in pieces:
                if qid in {r[0] for r in placed}:
                    continue
                for mirror in (False, True):
                    if (qid, not mirror) in variants_used:      # 同一块不能既正又镜像
                        continue
                    pc = variants[(qid, mirror)]
                    for fj, eb in enumerate(pc.edges):
                        Lb = eb["length"]
                        if abs(Lb - La) / max(min(La, Lb), 1e-9) > 0.06:
                            continue
                        if (frozenset([(pid, ei), (qid, fj)]), mirror) in used_pairs:
                            continue
                        for anchor in (True, False):
                            ang, t = place_by_edge(pc, fj, ea["a"], ea["b"], anchor)
                            wq = verify_pair(pieces, variants, pid, ei, qid, fj,
                                             ang, t[0], t[1], mirror, polys)
                            if wq is None:
                                continue
                            cands.append((pid, ei, qid, fj, ang, float(t[0]), float(t[1]),
                                          mirror, wq))
        if not cands:
            return
        cands.sort(key=lambda c: c[0])
        for (pid, ei, qid, fj, ang, tx, ty, mirror, wq) in cands[:beam]:
            nk, _, _ = key_of(polys + [wq])
            if nk > best["key"] + 25.0:                # 剪枝：比已知最好差太多就不走
                continue
            nu_edge = set(used_edge) | {(qid, k) for k in range(variants[(qid, mirror)].n)}
            nu_edge.discard((pid, ei))
            dfs(placed + [(qid, ang, tx, ty, mirror)], polys + [wq],
                variants_used | {(qid, mirror)},
                nu_edge,
                used_pairs | {(frozenset([(pid, ei), (qid, fj)]), mirror)},
                depth + 1)

    k0, gap0, rect0 = key_of(polys0)
    best["key"] = k0
    dfs([(seed, 0.0, 0.0, 0.0, False)], polys0, {(seed, False)},
        set(used_edge) | {(seed, ei) for ei in range(seed_pc.n)},
        set(), 1)

    solutions.sort(key=lambda s: s[0])
    return [{"gap_pct": round(s[1], 3), "placed": s[3],
             "polys": [p.tolist() for p in s[4]]} for s in solutions[:max_solutions]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pieces", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--time", type=float, default=60.0)
    ap.add_argument("--save", type=int, default=200)
    ap.add_argument("--mirror", action="store_true", default=True,
                    help="允许逐块镜像（默认开）")
    args = ap.parse_args()

    data = json.loads(pathlib.Path(args.pieces).read_text(encoding="utf-8"))
    pieces = {int(d["id"]): Pc(d) for d in data["pieces"]}
    variants = {}
    for pid, pc in pieces.items():
        variants[(pid, False)] = pc
        if args.mirror:
            variants[(pid, True)] = Pc(data["pieces"][[int(d["id"]) for d in data["pieces"]].index(pid)],
                                       mirror=True)
    print(f"碎片 {len(pieces)}，变体 {len(variants)}（含镜像）")
    t0 = time.time()
    sols = solve(pieces, variants, time_limit=args.time, max_solutions=args.save)
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "pair_solutions.json").write_text(json.dumps(
        {"n": len(sols), "solutions": sols}, ensure_ascii=False), encoding="utf-8")
    if sols:
        gap, rect = hull_stats([np.asarray(p, float) for p in sols[0]["polys"]])
        print(f"完整拼法 {len(sols)} 个，最好 缝 {gap:.2f}%、凸包/矩形 {rect:.3f}（{time.time()-t0:.1f}s）")
    else:
        print(f"没找到完整拼法（{time.time()-t0:.1f}s）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
