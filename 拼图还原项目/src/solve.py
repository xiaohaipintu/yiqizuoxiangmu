"""第二步：把碎片拼回去（v3：快速最优优先搜索）。

核心观察：碎片是同一张纸直线剪开的，两块要贴合就必须有一条等长的边；
拼成整体后「内部接缝总长 = (总周长 - 外轮廓周长)/2」。
于是「最大化贴合边长」≈「最小化外轮廓周长」，正是拼装要最大化的目标。

搜索：从最大碎片出发，把未用碎片贴到已拼簇的某条边上（旋转对齐 + 可翻面 + 可沿边滑动），
重叠面积超阈值直接剔除，用优先队列做全局最优优先；7 块全部放下即为完整解。
性能：重叠判定用 numpy 半平面裁剪，空穴用顶点凸包近似，避免 shapely 布尔运算进入热循环。

用法:
    python src/solve.py --pieces out/pieces.json --out out
"""
from __future__ import annotations

import argparse
import heapq
import itertools
import json
import math
import pathlib
import sys
import time

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import geometry as G  # noqa: E402

LEN_TOL = 0.15       # 边长相对容差（拍照透视 + 圆角会让长度有百分之几的误差）
OVERLAP_TOL = 0.010  # 允许的相对重叠面积（噪声）
SLIDES = (0.0, 0.5, 1.0)


class Piece:
    def __init__(self, d: dict):
        self.id = int(d["id"])
        self.poly = np.array(d["polygon"], dtype=float)
        self.area = float(d["area"])
        self.peri = float(d["perimeter"])
        self.edges = list(d["edges"])
        self.n = len(self.poly)


def world_poly(poly: np.ndarray, deg: float, t, mirror: bool) -> np.ndarray:
    return G.transform_poly(poly, G.rotation(deg), np.asarray(t, float), mirror)


def placement(piece: Piece, edge_idx: int, ta, tb, mirror: bool, slide: float, flip: bool):
    """把 piece 的第 edge_idx 条边贴到目标边 (ta, tb)，返回 (旋转角, 平移)。"""
    e = piece.edges[edge_idx]
    a = np.array(e["a"], float)
    b = np.array(e["b"], float)
    L = float(e["length"])
    if mirror:
        a = np.array([-a[0], a[1]])
        b = np.array([-b[0], b[1]])
    ds = b - a
    ds /= np.linalg.norm(ds) + 1e-12
    tab = np.asarray(tb, float) - np.asarray(ta, float)
    La = float(np.linalg.norm(tab))
    dt = tab / (La + 1e-12)

    if flip:
        want, src, dst = dt, a, np.asarray(ta, float)
    else:
        want, src, dst = -dt, a, np.asarray(tb, float)

    ang = math.degrees(math.atan2(want[1], want[0]) - math.atan2(ds[1], ds[0])) % 360.0
    t = dst - G.rotation(ang) @ src
    shift = max(0.0, La - L) * slide
    if shift:
        t = t + dt * shift
    return ang, t


class State:
    __slots__ = ("placed", "polys", "matched", "holes", "overlap", "sum_area", "sum_peri", "prio", "cov")

    def __init__(self, placed, polys, matched, holes, overlap, sum_area, sum_peri, cov):
        self.placed = placed      # [(id, deg, tx, ty, mirror)]
        self.polys = polys        # [ndarray]
        self.matched = matched
        self.holes = holes
        self.overlap = overlap
        self.sum_area = sum_area
        self.sum_peri = sum_peri
        self.cov = cov            # {(pid, ei): 已贴合比例}
        self.prio = (
            matched
            - 2.0 * (holes / max(sum_area, 1e-9)) * sum_peri
            - 6.0 * (overlap / max(sum_area, 1e-9)) * sum_peri
        )


def metrics(placed, polys, matched, sum_area, sum_peri, overlap, cov):
    pts = np.vstack(polys)
    hull = G.hull_area(pts)
    holes = max(0.0, hull - (sum_area - overlap))
    return State(placed, polys, matched, holes, overlap, sum_area, sum_peri, cov)


def spatial_key(placed, pieces, nd=3):
    parts = []
    for pid, deg, tx, ty, mir in sorted(placed):
        w = world_poly(pieces[pid].poly, deg, (tx, ty), mir)
        c = G.centroid(w)
        parts.append(f"{pid}_{round(c[0], nd)}_{round(c[1], nd)}_{round(deg, 1)}_{int(mir)}")
    return "|".join(parts)


class Solver:
    def __init__(self, pieces, seed=None, max_nodes=6000, max_seconds=240.0, verbose=True):
        self.pieces = {p.id: p for p in pieces}
        self.seed = seed or max(self.pieces, key=lambda i: self.pieces[i].area)
        self.max_nodes = max_nodes
        self.max_seconds = max_seconds
        self.verbose = verbose
        self.log = []
        self.nodes = 0
        self.complete = []
        self.seen = set()
        self.cands = 0

    def _say(self, msg):
        self.log.append(msg)
        if self.verbose:
            print(msg, flush=True)

    def initial(self) -> State:
        p = self.pieces[self.seed]
        poly = world_poly(p.poly, 0.0, (0.0, 0.0), False)
        return metrics([(self.seed, 0.0, 0.0, 0.0, False)], [poly], 0.0, p.area, p.peri, 0.0, {})

    def expand(self, st: State):
        used = {pid for pid, *_ in st.placed}
        out = []
        cluster_pts = np.vstack(st.polys)
        cb = np.array([cluster_pts[:, 0].min(), cluster_pts[:, 1].min(),
                       cluster_pts[:, 0].max(), cluster_pts[:, 1].max()])
        for k, (pid, deg, tx, ty, mir) in enumerate(st.placed):
            pw = st.polys[k]
            p = self.pieces[pid]
            for ei in range(p.n):
                if st.cov.get((pid, ei), 0.0) >= 0.92:   # 这条边已经贴满
                    continue
                aw = pw[ei]
                bw = pw[(ei + 1) % p.n]
                La = float(np.linalg.norm(bw - aw))
                if La < 1e-6:
                    continue
                for qid, q in self.pieces.items():
                    if qid in used:
                        continue
                    for fi in range(len(q.edges)):
                        Lb = float(q.edges[fi]["length"])
                        if abs(Lb - La) / max(La, 1e-9) > LEN_TOL:
                            continue
                        for mirq in (False, True):
                            for flip in (False, True):
                                for sl in SLIDES:
                                    ang, t = placement(q, fi, aw, bw, mirq, sl, flip)
                                    wq = world_poly(q.poly, ang, t, mirq)
                                    qb = np.array([wq[:, 0].min(), wq[:, 1].min(),
                                                   wq[:, 0].max(), wq[:, 1].max()])
                                    if qb[0] > cb[2] + 400 or qb[2] < cb[0] - 400 or \
                                       qb[1] > cb[3] + 400 or qb[3] < cb[1] - 400:
                                        continue
                                    ov = 0.0
                                    bad = False
                                    for sp in st.polys:
                                        sb = np.array([sp[:, 0].min(), sp[:, 1].min(),
                                                       sp[:, 0].max(), sp[:, 1].max()])
                                        if qb[0] >= sb[2] or qb[2] <= sb[0] or qb[1] >= sb[3] or qb[3] <= sb[1]:
                                            continue
                                        a = G.convex_overlap_area(wq, sp)
                                        if a > 0.0:
                                            ov += a
                                            if ov > OVERLAP_TOL * q.area:
                                                bad = True
                                                break
                                    self.cands += 1
                                    if bad:
                                        continue
                                    glued = min(La, Lb)
                                    cov = dict(st.cov)
                                    cov[(pid, ei)] = cov.get((pid, ei), 0.0) + glued / La
                                    cov[(qid, fi)] = glued / Lb
                                    new = metrics(
                                        st.placed + [(qid, ang, float(t[0]), float(t[1]), mirq)],
                                        st.polys + [wq],
                                        st.matched + glued,
                                        st.sum_area + q.area,
                                        st.sum_peri + q.peri,
                                        st.overlap + ov,
                                        cov,
                                    )
                                    out.append(new)
        return out

    def run(self):
        t0 = time.time()
        counter = itertools.count()
        start = self.initial()
        heap = [(-start.prio, next(counter), start)]
        self.seen.add(spatial_key(start.placed, self.pieces))
        self._say(f"种子碎片 #{self.seed}，共 {len(self.pieces)} 块")
        last = t0
        while heap:
            _, _, st = heapq.heappop(heap)
            if time.time() - t0 > self.max_seconds or self.nodes > self.max_nodes:
                self._say(f"达到预算：节点 {self.nodes}，用时 {time.time() - t0:.0f}s")
                break
            if len(st.placed) == len(self.pieces):
                self.complete.append(st)
                if len(self.complete) >= 30:
                    self._say(f"已收集 {len(self.complete)} 个完整解，停止")
                    break
                continue
            self.nodes += 1
            for child in self.expand(st):
                if len(child.placed) == len(self.pieces):
                    self.complete.append(child)
                    continue
                k = spatial_key(child.placed, self.pieces)
                if k in self.seen:
                    continue
                self.seen.add(k)
                heapq.heappush(heap, (-child.prio, next(counter), child))
            if self.nodes % 100 == 0:
                now = time.time()
                self._say(
                    f"节点 {self.nodes} | 队列 {len(heap)} | 完整解 {len(self.complete)} | "
                    f"评估候选 {self.cands} | {100.0 / (now - last):.0f} 节点/秒 | 用时 {now - t0:.0f}s"
                )
                last = now
        self._say(f"搜索结束：节点 {self.nodes}，候选 {self.cands}，完整解 {len(self.complete)}，"
                  f"用时 {time.time() - t0:.0f}s")
        return self.complete


def final_rank(st: State, pieces, square_area: float):
    """完整解排序：外轮廓越短、越方、空穴越少越好。"""
    union = unary_union([Polygon(p) for p in st.polys])
    hull = union.convex_hull
    holes = max(0.0, hull.area - union.area)
    coords = np.array(hull.exterior.coords)[:-1]
    nvert = len(G.drop_collinear(coords, tol_deg=5.0))
    square_bonus = abs(hull.area - square_area) / square_area
    return (round(square_bonus, 2), nvert > 4, union.length + 8.0 * holes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pieces", default="out/pieces.json")
    ap.add_argument("--out", default="out")
    ap.add_argument("--max-nodes", type=int, default=6000)
    ap.add_argument("--max-seconds", type=float, default=240.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = json.loads(pathlib.Path(args.pieces).read_text(encoding="utf-8"))
    pieces = [Piece(d) for d in data["pieces"]]
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    sum_area = sum(p.area for p in pieces)

    solver = Solver(pieces, seed=args.seed or None, max_nodes=args.max_nodes, max_seconds=args.max_seconds)
    complete = solver.run()
    (outdir / "solve.log").write_text("\n".join(solver.log) + "\n", encoding="utf-8")
    if not complete:
        print("没有找到完整解")
        return 1

    ranked = sorted(complete, key=lambda s: final_rank(s, pieces, sum_area))
    best = ranked[0]

    all_pts = np.vstack(best.polys)
    off = np.array([all_pts[:, 0].min(), all_pts[:, 1].min()])
    out = []
    for (pid, deg, tx, ty, mir), w in zip(best.placed, best.polys):
        coords = w - off
        out.append(
            {
                "id": pid,
                "rotate_deg": float(deg % 360.0),
                "mirror": bool(mir),
                "translate": [float(tx - off[0]), float(ty - off[1])],
                "polygon_world": coords.tolist(),
            }
        )
    (outdir / "solution.json").write_text(json.dumps({"pieces": out}, ensure_ascii=False, indent=2), encoding="utf-8")

    # 最终精确校验（用 shapely 重算一遍，和搜索里的近似互相印证）
    polys = [Polygon(p["polygon_world"]) for p in out]
    union = unary_union(polys)
    hull = union.convex_hull
    d = {
        "n_pieces": len(out),
        "sum_area": sum_area,
        "union_area": union.area,
        "overlap_area": sum_area - union.area,
        "hull_area": hull.area,
        "holes_area": hull.area - union.area,
        "hull_perimeter": hull.length,
        "boundary_length": union.length,
        "square_side_equiv": math.sqrt(sum_area),
        "hull_vertex_count": len(G.drop_collinear(np.array(hull.exterior.coords)[:-1], tol_deg=5.0)),
        "complete_candidates": len(complete),
    }
    print("=== 校验 ===")
    for k, v in d.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")
    (outdir / "solve_report.json").write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
