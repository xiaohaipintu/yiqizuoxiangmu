"""只用「等长边 + 整条覆盖」做闭环拼装搜索（有节点/时间上限，找不到就如实报告）。

用法:
    python src/closure_search.py --pieces out3/pieces.json --out out3 --tol 0.025 --max-nodes 3000000
输出:
    out3/solutions.json   找到的完整拼法（含空穴率）
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


def hull_gap_pct(polys):
    pts = np.vstack(polys).astype(np.float32)
    hull = abs(float(cv2.contourArea(cv2.convexHull(pts))))
    s = sum(G.polygon_area(p) for p in polys)
    return max(0.0, hull - s) / s * 100.0


def edge_world(deg, t, p):
    return (G.rotation(deg) @ np.asarray(p, float)) + np.asarray(t, float)


class Search:
    def __init__(self, pieces, tol=0.025, cov=0.92, overlap_tol=0.004,
                 max_nodes=3_000_000, max_solutions=200, time_budget=120.0):
        self.pieces = {p.id: p for p in pieces}
        self.tol = tol
        self.cov = cov
        self.overlap_tol = overlap_tol
        self.max_nodes = max_nodes
        self.max_solutions = max_solutions
        self.deadline = time.time() + time_budget
        self.nodes = 0
        self.solutions = []
        self.best_n = 0
        self.stopped = False

    def joins(self, qid, pid, pw, used_edges, polys):
        """把 qid 贴到 pid 的一条空闲边上（等长 + 整条覆盖 + 分居两侧 + 不重叠）。"""
        q, p = self.pieces[qid], self.pieces[pid]
        out = []
        for ei in range(p.n):
            if ei in used_edges:
                continue
            aw, bw = pw[ei], pw[(ei + 1) % p.n]
            La = float(np.linalg.norm(bw - aw))
            if La < 1e-6:
                continue
            u = (bw - aw) / La
            for fi in range(q.n):
                eb = q.edges[fi]
                Lb = float(eb["length"])
                if abs(La - Lb) / max(La, Lb) > self.tol:
                    continue
                ang, t = place_edge(q, fi, aw, bw, True)
                wq = world(q, ang, t)
                qaw = edge_world(ang, t, eb["a"])
                qbw = edge_world(ang, t, eb["b"])
                Lq = float(np.linalg.norm(qbw - qaw))
                if Lq < 1e-6:
                    continue
                proj = np.clip([((qaw - aw) @ u), ((qbw - aw) @ u)], 0, La)
                shared = float(max(0.0, proj.max() - proj.min()))
                if shared < self.cov * min(La, Lq):
                    continue
                ca, cq = p.poly.mean(axis=0), wq.mean(axis=0)
                sa = u[0] * (ca[1] - aw[1]) - u[1] * (ca[0] - aw[0])
                sq = u[0] * (cq[1] - aw[1]) - u[1] * (cq[0] - aw[0])
                if sa * sq >= 0:
                    continue
                ov = sum(G.poly_overlap_area(wq, pp) for pp in polys)
                if ov > self.overlap_tol * q.area:
                    continue
                out.append((ei, fi, ang, t, wq, abs(La - Lb) / max(La, Lb)))
        return out

    def run(self):
        seed = max(self.pieces, key=lambda i: self.pieces[i].area)
        polys = [world(self.pieces[seed], 0.0, (0.0, 0.0))]
        used = {pid: set() for pid in self.pieces}
        self._dfs([(seed, 0.0, 0.0, 0.0)], polys, used)
        return self.solutions

    def _dfs(self, placed, polys, used):
        if self.stopped:
            return
        if len(placed) == len(self.pieces):
            self.solutions.append({"placed": placed,
                                   "polys": [p.tolist() for p in polys],
                                   "gap_pct": hull_gap_pct(polys)})
            if len(self.solutions) >= self.max_solutions:
                self.stopped = True
            return
        self.best_n = max(self.best_n, len(placed))
        done = {r[0] for r in placed}
        todo = [q for q in self.pieces if q not in done]
        for k, (pid, _, _, _) in enumerate(placed):
            pw = polys[k]
            for qid in todo:
                cands = self.joins(qid, pid, pw, used[pid], polys)
                for ei, fi, ang, t, wq, _ in cands:
                    self.nodes += 1
                    if self.nodes % 20000 == 0:
                        if self.nodes > self.max_nodes or time.time() > self.deadline:
                            self.stopped = True
                            return
                    if len(placed) >= 6 and hull_gap_pct(polys + [wq]) > 30.0:
                        continue
                    nu = {kk: set(v) for kk, v in used.items()}
                    nu[pid].add(ei)
                    nu[qid] = {fi}
                    self._dfs(placed + [(qid, ang, float(t[0]), float(t[1]))],
                              polys + [wq], nu)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pieces", default="out3/pieces.json")
    ap.add_argument("--out", default="out3")
    ap.add_argument("--tol", type=float, default=0.025)
    ap.add_argument("--max-nodes", type=int, default=3_000_000)
    ap.add_argument("--cov", type=float, default=0.92)
    ap.add_argument("--time", type=float, default=120.0)
    args = ap.parse_args()

    outdir = pathlib.Path(args.out)
    data = json.loads(pathlib.Path(args.pieces).read_text(encoding="utf-8"))
    pieces = [Piece(d) for d in data["pieces"]]
    s = Search(pieces, tol=args.tol, cov=args.cov, max_nodes=args.max_nodes, time_budget=args.time)
    sols = s.run()
    print(f"搜索节点 {s.nodes}，最深拼到 {s.best_n}/{len(pieces)} 块，完整拼法 {len(sols)} 个")
    sols.sort(key=lambda x: x["gap_pct"])
    for i, sol in enumerate(sols[:10], 1):
        print(f"  {i}: 空穴率 {sol['gap_pct']:.2f}%")
    (outdir / "solutions.json").write_text(
        json.dumps({"tol": args.tol, "nodes": s.nodes, "deepest": s.best_n,
                    "n_solutions": len(sols), "solutions": sols}, ensure_ascii=False),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



