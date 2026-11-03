"""检查 v2 的边匹配：真搭档排第几？代价分布如何？"""
import sys, pathlib
import numpy as np

HERE = pathlib.Path(__file__).resolve()
PROJ = HERE.parents[2]
sys.path.insert(0, str(PROJ / "src"))
sys.path.insert(0, str(PROJ / "svg_tool" / "tests"))

import jigsaw_poly_solve as J
import jigsaw_solve_v2 as V2
import bench_solve as B

S = 760
NOISE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
tex = B.load_tex("", S)
polys, cells, gt_pose, scat, side = B.build_case(N, S, tex, 7, NOISE)
polys = [p if J._signed_area(p) > 0 else np.asarray(p, float)[::-1] for p in polys]

G = V2.build_edges(polys, K=14)
E = G["E"]; PID, L = G["pid"], G["L"]
P0, P1 = G["p0"], G["p1"]


def to_orig(pts, pid):
    Rg, tg = gt_pose[pid]
    return (pts - tg) @ Rg


truth = {}
for i in range(E):
    a0, a1 = to_orig(P0[i], PID[i]), to_orig(P1[i], PID[i])
    best, bj = 1e9, -1
    for j in range(E):
        if PID[j] == PID[i]:
            continue
        b0, b1 = to_orig(P0[j], PID[j]), to_orig(P1[j], PID[j])
        d = min(np.linalg.norm(a0 - b1) + np.linalg.norm(a1 - b0),
                np.linalg.norm(a0 - b0) + np.linalg.norm(a1 - b1))
        if d < best:
            best, bj = d, j
    truth[i] = (bj, best)

inner = [i for i in range(E) if truth[i][1] < 3.0 + NOISE * 2]
print(f"σ={NOISE}  边 {E}  内部边(有真搭档) {len(inner)}")

G2, bestmap, jp, jr = V2.edge_matches(polys, scat, topk=8)
per = {i: [j for _, j in sorted([(v[0], j) for j, v in bestmap[i].items()])] for i in range(E)}

rank_hist = {}
tcost, bcost = [], []
for i in inner:
    tj = truth[i][0]
    lst = per[i]
    r = lst.index(tj) if tj in lst else -1
    rank_hist[r] = rank_hist.get(r, 0) + 1
    if r >= 0:
        tcost.append(bestmap[i][tj][0])
    if lst:
        bcost.append(bestmap[i][lst[0]][0])
print("真搭档排名分布:", dict(sorted(rank_hist.items())))
print(f"真搭档代价 中位数 {np.median(tcost):.1f} / 最好候选代价 中位数 {np.median(bcost):.1f}")
