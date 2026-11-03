"""最小探针：6 片、真搭档 join，逐片看投票与落子对不对。"""
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
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
tex = B.load_tex("", S)
polys, cells, gt_pose, scat, side = B.build_case(N, S, tex, 7, 0.0)
polys = [p if J._signed_area(p) > 0 else np.asarray(p, float)[::-1] for p in polys]

G = V2.build_edges(polys, K=20)
E = G["E"]; PID = G["pid"]; P0, P1 = G["p0"], G["p1"]


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

truthpose = []
for (Rg, tg) in gt_pose:
    Rt = Rg.T
    truthpose.append((Rt, -Rt @ tg))

bmap = [dict() for _ in range(E)]
joins = []
for i in range(E):
    j, d = truth[i]
    if d > 3.0:
        continue
    R = V2.rot(G["th_fwd"][i] - G["th_rev"][j])
    t = P0[i] - R @ P1[j]
    bmap[i][j] = (0.0, R, t)
    joins.append((0.0, i, j))

print("E", E, "pieces", len(polys), "joins", len(joins))
pose, placed, nbr = V2.assemble(polys, G, bmap, joins, overlap_slack=3.0)
print("placed", len(placed))
# 全局相对锚 0 的角度
a = 0
if pose[a] is not None:
    Ra, ta = truthpose[a]
    for k in range(len(polys)):
        if pose[k] is None:
            print(f"  {k}: None"); continue
        Rk, tk = pose[k]
        G_ = (Ra @ Rk.T, ta - Ra @ Rk.T @ tk)
        ang = np.degrees(np.arctan2(G_[0][1, 0], G_[0][0, 0]))
        print(f"  {k}: 角度偏差 {ang:8.2f}°  平移 {np.round(G_[1],1)}")
print("score:", B.score(polys, cells, gt_pose, list(pose), side))
