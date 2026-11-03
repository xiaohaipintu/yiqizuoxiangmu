"""只喂真搭档 join，逐步定位 assemble / relax 的错。"""
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
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
NOISE = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
tex = B.load_tex("", S)
polys, cells, gt_pose, scat, side = B.build_case(N, S, tex, 7, NOISE)
polys = [p if J._signed_area(p) > 0 else np.asarray(p, float)[::-1] for p in polys]

G = V2.build_edges(polys, K=14)
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
badjoin = 0
for i in range(E):
    j, d = truth[i]
    if d > 3.0 + NOISE * 2:
        continue
    R = V2.rot(G["th_fwd"][i] - G["th_rev"][j])
    t = P0[i] - R @ P1[j]
    bmap[i][j] = (0.0, R, t)
    joins.append((0.0, i, j))
    # 单条 join 自检：以 A 的真值位姿推 B，应该等于 B 的真值位姿
    A, Bp = int(PID[i]), int(PID[j])
    RA, tA = truthpose[A]
    R_B = RA @ R
    t_B = RA @ t + tA
    RB, tB = truthpose[Bp]
    da = np.degrees(np.arctan2((RB.T @ R_B)[1, 0], (RB.T @ R_B)[0, 0]))
    dt = np.linalg.norm(t_B - tB)
    if abs(da) > 0.5 or dt > 2.0:
        badjoin += 1
        if badjoin <= 6:
            print(f"  坏 join 片{A}-片{Bp}(边{i}-{j}) 角差 {da:7.2f}° 位差 {dt:6.1f}")

print(f"真搭档 join {len(joins)} 条，其中 {badjoin} 条的相对变换推不出正确位姿")

pose, placed, nbr = V2.assemble(polys, G, bmap, joins, overlap_slack=3.0)
print("assemble:", B.score(polys, cells, gt_pose, list(pose), side))

p2, pl2 = V2.relax(polys, list(truthpose), {}, nbr, iters=3)
print("relax(从真值出发):", B.score(polys, cells, gt_pose, list(p2), side))

p3, pl3 = V2.relax(polys, pose, placed, nbr, iters=3)
print("relax(从 assemble 结果):", B.score(polys, cells, gt_pose, list(p3), side))
