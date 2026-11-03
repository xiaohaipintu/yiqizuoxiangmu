"""bench_solve 的单元自检：喂真值位姿，拼对率必须是 100%。"""
import sys, pathlib
import numpy as np
import cv2

HERE = pathlib.Path(__file__).resolve()
PROJ = HERE.parents[2]
sys.path.insert(0, str(PROJ / "src"))
sys.path.insert(0, str(PROJ / "svg_tool" / "tests"))

import jigsaw_poly_solve as J
import bench_solve as B
import jigsaw_solve_v2 as V2

S = 760
tex = B.load_tex("", S)
polys, cells, gt_pose, scat, side = B.build_case(20, S, tex, 7, 0.0)

# 真值 pose（散落 → 原图）
truth = []
for (Rg, tg) in gt_pose:
    Rt = Rg.T
    truth.append((Rt, -Rt @ tg))
mon = B.score(polys, cells, gt_pose, truth, side)
print("真值位姿自检:", mon)
assert mon["rate"] == 1.0, "真值都判不对 → 打分器有 bug"

# 求解器位姿 vs 真值位姿的逐对相对误差
pose, placed, pp = V2.solve(polys, scat, verbose=True)
ok = [k for k in range(len(polys)) if pose[k] is not None]
print("placed", placed, "of", len(polys))
# 用真值做参照，算每片相对「锚 0」的变换，跟真值比
a = ok[0]
Ra_t, ta_t = truth[a]
na, nb = None, None
for k in ok[:6]:
    Rk, tk = pose[k]
    # 预测: world = Rk x + tk ；真值: orig = Rt x + tt
    # 相对变换（预测世界 → 原图）= truth[k] ∘ pred[k]^-1
    G = (Ra_t @ Rk.T, ta_t - Ra_t @ Rk.T @ tk)
    ang = np.degrees(np.arctan2(G[0][1, 0], G[0][0, 0]))
    print(f"  piece {k:2d} 相对锚 {a} 的角度偏差 {ang:8.2f}°  平移 {np.round(G[1],1)}")
