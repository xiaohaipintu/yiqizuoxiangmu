"""打乱碎片顺序 + 纹理检测强制开启，验证求解器不靠答案。"""
import sys
import numpy as np
import cv2

sys.path.insert(0, "拼图还原项目/src")
import jigsaw_poly_solve as J

S = 760
cells = J.make_cells(60, S, 7)
polys, poses = J.scatter(cells, int(S * 1.8), 7)
tex = J.read_color("拼图还原项目/svg_tool/test_out/letter_h5.png")

scat_side = int(S * 1.8)
scat = np.zeros((scat_side, scat_side, 3), np.uint8)
for idx, (theta, center, cen) in enumerate(poses):
    cell = cells[idx]; R = J.rot(theta)
    M = np.array([[R[0, 0], R[0, 1], center[0] - (R @ cen)[0]],
                  [R[1, 0], R[1, 1], center[1] - (R @ cen)[1]]], float)
    warped = cv2.warpAffine(tex, M, (scat_side, scat_side), flags=cv2.INTER_LINEAR)
    mask = np.zeros((S, S), np.uint8)
    cv2.fillPoly(mask, [np.round(cell).astype(np.int32)], 255)
    wmask = cv2.warpAffine(mask, M, (scat_side, scat_side), flags=cv2.INTER_NEAREST)
    scat = np.where((wmask > 0)[..., None], warped, scat)

rng = np.random.default_rng(0)
order = rng.permutation(len(polys))
polys2 = [polys[i] for i in order]

pose, placed, pp = J.solve(polys2, scat=scat)   # 打乱顺序；纹理检测开启(不可关)
m = J.evaluate(pp, S)
print(f"打乱顺序 + 纹理检测开启: placed {placed}/{len(polys2)}  hull_frac {m['hull_frac']:.4f}")
