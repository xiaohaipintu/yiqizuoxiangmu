"""临时调试：打印真邻接采样点的坐标与颜色。用完即删。"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "纹理拼合" / "tools"))
sys.path.insert(0, str(ROOT / "拼图还原项目" / "src"))

import geometry as G
from assemble import Piece, place_edge
from irregular_cut import _load_square, _sample, to_pieces_json, voronoi_cut

size = 600
img = _load_square(Path("纹理拼合/照片/001.jpg"), size)
rng = np.random.default_rng(1)
data = to_pieces_json(voronoi_cut(size, 6, rng))
pieces = {int(d["id"]): Piece(d) for d in data["pieces"]}

A, B = pieces[1], pieces[3]
ea, eb = 0, 3
Ae = A.edges[ea]
aw = np.array(Ae["a"], float)
bw = np.array(Ae["b"], float)
ang, t = place_edge(B, eb, aw, bw, True)
print("A 的边：", aw, bw, " 长度", Ae["length"])
print("B 贴上去的位姿：ang=%.6f  t=%s" % (ang, t))

# B 贴合后应当回到原位（偏差 0），验证一下
wB = (G.rotation(ang) @ B.poly.T).T + np.asarray(t, float)
print("B 贴合后与原位最大偏差：%.6f px" % np.abs(wB - B.poly).max())

u = (bw - aw) / np.linalg.norm(bw - aw)
n = np.array([-u[1], u[0]], float)
ca = A.poly.mean(axis=0)
sa = u[0] * (ca[1] - aw[1]) - u[1] * (ca[0] - aw[0])
side = 1.0 if sa > 0 else -1.0
print("side =", side, "  A 质心 =", ca)

k = 1.6
ts = np.linspace(0.3, 0.7, 5)[:, None]
base = aw[None, :] + (bw - aw)[None, :] * ts


def to_orig(world_pts, pose):
    a_, tx, ty = pose
    Rm = G.rotation(-a_)
    return (Rm @ (world_pts - np.array([tx, ty], float)).T).T


pa = to_orig(base + n * k * side, (0.0, 0.0, 0.0))
pb = to_orig(base - n * k * side, (ang, float(t[0]), float(t[1])))
print()
print("世界坐标采样点 base:", base[:2])
print("A 侧 → 原图坐标:", pa[:2])
print("B 侧 → 原图坐标:", pb[:2])
print("A 侧颜色:", _sample(img, pa)[:2])
print("B 侧颜色:", _sample(img, pb)[:2])

# 直接用原图坐标验证：A 侧的点和 B 侧的点应该只是跨过接缝的邻居
print()
print("最近的真实像素差（原图里同一位置）：", img[int(pa[0][1]), int(pa[0][0])],
      img[int(pb[0][1]), int(pb[0][0])])
d = _sample(img, pa) - _sample(img, pb)
print("mse =", float(np.mean(np.sum(d * d, axis=1))))
