"""临时诊断：在「笔迹连续性已知很好」的接缝上扫 k，看哪个 k 能给出低成本。用完即删。"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "拼图还原项目" / "src"))
sys.path.insert(0, str(ROOT / "纹理拼合" / "tools"))

import geometry as G
import irregular_cut as IC

src = cv2.imdecode(np.fromfile("手撕拼图.jpg", dtype=np.uint8), cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB).astype(np.float32)

sol = json.loads(Path("纹理拼合/手撕拼图/solution_纯形状.json").read_text(encoding="utf-8"))
polys = [np.asarray(p, float) for p in sol["polys"]]
poses = [(float(it[1]), float(it[2]), float(it[3])) for it in sol["placed"]]

from shapely.geometry import Polygon as SPoly

pairs = []
for i in range(len(polys)):
    for j in range(i + 1, len(polys)):
        inter = SPoly(polys[i]).buffer(0).intersection(SPoly(polys[j]).buffer(0))
        if inter.is_empty:
            continue
        seg = inter if inter.geom_type == "LineString" else getattr(inter, "geoms", [None])[0]
        if seg is None or seg.geom_type != "LineString" or seg.length < 20:
            continue
        c = list(seg.coords)
        aw, bw = np.array(c[0]), np.array(c[-1])
        u = (bw - aw) / (np.linalg.norm(bw - aw) + 1e-12)
        ca = polys[i].mean(axis=0)
        sa = u[0] * (ca[1] - aw[1]) - u[1] * (ca[0] - aw[0])
        pairs.append((i, j, aw, bw, 1.0 if sa > 0 else -1.0, poses[i], poses[j],
                      float(np.linalg.norm(bw - aw))))

print("纯形状解里的 %d 条接缝（笔迹连续性 1.2%%，应判为「好」）：" % len(pairs))
for k in (0.35, 1.0, 2.0, 3.0, 5.0):
    costs = []
    for i, j, aw, bw, side, pa, pb, L in pairs:
        costs.append(IC._texture_cost(rgb, pa, pb, aw, bw, side, k=k))
    print("  k=%-4.2f  各缝代价 %s   均值 %.3f"
          % (k, ["%.3f" % c for c in costs], float(np.mean(costs))))

# 顺带看看接缝附近的像素长什么样（判断是不是被撕裂边的暗边带偏了）
i, j, aw, bw, side, pa, pb, L = pairs[0]
u = (bw - aw) / L
n = np.array([-u[1], u[0]]) * side
ts = np.linspace(0.2, 0.8, 6)[:, None]
base = aw[None, :] + (bw - aw)[None, :] * ts


def to_orig(world_pts, pose):
    ang, tx, ty = pose
    return (G.rotation(-ang) @ (world_pts - np.array([tx, ty])).T).T


print()
print("第一条缝的取样剖面（跨缝方向的灰度，负=进 A，正=进 B）：")
h, w = src.shape[:2]
for off in (-4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4):
    pts = to_orig(base + n * off, pa)
    x = np.clip(pts[:, 0], 0, w - 1).astype(int)
    y = np.clip(pts[:, 1], 0, h - 1).astype(int)
    vals = src[y, x]
    print("  off=%+5.1f  BGR 均值 %s" % (off, np.round(vals.mean(axis=0), 1)))
