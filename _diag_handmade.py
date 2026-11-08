"""临时诊断：比较两个解在「纹理连续性」和「笔迹连续性」上的表现。用完即删。"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "拼图还原项目" / "src"))
sys.path.insert(0, str(ROOT / "纹理拼合" / "tools"))

import geometry as G
from irregular_cut import _texture_cost

src = cv2.imdecode(np.fromfile("手撕拼图.jpg", dtype=np.uint8), cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB).astype(np.float32)

# 笔迹掩膜：蓝色通道明显高于红/绿
b, g, r = src[..., 0].astype(np.int16), src[..., 1].astype(np.int16), src[..., 2].astype(np.int16)
ink = ((b.astype(np.int16) - np.maximum(g, r)) > 25).astype(np.float32)


def seam_pairs(sol):
    """返回所有相邻块对：(i, j, aw, bw, side)，world 坐标。"""
    polys = [np.asarray(p, float) for p in sol["polys"]]
    poses = [(float(it[1]), float(it[2]), float(it[3])) for it in sol["placed"]]
    pairs = []
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            from shapely.geometry import LineString, Polygon as SPoly
            inter = SPoly(polys[i]).buffer(0).intersection(SPoly(polys[j]).buffer(0))
            if inter.is_empty:
                continue
            seg = inter if inter.geom_type == "LineString" else getattr(inter, "geoms", [None])[0]
            if seg is None or seg.geom_type != "LineString" or seg.length < 20:
                continue
            coords = list(seg.coords)
            aw, bw = np.array(coords[0]), np.array(coords[-1])
            u = (bw - aw) / (np.linalg.norm(bw - aw) + 1e-12)
            ca = polys[i].mean(axis=0)
            sa = u[0] * (ca[1] - aw[1]) - u[1] * (ca[0] - aw[0])
            pairs.append((i, j, aw, bw, 1.0 if sa > 0 else -1.0, poses[i], poses[j]))
    return pairs


def ink_continuity(pairs):
    """沿每条缝采样，看两侧的"是不是笔迹"是否一致。不一致的比例越低越好。"""
    bad, total = 0, 0
    for i, j, aw, bw, side, pa, pb in pairs:
        u = (bw - aw) / (np.linalg.norm(bw - aw) + 1e-12)
        n = np.array([-u[1], u[0]]) * side
        ts = np.linspace(0.1, 0.9, 40)[:, None]
        base = aw[None, :] + (bw - aw)[None, :] * ts

        def to_orig(world_pts, pose):
            ang, tx, ty = pose
            return (G.rotation(-ang) @ (world_pts - np.array([tx, ty])).T).T

        def samp(pts):
            h, w = ink.shape
            x = np.clip(pts[:, 0], 0, w - 1.001)
            y = np.clip(pts[:, 1], 0, h - 1.001)
            x0 = np.floor(x).astype(int)
            y0 = np.floor(y).astype(int)
            return ink[y0, x0]

        ia = samp(to_orig(base + n * 0.4, pa))
        ib = samp(to_orig(base - n * 0.4, pb))
        bad += int(np.sum(ia != ib))
        total += len(ia)
    return bad, total


for name in ("solution_纯形状.json", "solution_形状加纹理.json"):
    p = Path("纹理拼合/手撕拼图") / name
    if not p.exists():
        continue
    sol = json.loads(p.read_text(encoding="utf-8"))
    pairs = seam_pairs(sol)
    tex = 0.0
    for i, j, aw, bw, side, pa, pb in pairs:
        tex += _texture_cost(rgb, pa, pb, aw, bw, side)
    bad, total = ink_continuity(pairs)
    print("%-22s 相邻块对=%d  纹理代价合计=%.3f（均值 %.3f）  笔迹不连续 %d/%d = %.1f%%"
          % (name, len(pairs), tex, tex / max(1, len(pairs)), bad, total,
             100.0 * bad / max(1, total)))
