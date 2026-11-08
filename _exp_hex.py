"""临时实验：等边六边形平铺（所有边长几乎相同）—— 形状信息量接近 0，
看纹理能不能救。对比「纯形状」vs「形状+纹理」。用完即删。
"""
import sys
from pathlib import Path

import numpy as np
import cv2
from shapely.geometry import Polygon as SPoly, box as sbox
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "纹理拼合" / "tools"))
sys.path.insert(0, str(ROOT / "拼图还原项目" / "src"))

from assemble import Piece
from irregular_cut import (assemble, to_pieces_json, verify, _load_square)

SIZE = 600


def hex_cut(size: int, r: float = 80.0) -> list[np.ndarray]:
    """正六边形平铺（尖顶朝上），裁剪到正方形内。"""
    dx = r * np.sqrt(3.0)          # 水平间距
    dy = 1.5 * r                   # 垂直间距
    square = sbox(0, 0, size, size)
    cells = []
    col = 0
    x = 0.0
    while x < size + dx:
        y0 = 0.0 if col % 2 == 0 else dy
        y = y0
        while y < size + dy:
            pts = [(x + r * np.cos(np.pi / 2 + k * np.pi / 3),
                    y + r * np.sin(np.pi / 2 + k * np.pi / 3)) for k in range(6)]
            cell = SPoly(pts).buffer(0).intersection(square)
            if not cell.is_empty and cell.area > 2500:
                if cell.geom_type == "MultiPolygon":
                    cell = max(cell.geoms, key=lambda g: g.area)
                if cell.geom_type == "Polygon" and len(cell.exterior.coords) >= 4:
                    cells.append(np.asarray(cell.exterior.coords[:-1], float))
            y += 2 * dy
        x += dx
        col += 1
    return cells


def edge_stats(pieces):
    L = [e["length"] for p in pieces.values() for e in p.edges]
    return np.min(L), np.median(L), np.max(L), len(L)


for name in ("001.jpg", "002.jpg", "003.jpg"):
    path = Path("纹理拼合/照片") / name
    cells = hex_cut(SIZE, r=118.0)
    data = to_pieces_json(cells)
    pieces = {int(d["id"]): Piece(d) for d in data["pieces"]}
    mn, med, mx, n = edge_stats(pieces)
    img = _load_square(path, SIZE)

    print("%s：%d 块，边长 min=%.1f 中位=%.1f max=%.1f（几乎相同 → 形状无信息）"
          % (name, len(pieces), mn, med, mx))

    for tag, tw in (("纯形状", 0.0), ("形状+纹理", 1.0)):
        sol = assemble(pieces, beam=80, time_budget=60.0,
                       img=(img if tw else None), tw=tw)
        if sol is None:
            print("   %-9s 没有完整拼法" % tag)
            continue
        verd = verify(cells, {k: sol[k] for k in ("placed", "polys")})
        print("   %-9s 质心误差 %.4f  空隙 %.2f%%  seam %.3f  %s"
              % (tag, verd["max_rel_dist_err"], sol["gap_pct"], sol["seam"],
                 "✓ 正确" if verd["ok"] else "✗ 拼错"))
    print()
