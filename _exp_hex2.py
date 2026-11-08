"""临时实验 2：六边形平铺 + 更大的 beam，看纹理能否救回形状分不出的解。用完即删。"""
import sys
from pathlib import Path

import numpy as np
import cv2
from shapely.geometry import Polygon as SPoly, box as sbox

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "纹理拼合" / "tools"))
sys.path.insert(0, str(ROOT / "拼图还原项目" / "src"))

from assemble import Piece
from irregular_cut import assemble, to_pieces_json, verify, _load_square

SIZE = 600


def hex_cut(size, r):
    dx, dy = r * np.sqrt(3.0), 1.5 * r
    square = sbox(0, 0, size, size)
    cells, col, x = [], 0, 0.0
    while x < size + dx:
        y = 0.0 if col % 2 == 0 else dy
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


name = "001.jpg"
path = Path("纹理拼合/照片") / name
cells = hex_cut(SIZE, 118.0)
data = to_pieces_json(cells)
pieces = {int(d["id"]): Piece(d) for d in data["pieces"]}
img = _load_square(path, SIZE)
print("%s：%d 块（正六边形，边长几乎全相等）" % (name, len(pieces)))

for beam in (80, 400, 1500):
    for tag, tw in (("纯形状", 0.0), ("形状+纹理", 1.0)):
        sol = assemble(pieces, beam=beam, time_budget=90.0,
                       img=(img if tw else None), tw=tw)
        if sol is None:
            print("  beam=%-5d %-9s 没有完整拼法" % (beam, tag))
            continue
        v = verify(cells, {k: sol[k] for k in ("placed", "polys")})
        print("  beam=%-5d %-9s 质心误差 %.4f 空隙 %5.2f%% 代价 %.3f  %s"
              % (beam, tag, v["max_rel_dist_err"], sol["gap_pct"], sol["seam"],
                 "✓ 正确" if v["ok"] else "✗ 拼错"))
