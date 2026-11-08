"""临时诊断：真邻接 vs 假配对 的纹理代价到底有没有区分度。用完即删。"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "纹理拼合" / "tools"))
sys.path.insert(0, str(ROOT / "拼图还原项目" / "src"))

import geometry as G
from assemble import Piece, place_edge, world
from irregular_cut import _load_square, _texture_cost, to_pieces_json, voronoi_cut

size = 600
img_path = Path("纹理拼合/照片/001.jpg")
rng = np.random.default_rng(1)
cells = voronoi_cut(size, 6, rng)
data = to_pieces_json(cells)
pieces = {int(d["id"]): Piece(d) for d in data["pieces"]}
img = _load_square(img_path, size)

# 真邻接边对：两条边端点重合（碎片都在原图坐标，位姿 = identity）
edges = []
for pid, p in pieces.items():
    for ei, e in enumerate(p.edges):
        edges.append((pid, ei, np.array(e["a"], float), np.array(e["b"], float), e["length"]))


def endpoints_match(a, b, tol=0.6):
    return (min(np.linalg.norm(a[2] - b[2]), np.linalg.norm(a[2] - b[3])) < tol and
            min(np.linalg.norm(a[3] - b[2]), np.linalg.norm(a[3] - b[3])) < tol)


true_pairs = []
for i in range(len(edges)):
    for j in range(i + 1, len(edges)):
        A, B = edges[i], edges[j]
        if A[0] == B[0]:
            continue
        if endpoints_match(A, B):
            true_pairs.append((A, B))

print("真邻接边对：", len(true_pairs))
print()
print("真邻接：纹理代价（理想应接近 0）")
for A, B in true_pairs:
    # A 在原位（identity），把 B 的这条边贴到 A 的这条边上
    ang, t = place_edge(pieces[B[0]], B[1], A[2], A[3], True)
    u = (A[3] - A[2]) / np.linalg.norm(A[3] - A[2])
    ca = pieces[A[0]].poly.mean(axis=0)
    sa = u[0] * (ca[1] - A[2][1]) - u[1] * (ca[0] - A[2][0])
    cost = _texture_cost(img, (0.0, 0.0, 0.0), (ang, float(t[0]), float(t[1])),
                         A[2], A[3], 1.0 if sa > 0 else -1.0)
    d = np.linalg.norm((G.rotation(ang) @ pieces[B[0]].poly.T).T + t - pieces[B[0]].poly, axis=1).max()
    print("  P%d e%d  ↔ P%d e%d   纹理代价 %.4f   （贴合后与原位姿偏差 %.2f px）"
          % (A[0], A[1], B[0], B[1], cost, d))

print()
print("假配对：长度差 <3% 但并非真邻接的边对")
cnt = 0
for i in range(len(edges)):
    for j in range(i + 1, len(edges)):
        A, B = edges[i], edges[j]
        if A[0] == B[0] or endpoints_match(A, B):
            continue
        r = abs(A[4] - B[4]) / max(A[4], B[4])
        if r >= 0.03:
            continue
        ang, t = place_edge(pieces[B[0]], B[1], A[2], A[3], True)
        u = (A[3] - A[2]) / np.linalg.norm(A[3] - A[2])
        ca = pieces[A[0]].poly.mean(axis=0)
        sa = u[0] * (ca[1] - A[2][1]) - u[1] * (ca[0] - A[2][0])
        cost = _texture_cost(img, (0.0, 0.0, 0.0), (ang, float(t[0]), float(t[1])),
                             A[2], A[3], 1.0 if sa > 0 else -1.0)
        print("  P%d e%d  ↔ P%d e%d   长度差 %.2f%%   纹理代价 %.4f"
              % (A[0], A[1], B[0], B[1], r * 100, cost))
        cnt += 1
print("假配对数：", cnt)
