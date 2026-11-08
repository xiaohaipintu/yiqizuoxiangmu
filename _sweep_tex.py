"""临时实验：扫 k 与基线 epsilon，看纹理代价能否把真邻接/假配对分开。用完即删。"""
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
rng = np.random.default_rng(1)
data = to_pieces_json(voronoi_cut(size, 6, rng))
pieces = {int(d["id"]): Piece(d) for d in data["pieces"]}
img = _load_square(Path("纹理拼合/照片/001.jpg"), size)

edges = []
for pid, p in pieces.items():
    for ei, e in enumerate(p.edges):
        edges.append((pid, ei, np.array(e["a"], float), np.array(e["b"], float), e["length"]))


def endpoints_match(a, b, tol=0.6):
    return (min(np.linalg.norm(a[2] - b[2]), np.linalg.norm(a[2] - b[3])) < tol and
            min(np.linalg.norm(a[3] - b[2]), np.linalg.norm(a[3] - b[3])) < tol)


true_pairs, false_pairs = [], []
for i in range(len(edges)):
    for j in range(i + 1, len(edges)):
        A, B = edges[i], edges[j]
        if A[0] == B[0]:
            continue
        if endpoints_match(A, B):
            true_pairs.append((A, B))
        elif abs(A[4] - B[4]) / max(A[4], B[4]) < 0.03:
            false_pairs.append((A, B))


def cost(A, B, k, eps, mode):
    ang, t = place_edge(pieces[B[0]], B[1], A[2], A[3], True)
    u = (A[3] - A[2]) / np.linalg.norm(A[3] - A[2])
    n = np.array([-u[1], u[0]], float)
    ca = pieces[A[0]].poly.mean(axis=0)
    sa = u[0] * (ca[1] - A[2][1]) - u[1] * (ca[0] - A[2][0])
    side = 1.0 if sa > 0 else -1.0

    ts = np.linspace(0.15, 0.85, 24)[:, None]
    base = A[2][None, :] + (A[3] - A[2])[None, :] * ts
    to = lambda wp, pose: (G.rotation(-pose[0]) @ (wp - np.array(pose[1:], float)).T).T
    pa1 = _sample(img, to(base + n * k * side, (0.0, 0.0, 0.0)))
    pb1 = _sample(img, to(base - n * k * side, (ang, float(t[0]), float(t[1]))))
    sq = lambda a, b: float(np.mean(np.sum((a - b) ** 2, axis=1)))
    cross = sq(pa1, pb1)
    if mode == "abs":
        ratio = cross
    else:
        pa2 = _sample(img, to(base + n * 2 * k * side, (0.0, 0.0, 0.0)))
        pb2 = _sample(img, to(base - n * 2 * k * side, (ang, float(t[0]), float(t[1]))))
        base_e = 0.5 * (sq(pa1, pa2) + sq(pb1, pb2)) + eps
        ratio = cross / base_e
    return ratio


for mode in ("abs", "norm"):
    for k in (0.35, 0.6, 1.0, 1.6):
        for eps in (1.0, 25.0):
            if mode == "abs" and eps != 1.0:
                continue
            tr = [cost(A, B, k, eps, mode) for A, B in true_pairs]
            fa = [cost(A, B, k, eps, mode) for A, B in false_pairs]
            sep = (min(fa) / max(tr)) if max(tr) > 0 else float("inf")
            print("mode=%-4s k=%.2f eps=%-5.0f  真邻接 max=%8.1f 中位=%8.1f | "
                  "假配对 min=%8.1f 中位=%8.1f | 分离度 %.2fx %s"
                  % (mode, k, eps, max(tr), np.median(tr), min(fa), np.median(fa), sep,
                     "✓ 可分" if sep > 1.5 else ""))
