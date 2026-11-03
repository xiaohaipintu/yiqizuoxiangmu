"""候选片对的查准/查全：真邻居有没有被找到？"""
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
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
NOISE = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
tex = B.load_tex("", S)
polys, cells, gt_pose, scat, side = B.build_case(N, S, tex, 7, NOISE)
polys = [p if J._signed_area(p) > 0 else np.asarray(p, float)[::-1] for p in polys]
clean = [np.asarray(c, float) for c in J.make_cells(N, S, 7)]
print(f"n={len(polys)} σ={NOISE}")

# 真邻居：干净 Voronoi 里共边的片对
def seg_key(a, b):
    return (round(float(a[0]), 3), round(float(a[1]), 3), round(float(b[0]), 3), round(float(b[1]), 3))

seg2piece = {}
for pid, c in enumerate(clean):
    m = len(c)
    for k in range(m):
        a, b = c[k], c[(k + 1) % m]
        key = tuple(sorted([(round(a[0], 2), round(a[1], 2)), (round(b[0], 2), round(b[1], 2))]))
        seg2piece.setdefault(key, []).append(pid)
true_pairs = set()
for key, ps in seg2piece.items():
    if len(ps) == 2:
        true_pairs.add(tuple(sorted(ps)))
print("真邻居片对", len(true_pairs))

G, best, jp, jr = V2.edge_matches(polys, scat)
PID = G["pid"]
print(f"{'ang':>5}{'pos':>6}{'候选':>7}{'作证':>6}{'作证中真':>9}{'作证查准':>9}{'作证查全':>9}")
for ang in (3.0, 5.0, 8.0, 12.0):
    for pos in (6.0, 12.0, 20.0, 30.0):
        rels, sup = V2.join_table(polys, G, best, sorted(jp + jr), ang_tol=ang, pos_tol=pos)
        stp = sum(1 for k in sup if k in true_pairs)
        print(f"{ang:5.1f}{pos:6.1f}{len(rels):7d}{len(sup):6d}{stp:9d}"
              f"{stp/max(len(sup),1)*100:8.1f}%{stp/max(len(true_pairs),1)*100:8.1f}%")
