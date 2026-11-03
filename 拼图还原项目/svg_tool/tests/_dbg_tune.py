"""调参：min_votes / grow_single / rel_len / topk 对拼对率的影响。"""
import sys, pathlib, time, itertools
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
NOISE = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
tex = B.load_tex("", S)
polys, cells, gt_pose, scat, side = B.build_case(N, S, tex, 7, NOISE)
print(f"n={len(polys)} σ={NOISE}")

combos = []
for mv, tk in itertools.product((2, 3, 4), (3, 4, 5, 6)):
    combos.append(dict(min_votes=mv, topk=tk))

print(f"{'params':<46}{'placed':>8}{'拼对':>6}{'拼对率':>8}{'meanIoU':>9}{'es':>7}")
for p in combos:
    t0 = time.time()
    pose, placed, pp = V2.solve(polys, scat, verbose=False, **p)
    dt = time.time() - t0
    m = B.score(polys, cells, gt_pose, list(pose), side)
    print(f"{str(p):<46}{placed:>8}{m['correct']:>6}{m['rate']*100:>7.1f}%"
          f"{m['mean_iou']:>9.3f}{dt:>7.2f}")
