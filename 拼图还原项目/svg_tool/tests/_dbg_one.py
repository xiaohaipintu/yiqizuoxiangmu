"""单例诊断：直接跑一次 solve 看中间量。"""
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
print(f"n={len(polys)} σ={NOISE}")
pose, placed, pp = V2.solve(polys, scat, verbose=True)
B._POLYS = None
print("score:", B.score(polys, cells, gt_pose, list(pose), side))
