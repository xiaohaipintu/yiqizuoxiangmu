"""扫描纹理代价的各种配置，看哪种让「真搭档排第一」的比例最高。"""
import sys, pathlib, itertools, time
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
polys = [p if J._signed_area(p) > 0 else np.asarray(p, float)[::-1] for p in polys]

G0 = V2.build_edges(polys, K=28)
E = G0["E"]; PID = G0["pid"]; P0, P1 = G0["p0"], G0["p1"]


def to_orig(pts, pid):
    Rg, tg = gt_pose[pid]
    return (pts - tg) @ Rg


truth = {}
for i in range(E):
    a0, a1 = to_orig(P0[i], PID[i]), to_orig(P1[i], PID[i])
    best, bj = 1e9, -1
    for j in range(E):
        if PID[j] == PID[i]:
            continue
        b0, b1 = to_orig(P0[j], PID[j]), to_orig(P1[j], PID[j])
        d = min(np.linalg.norm(a0 - b1) + np.linalg.norm(a1 - b0),
                np.linalg.norm(a0 - b0) + np.linalg.norm(a1 - b1))
        if d < best:
            best, bj = d, j
    truth[i] = (bj, best)

inner = [i for i in range(E) if truth[i][1] < 3.0 + NOISE * 2]
print(f"σ={NOISE}  边 {E}  内部边 {len(inner)}")


def make_cost(dd, lo, hi, agg, K):
    def f(imgf, rs_a, cen_a, R, t, d=None):
        n = len(rs_a)
        i0, i1 = int(np.floor(lo * n)), int(np.ceil(hi * n))
        pts = rs_a[i0:i1] if (i1 - i0) >= 4 else rs_a
        e = rs_a[-1] - rs_a[0]
        L = float(np.linalg.norm(e))
        if L < 1e-6:
            return 60.0
        u = e / L
        nrm = np.array([-u[1], u[0]])
        if float(np.dot(nrm, cen_a - rs_a.mean(0))) < 0:
            nrm = -nrm
        pa = pts + nrm * dd
        pb = ((pts - nrm * dd) - t) @ R
        ca, oka = V2._samp_arr(imgf, pa)
        cb, okb = V2._samp_arr(imgf, pb)
        ok = oka & okb
        if ok.sum() < max(3, len(pts) // 3):
            return 60.0
        dif = np.abs(ca[ok] - cb[ok]).mean(axis=1)
        if agg == "mean":
            return float(dif.mean())
        if agg == "median":
            return float(np.median(dif))
        if agg == "p75":
            return float(np.percentile(dif, 75))
        if agg == "trim":                      # 去掉最好/最差各 15% 再取均值
            k = max(1, int(0.15 * len(dif)))
            s = np.sort(dif)[k:len(dif) - k] if len(dif) > 2 * k + 1 else dif
            return float(s.mean())
        raise ValueError(agg)
    return f


rows = []
for d, (lo, hi), agg, K in itertools.product(
        (2.0, 3.0, 4.0), ((0.15, 0.85), (0.2, 0.8), (0.25, 0.75)),
        ("mean", "median", "p75", "trim"), (20,)):
    V2._tex_cost = make_cost(d, lo, hi, agg, K)
    t0 = time.time()
    G, bmap, jp, jr = V2.edge_matches(polys, scat, topk=8, K=K)
    per = {i: [j for _, j in sorted([(v[0], j) for j, v in bmap[i].items()])] for i in range(E)}
    r0 = miss = 0
    for i in inner:
        tj = truth[i][0]
        lst = per[i]
        if not lst or lst[0] != tj:
            miss += 1
        if lst and lst[0] == tj:
            r0 += 1
    rows.append((r0 / len(inner), d, lo, hi, agg, time.time() - t0))

rows.sort(reverse=True)
print(f"{'rank0率':>9}{'d':>5}{'lo-hi':>13}{'agg':>7}{'es':>7}")
for r in rows[:15]:
    print(f"{r[0]*100:8.1f}%{r[1]:5.1f}f\"  {r[2]:.2f}-{r[3]:.2f} {r[4]:>6} {r[5]:6.1f}")
print("...")
for r in rows[-3:]:
    print(f"{r[0]*100:8.1f}%{r[1]:5.1f}  {r[2]:.2f}-{r[3]:.2f} {r[4]:>6} {r[5]:6.1f}")
