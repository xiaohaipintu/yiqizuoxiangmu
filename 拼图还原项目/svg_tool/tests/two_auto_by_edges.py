"""自动拼两片(边长约束 + 贴合点必须落在被假设的那两条边上并覆盖大部分) → 抗近义错配。"""
import sys, json, pathlib, tempfile, os
import numpy as np
import cv2

TESTS = pathlib.Path(__file__).resolve().parent
ROOT = TESTS.parents[1]
sys.path.insert(0, str(TESTS)); sys.path.insert(0, str(ROOT / "src"))
import jigsaw_torn_solve as T
import two_piece_join as TP
from scipy.spatial import cKDTree

tmp = pathlib.Path(tempfile.gettempdir()) / "two"
out = ROOT / "svg_tool" / "test_out"


def rd(p): return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
def wr(p, im): cv2.imencode(".png", im)[1].tofile(str(p))


def start_topmost(poly):
    poly = np.asarray(poly, float); i = np.lexsort((poly[:, 0], poly[:, 1]))[0]
    return np.roll(poly, -i, axis=0)


def fit2(q0, q1, P0, P1):
    ang = np.arctan2(P1[1] - P0[1], P1[0] - P0[0]) - np.arctan2(q1[1] - q0[1], q1[0] - q0[0])
    R = T.rot(ang); t = P0 - R @ q0; return R, t


def edge_label(P, simpl):
    m = len(simpl); best = np.full(len(P), 1e18); lab = np.zeros(len(P), int)
    for k in range(m):
        a = simpl[k]; b = simpl[(k + 1) % m]; ab = b - a; den = ab @ ab + 1e-12
        t = np.clip(((P - a) @ ab) / den, 0, 1); proj = a + t[:, None] * ab
        d = np.linalg.norm(P - proj, axis=1); u = d < best; best[u] = d[u]; lab[u] = k
    return lab


def overlap(pa, pb):
    from shapely.geometry import Polygon
    a = Polygon(pa); b = Polygon(pb)
    if not a.is_valid: a = a.buffer(0)
    if not b.is_valid: b = b.buffer(0)
    return a.intersection(b).area


def main():
    doc = json.loads((tmp / "seg" / "pieces.json").read_text(encoding="utf-8"))
    ps = sorted(doc["pieces"], key=lambda p: (round(p["centroid"][1] / 200), p["centroid"][0]))
    A = start_topmost(np.array(ps[0]["polygon"], float)); B = start_topmost(np.array(ps[1]["polygon"], float))
    nA, nB = len(A), len(B)
    LA = np.array([np.linalg.norm(A[(k + 1) % nA] - A[k]) for k in range(nA)])
    LB = np.array([np.linalg.norm(B[(k + 1) % nB] - B[k]) for k in range(nB)])
    dense = TP.dense_contours(2.5)
    dcen = [d.mean(0) for d in dense]
    iA = int(np.argmin([np.linalg.norm(c - A.mean(0)) for c in dcen])); iB = 1 - iA if len(dense) == 2 else 0
    dA, dB = dense[iA], dense[iB]
    kdA = cKDTree(dA); areaB = T.rd_poly_area(B)
    labA_d = edge_label(dA, A); labB_d = edge_label(dB, B)
    cntA = [int((labA_d == k).sum()) for k in range(nA)]
    cntB = [int((labB_d == k).sum()) for k in range(nB)]

    hyps = []  # (name, k, Bend_pts, Bedge_indices)
    for k in range(nA):
        for j in range(nB):
            if abs(LB[j] - LA[k]) / max(LA[k], 1) < 0.18:
                hyps.append((f"A{k+1}~B{j+1}", k, [B[j], B[(j + 1) % nB]], [j]))
            s2 = LB[j] + LB[(j + 1) % nB]
            if abs(s2 - LA[k]) / max(LA[k], 1) < 0.18:
                hyps.append((f"A{k+1}~B{j+1}+B{j+2}", k, [B[j], B[(j + 2) % nB]], [j, (j + 1) % nB]))
    print("长度相容候选:", [h[0] for h in hyps], flush=True)

    results = []
    for name, k, Bend, Bedges in hyps:
        A0, A1 = A[k], A[(k + 1) % nA]
        for (P0, P1) in [(A1, A0), (A0, A1)]:
            R, t = fit2(Bend[0], Bend[1], P0, P1)
            for _ in range(12):
                d, idx = kdA.query(dB @ R.T + t, distance_upper_bound=6.0)
                m = np.isfinite(d)
                if m.sum() < 15: break
                R, t = T.procrustes(dB[m], dA[idx[m]], np.ones(int(m.sum())))
            dd, _ = kdA.query(dB @ R.T + t, distance_upper_bound=3.0)
            seam = np.isfinite(dd)
            anyc = int(seam.sum())
            aptsA = dB[seam] @ R.T + t
            ae = edge_label(aptsA, A); be = edge_label(dB[seam], B)
            onedge = (ae == k) & np.isin(be, Bedges)
            oncnt = int(onedge.sum())
            cov = oncnt / max(1, min(cntA[k], sum(cntB[b] for b in Bedges)))
            ov = overlap(B @ R.T + t, A) / areaB * 100
            if ov < 3.0:
                results.append((oncnt, cov, anyc, name, ov, R, t))
    results.sort(key=lambda x: (-x[0], -x[1]))
    print("按【落在被假设边上的贴合点数】排序(前8):")
    for oncnt, cov, anyc, name, ov, R, t in results[:8]:
        print(f"   {name:>16}  onedge={oncnt:4d}  cov={cov*100:5.1f}%  any={anyc:4d}  overlap={ov:.2f}%", flush=True)
    good = [r for r in results if r[1] >= 0.40 and r[0] >= 40]
    if not good:
        print("没有覆盖足够的解"); good = results
    oncnt, cov, anyc, name, ov, R, t = good[0]
    print(f"==> 自动选: {name}  onedge={oncnt} cov={cov*100:.1f}% overlap={ov:.2f}%   (人工答案 A6~B2+B3)", flush=True)

    polys = [(dA, np.eye(2), np.zeros(2)), (dB, R, t)]
    allp = np.vstack([p @ RR.T + tt for p, RR, tt in polys])
    x0 = int(allp[:, 0].min()) - 6; y0 = int(allp[:, 1].min()) - 6
    W = int(allp[:, 0].max() - x0) + 12; H = int(allp[:, 1].max() - y0) + 12
    cv = np.full((H, W, 3), (40, 55, 40), np.uint8); orig = rd(tmp / "two.jpg")
    for cell, RR, tt in polys:
        tt2 = tt + np.array([-x0, -y0])
        cx0 = max(0, int(cell[:, 0].min())); cy0 = max(0, int(cell[:, 1].min()))
        cx1 = min(orig.shape[1], int(cell[:, 0].max()) + 1); cy1 = min(orig.shape[0], int(cell[:, 1].max()) + 1)
        crop = orig[cy0:cy1, cx0:cx1]; m = np.zeros(crop.shape[:2], np.uint8)
        cv2.fillPoly(m, [np.round(cell - [cx0, cy0]).astype(np.int32)], 255)
        sh = RR @ np.array([cx0, cy0], float) + tt2
        M = np.array([[RR[0, 0], RR[0, 1], sh[0]], [RR[1, 0], RR[1, 1], sh[1]]], float)
        wc = cv2.warpAffine(crop, M, (W, H)); wm = cv2.warpAffine(m, M, (W, H))
        cv[wm > 0] = wc[wm > 0]
    wr(out / "two_auto_robust.png", cv)
    print("saved two_auto_robust.png", cv.shape, flush=True)
    os.startfile(os.path.abspath(str(out / "two_auto_robust.png")))


if __name__ == "__main__":
    main()
