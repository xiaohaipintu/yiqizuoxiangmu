"""两片手撕件自动拼合（服务端）：分割 → 边长+落边覆盖率自动选边对 → 拼合渲染 → base64。

供 web/server.py 的 /api/two 调用。仅支持正好 2 片。
"""
from __future__ import annotations
import sys, os, json, base64, subprocess, tempfile
import pathlib

WEB_DIR = pathlib.Path(__file__).resolve().parent
PROJ = WEB_DIR.parent
SRC = PROJ / "src"
SEG_PY = SRC / "segment.py"
sys.path.insert(0, str(SRC))

import numpy as np
import cv2
import jigsaw_torn_solve as T
from scipy.spatial import cKDTree


def _b64(im):
    ok, enc = cv2.imencode(".png", im)
    return base64.b64encode(enc.tobytes()).decode() if ok else ""


def _read(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def _start_topmost(poly):
    poly = np.asarray(poly, float)
    i = np.lexsort((poly[:, 0], poly[:, 1]))[0]
    return np.roll(poly, -i, axis=0)


def _fit2(q0, q1, P0, P1):
    ang = np.arctan2(P1[1] - P0[1], P1[0] - P0[0]) - np.arctan2(q1[1] - q0[1], q1[0] - q0[0])
    R = T.rot(ang); t = P0 - R @ q0
    return R, t


def _edge_label(P, simpl):
    m = len(simpl); best = np.full(len(P), 1e18); lab = np.zeros(len(P), int)
    for k in range(m):
        a = simpl[k]; b = simpl[(k + 1) % m]; ab = b - a; den = ab @ ab + 1e-12
        t = np.clip(((P - a) @ ab) / den, 0, 1); proj = a + t[:, None] * ab
        d = np.linalg.norm(P - proj, axis=1); u = d < best; best[u] = d[u]; lab[u] = k
    return lab


def _overlap(pa, pb):
    try:
        from shapely.geometry import Polygon
        a = Polygon(pa); b = Polygon(pb)
        if not a.is_valid: a = a.buffer(0)
        if not b.is_valid: b = b.buffer(0)
        return a.intersection(b).area
    except Exception:
        return 1e9


def _dense_contours(mask_png, ds=2.5, minarea=2500):
    mask = cv2.imdecode(np.fromfile(str(mask_png), np.uint8), cv2.IMREAD_GRAYSCALE)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 127).astype(np.uint8), 8)
    polys = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < minarea:
            continue
        cnts, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(float)
        polys.append(T.resample(c, ds))
    return polys


def assemble(image_path: str) -> dict:
    tmp = tempfile.mkdtemp(prefix="jigsaw-two-")
    seg = os.path.join(tmp, "seg"); os.makedirs(seg, exist_ok=True)
    rc = subprocess.run([sys.executable, str(SEG_PY), "--image", image_path, "--out", seg,
                         "--concave", "--min-area", "3000"],
                        cwd=str(PROJ), capture_output=True, text=True, timeout=240)
    pj = os.path.join(seg, "pieces.json")
    if not os.path.exists(pj):
        return {"ok": False, "error": "分割失败", "log": (rc.stdout + rc.stderr)[-800:]}
    pieces = json.load(open(pj, encoding="utf-8"))["pieces"]
    cut_path = os.path.join(seg, "03_cutouts.png")
    preview = _b64(_read(cut_path)) if os.path.exists(cut_path) else ""
    if len(pieces) != 2:
        return {"ok": False, "pieces": len(pieces),
                "error": f"检测到 {len(pieces)} 片；自动拼合目前只支持正好 2 片。",
                "preview_png": preview}

    ps = sorted(pieces, key=lambda p: (p["centroid"][1], p["centroid"][0]))
    A = _start_topmost(np.array(ps[0]["polygon"], float))
    B = _start_topmost(np.array(ps[1]["polygon"], float))
    nA, nB = len(A), len(B)
    LA = np.array([np.linalg.norm(A[(k + 1) % nA] - A[k]) for k in range(nA)])
    LB = np.array([np.linalg.norm(B[(k + 1) % nB] - B[k]) for k in range(nB)])

    dense = _dense_contours(os.path.join(seg, "01_mask.png"))
    if len(dense) != 2:
        return {"ok": False, "error": "稠密轮廓数不为 2", "preview_png": preview}
    dcen = [d.mean(0) for d in dense]
    iA = int(np.argmin([np.linalg.norm(c - A.mean(0)) for c in dcen])); iB = 1 - iA
    dA, dB = dense[iA], dense[iB]
    kdA = cKDTree(dA); areaB = T.rd_poly_area(B)
    labA_d = _edge_label(dA, A); labB_d = _edge_label(dB, B)
    cntA = [int((labA_d == k).sum()) for k in range(nA)]
    cntB = [int((labB_d == k).sum()) for k in range(nB)]

    hyps = []
    for k in range(nA):
        for j in range(nB):
            if abs(LB[j] - LA[k]) / max(LA[k], 1) < 0.18:
                hyps.append((f"A{k+1}~B{j+1}", k, [B[j], B[(j + 1) % nB]], [j]))
            s2 = LB[j] + LB[(j + 1) % nB]
            if abs(s2 - LA[k]) / max(LA[k], 1) < 0.18:
                hyps.append((f"A{k+1}~B{j+1}+B{j+2}", k, [B[j], B[(j + 2) % nB]], [j, (j + 1) % nB]))

    results = []
    for name, k, Bend, Bedges in hyps:
        A0, A1 = A[k], A[(k + 1) % nA]
        for (P0, P1) in [(A1, A0), (A0, A1)]:
            R, t = _fit2(Bend[0], Bend[1], P0, P1)
            for _ in range(12):
                d, idx = kdA.query(dB @ R.T + t, distance_upper_bound=6.0)
                m = np.isfinite(d)
                if m.sum() < 15: break
                R, t = T.procrustes(dB[m], dA[idx[m]], np.ones(int(m.sum())))
            dd, _ = kdA.query(dB @ R.T + t, distance_upper_bound=3.0)
            seam = np.isfinite(dd)
            aptsA = dB[seam] @ R.T + t
            ae = _edge_label(aptsA, A); be = _edge_label(dB[seam], B)
            onedge = (ae == k) & np.isin(be, Bedges)
            oncnt = int(onedge.sum())
            cov = oncnt / max(1, min(cntA[k], sum(cntB[b] for b in Bedges)))
            ov = _overlap(B @ R.T + t, A) / max(areaB, 1) * 100
            if ov < 3.0:
                results.append((oncnt, cov, name, ov, R, t))
    if not results:
        return {"ok": False, "error": "没找到不重叠的拼合", "preview_png": preview}
    results.sort(key=lambda x: (-x[0], -x[1]))
    good = [r for r in results if r[1] >= 0.40 and r[0] >= 40] or results
    oncnt, cov, name, ov, R, t = good[0]

    # 渲染拼合（用稠密轮廓，保留锯齿撕边）
    orig = _read(image_path)
    polys = [(dA, np.eye(2), np.zeros(2)), (dB, R, t)]
    allp = np.vstack([p @ RR.T + tt for p, RR, tt in polys])
    x0 = int(allp[:, 0].min()) - 6; y0 = int(allp[:, 1].min()) - 6
    W = int(allp[:, 0].max() - x0) + 12; H = int(allp[:, 1].max() - y0) + 12
    cv = np.full((H, W, 3), (40, 55, 40), np.uint8)
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
    return {"ok": True, "pieces": 2, "edges": name.replace("~", " ↔ "),
            "contact": oncnt, "coverage_pct": round(cov * 100, 1),
            "overlap_pct": round(ov, 3), "joined_png": _b64(cv), "preview_png": preview}
