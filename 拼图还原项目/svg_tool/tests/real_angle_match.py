"""真实手撕照片 · 简单版：中等多边形(带真实边长+转角) 反序匹配 → 拼出最大的碎片簇。"""
import sys, pathlib, tempfile, time, os
from collections import defaultdict, deque
import numpy as np
import cv2

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import jigsaw_torn_solve as T

tmp = pathlib.Path(tempfile.gettempdir()) / "tornreal"
out = ROOT / "svg_tool" / "test_out"
SRC = "H:/eb0bf8ee40415bc6c728ceb66879b3ac.jpg"


def rd(p): return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
def wr(p, im): cv2.imencode(".png", im)[1].tofile(str(p))


def mid_polys(eps_frac=0.010, minarea=3000):
    mask = cv2.imdecode(np.fromfile(str(tmp / "seg" / "01_mask.png"), np.uint8), cv2.IMREAD_GRAYSCALE)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 127).astype(np.uint8), 8)
    polys = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < minarea:
            continue
        cnts, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        peri = cv2.arcLength(c, True)
        ap = cv2.approxPolyDP(c, eps_frac * peri, True).reshape(-1, 2).astype(float)
        if len(ap) < 4:
            continue
        polys.append(ap if T.signed_area(ap) > 0 else ap[::-1])
    return polys


def geom(poly):
    m = len(poly)
    e = np.array([poly[(k + 1) % m] - poly[k] for k in range(m)])
    L = np.linalg.norm(e, axis=1); E = e / np.maximum(L[:, None], 1e-9)
    turn = np.zeros(m)
    for k in range(m):
        a = E[(k - 1) % m]; b = E[k]
        turn[k] = abs(np.arctan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]))
    return L, turn


def match(polys, tolL=4.0, tolA=0.6, minK=4):
    n = len(polys); G = [geom(p) for p in polys]; edges = []
    for i in range(n):
        Li, Ti = G[i]; na = len(Li)
        for j in range(i + 1, n):
            Lj, Tj = G[j]; nb = len(Lj); best = (0, 0, 0)  # k, ai, bj
            for ai in range(na):
                for bj in range(nb):
                    k = 0
                    while (k < min(na, nb) // 2
                           and abs(Li[(ai + k) % na] - Lj[(bj - k) % nb]) < tolL
                           and abs(Ti[(ai + k) % na] - Tj[(bj + 1 - k) % nb]) < tolA):
                        k += 1
                    if k > best[0]:
                        best = (k, ai, bj)
            k, ai, bj = best
            if k < minK:
                continue
            Apts = np.array([polys[i][(ai + t) % na] for t in range(k + 1)])
            Bpts = np.array([polys[j][(bj + 1 - t) % nb] for t in range(k + 1)])
            R, tp = T.procrustes(Bpts, Apts)
            res = float(np.linalg.norm(Apts - (Bpts @ R.T + tp), axis=1).mean())
            edges.append((res, k, i, j, R, tp))
    edges.sort(key=lambda e: (e[0], -e[1]))
    return edges


def clusters(n, edges, res_thr):
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for (res, k, i, j, R, tp) in edges:
        if res <= res_thr:
            a, b = find(i), find(j)
            if a != b: parent[a] = b
    cl = defaultdict(list)
    for x in range(n): cl[find(x)].append(x)
    return sorted(cl.values(), key=len, reverse=True)


def render(group, polys, edges, orig):
    ein = {}
    for (res, k, i, j, R, tp) in edges:
        if i in group and j in group:
            ein.setdefault((i, j), (R, tp)); ein.setdefault((j, i), (R.T, -R.T @ tp))
    pose = {group[0]: (np.eye(2), np.zeros(2))}
    ch = True
    while ch:
        ch = False
        for (a, b), (R, tp) in ein.items():
            if a in pose and b not in pose:
                RA, tA = pose[a]; pose[b] = (RA @ R, RA @ tp + tA); ch = True
    allp = np.vstack([polys[i] @ pose[i][0].T + pose[i][1] for i in pose])
    x0 = int(allp[:, 0].min()) - 3; y0 = int(allp[:, 1].min()) - 3
    side = int(max(allp[:, 0].max() - x0, allp[:, 1].max() - y0)) + 6
    cv = np.full((side, side, 3), (50, 70, 50), np.uint8)
    for i in pose:
        R, t = pose[i]; t = t + np.array([-x0, -y0]); cell = polys[i]
        cx0 = max(0, int(cell[:, 0].min())); cy0 = max(0, int(cell[:, 1].min()))
        cx1 = min(orig.shape[1], int(cell[:, 0].max()) + 1); cy1 = min(orig.shape[0], int(cell[:, 1].max()) + 1)
        crop = orig[cy0:cy1, cx0:cx1]; m = np.zeros(crop.shape[:2], np.uint8)
        cv2.fillPoly(m, [np.round(cell - [cx0, cy0]).astype(np.int32)], 255)
        sh = R @ np.array([cx0, cy0], float) + t
        M = np.array([[R[0, 0], R[0, 1], sh[0]], [R[1, 0], R[1, 1], sh[1]]], float)
        wc = cv2.warpAffine(crop, M, (side, side)); wm = cv2.warpAffine(m, M, (side, side))
        cv[wm > 0] = wc[wm > 0]
    return cv


def main():
    polys = mid_polys(0.010)
    print("中等多边形碎片数", len(polys), " 顶点数", [len(p) for p in polys], flush=True)
    t0 = time.time(); edges = match(polys); print(f"候选相邻边 {len(edges)} 条, {time.time()-t0:.1f}s", flush=True)
    if edges:
        rs = sorted(e[0] for e in edges)
        print("残差 min/中位:", round(rs[0], 2), round(rs[len(rs)//2], 2), " 最长撕痕(边数)前5:", [e[1] for e in edges[:5]], flush=True)
    orig = rd(tmp / "torn.jpg") if (tmp / "torn.jpg").exists() else rd(SRC)
    best = None
    for thr in [2.0, 3.0, 4.0, 6.0]:
        gs = clusters(len(polys), edges, thr); big = gs[0]
        print(f"res_thr={thr}: 最大簇 {len(big)} 片, 簇 {[len(g) for g in gs[:6]]}", flush=True)
        if best is None or len(big) > len(best[1]): best = (thr, big)
    thr, big = best
    if len(big) < 2:
        print("没有高置信配对。"); return
    canvas = render(big, polys, edges, orig)
    wr(out / "real_angle_assembled.png", canvas)
    print(f"渲染最大簇 {len(big)} 片 -> {out/'real_angle_assembled.png'}", canvas.shape)
    os.startfile(os.path.abspath(str(out / "real_angle_assembled.png")))


if __name__ == "__main__":
    main()
