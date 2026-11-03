"""真实手撕照片：找出高置信相邻片，拼出最大的一个碎片簇（部分拼合），渲染展示。"""
import sys, pathlib, tempfile, time, os
from collections import defaultdict
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


def dense_polys(ds=3.5):
    mask = cv2.imdecode(np.fromfile(str(tmp / "seg" / "01_mask.png"), np.uint8), cv2.IMREAD_GRAYSCALE)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 127).astype(np.uint8), 8)
    polys = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 3000:
            continue
        cnts, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(float)
        if len(c) < 12:
            continue
        polys.append(T.resample(c, ds))
    return polys


def match_edges(polys, tol=3.0, minK=8, res_tol=4.0):
    n = len(polys)
    Ls = [np.array([np.linalg.norm(p[(k + 1) % len(p)] - p[k]) for k in range(len(p))]) for p in polys]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            Li = Ls[i]; Lj = Ls[j]; na = len(Li); nb = len(Lj); cap = min(na, nb) // 2
            qb = defaultdict(list)
            for bj in range(nb):
                qb[round(float(Lj[bj]) / tol)].append(bj)
            bestk = 0; bi = bj0 = 0
            for ai in range(na):
                qa = round(float(Li[ai]) / tol)
                for dq in (-1, 0, 1):
                    for c in qb.get(qa + dq, ()):
                        k = 0
                        while k < cap and abs(Li[(ai + k) % na] - Lj[(c - k) % nb]) < tol:
                            k += 1
                        if k > bestk:
                            bestk, bi, bj0 = k, ai, c
            if bestk < minK:
                continue
            Apts = np.array([polys[i][(bi + t) % na] for t in range(bestk + 1)])
            Bpts = np.array([polys[j][(bj0 + 1 - t) % nb] for t in range(bestk + 1)])
            R, tp = T.procrustes(Bpts, Apts)
            res = float(np.linalg.norm(Apts - (Bpts @ R.T + tp), axis=1).mean())
            if res < res_tol:
                edges.append((res, bestk, i, j, R, tp))
    edges.sort(key=lambda e: (e[0], -e[1]))
    return edges


def build_clusters(n, edges, res_thresh):
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for (res, k, i, j, R, tp) in sorted(edges):
        if res <= res_thresh:
            a, b = find(i), find(j)
            if a != b:
                parent[a] = b
    cl = defaultdict(list)
    for x in range(n):
        cl[find(x)].append(x)
    groups = sorted(cl.values(), key=len, reverse=True)
    return groups


def render_cluster(group, polys, edges, orig):
    # 用组内最强边 BFS 摆位
    edges_in = {}
    for (res, k, i, j, R, tp) in edges:
        if i in group and j in group:
            edges_in.setdefault((i, j), (res, k, R, tp))
            edges_in.setdefault((j, i), (res, k, R.T, -R.T @ tp))
    root = group[0]
    pose = {root: (np.eye(2), np.zeros(2))}
    changed = True
    while changed:
        changed = False
        for (a, b), (res, k, R, tp) in edges_in.items():
            if a in pose and b not in pose:
                RA, tA = pose[a]; pose[b] = (RA @ R, RA @ tp + tA); changed = True
    allp = np.vstack([polys[i] @ pose[i][0].T + pose[i][1] for i in pose])
    x0 = int(np.floor(allp[:, 0].min())) - 3; y0 = int(np.floor(allp[:, 1].min())) - 3
    side = int(np.ceil(max(allp[:, 0].max() - x0, allp[:, 1].max() - y0))) + 6
    canvas = np.full((side, side, 3), (50, 70, 50), np.uint8)
    for i in pose:
        R, t = pose[i]; t = t + np.array([-x0, -y0])
        cell = polys[i]
        cx0 = max(0, int(cell[:, 0].min())); cy0 = max(0, int(cell[:, 1].min()))
        cx1 = min(orig.shape[1], int(cell[:, 0].max()) + 1); cy1 = min(orig.shape[0], int(cell[:, 1].max()) + 1)
        crop = orig[cy0:cy1, cx0:cx1]
        m = np.zeros(crop.shape[:2], np.uint8)
        cv2.fillPoly(m, [np.round(cell - [cx0, cy0]).astype(np.int32)], 255)
        sh = R @ np.array([cx0, cy0], float) + t
        M = np.array([[R[0, 0], R[0, 1], sh[0]], [R[1, 0], R[1, 1], sh[1]]], float)
        wc = cv2.warpAffine(crop, M, (side, side)); wm = cv2.warpAffine(m, M, (side, side))
        canvas[wm > 0] = wc[wm > 0]
    return canvas


def main():
    polys = dense_polys(8.0)
    polys = sorted(polys, key=lambda p: -T.rd_poly_area(p))[:8]
    print("用最大 8 片尝试, 碎片数", len(polys), flush=True)
    t0 = time.time()
    edges = match_edges(polys)
    print(f"找到 {len(edges)} 条候选相邻边, {time.time()-t0:.1f}s", flush=True)
    if edges:
        rs = sorted(e[0] for e in edges)
        print("残差分布 min/中位/max:", round(rs[0], 2), round(rs[len(rs)//2], 2), round(rs[-1], 2))
    orig = rd(tmp / "torn.jpg") if (tmp / "torn.jpg").exists() else rd(SRC)
    best = None
    for rt in [1.2, 2.0, 3.0, 4.5]:
        groups = build_clusters(len(polys), edges, rt)
        big = groups[0] if groups else []
        print(f"res_thresh={rt}: 最大簇 {len(big)} 片  簇大小分布 {[len(g) for g in groups[:6]]}", flush=True)
        if best is None or len(big) > len(best[1]):
            best = (rt, big)
    rt, big = best
    if len(big) < 2:
        print("没有能高置信连起来的片对（残差都太大）。")
        return
    canvas = render_cluster(big, polys, edges, orig)
    wr(out / "real_partial_assembled.png", canvas)
    print(f"渲染最大簇：{len(big)} 片 (res_thresh={rt}) -> {out/'real_partial_assembled.png'}", canvas.shape)
    os.startfile(os.path.abspath(str(out / "real_partial_assembled.png")))


if __name__ == "__main__":
    main()
