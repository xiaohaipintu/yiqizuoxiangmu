"""两片手撕件拼合：找刚体变换让 B 的边界长弧贴合 A 的边界(撕痕)、且不重叠，渲染拼合图。"""
import sys, pathlib, tempfile, time, os
import numpy as np
import cv2
import itertools

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import jigsaw_torn_solve as T

tmp = pathlib.Path(tempfile.gettempdir()) / "two"
out = ROOT / "svg_tool" / "test_out"


def rd(p): return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
def wr(p, im): cv2.imencode(".png", im)[1].tofile(str(p))


def dense_contours(ds=2.5, minarea=3000):
    mask = cv2.imdecode(np.fromfile(str(tmp / "seg" / "01_mask.png"), np.uint8), cv2.IMREAD_GRAYSCALE)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 127).astype(np.uint8), 8)
    polys = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < minarea:
            continue
        cnts, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(float)
        polys.append(T.resample(c, ds))
    return polys


def tangents_of(dense):
    t = np.roll(dense, -1, axis=0) - np.roll(dense, 1, axis=0)
    return t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)


def overlap_area(pa, pb):
    try:
        from shapely.geometry import Polygon
        a = Polygon(pa); b = Polygon(pb)
        if not a.is_valid: a = a.buffer(0)
        if not b.is_valid: b = b.buffer(0)
        return a.intersection(b).area
    except Exception:
        return 1e9


def join(A, B, eps=3.0, stepA=10, stepB=10, qstep=4):
    """找把 B 贴到 A 上的位姿：最大化边界贴合点数，且重叠极小。"""
    from scipy.spatial import cKDTree
    kdA = cKDTree(A)
    tanA = tangents_of(A); tanB = tangents_of(B)
    sA = A[::stepA]; tA = tanA[::stepA]
    sB = B[::stepB]; tB = tanB[::stepB]
    Bq = B[::qstep]
    areaB = T.rd_poly_area(B)
    best = (-1, None)
    for ia in range(len(sA)):
        ta = tA[ia]; a = sA[ia]
        for ib in range(len(sB)):
            tb = tB[ib]; b = sB[ib]
            theta = np.arctan2(-ta[1], -ta[0]) - np.arctan2(tb[1], tb[0])
            R = T.rot(theta); t = a - R @ b
            d, _ = kdA.query(Bq @ R.T + t, distance_upper_bound=eps)
            contact = int(np.isfinite(d).sum())
            if contact > best[0]:
                pb = B @ R.T + t
                if overlap_area(pb, A) < 0.03 * areaB:   # 只贴不叠
                    best = (contact, (R, t))
    if best[1] is None:
        return None, 0
    R, t = best[1]
    # ICP 精修（只用落在 A 边界附近的 B 点=撕痕）
    for _ in range(8):
        d, idx = kdA.query(B @ R.T + t, distance_upper_bound=eps + 1.5)
        m = np.isfinite(d)
        if m.sum() < 20:
            break
        R, t = T.procrustes(B[m], A[idx[m]], np.ones(int(m.sum())))
    # 最终贴合长度
    d, _ = kdA.query(B @ R.T + t, distance_upper_bound=eps)
    return (R, t), int(np.isfinite(d).sum())


def render(A, B, RT, orig):
    polys = [("A", A, (np.eye(2), np.zeros(2))), ("B", B, RT)]
    allp = np.vstack([p @ R.T + t for _, p, (R, t) in polys])
    x0 = int(allp[:, 0].min()) - 4; y0 = int(allp[:, 1].min()) - 4
    side_w = int(allp[:, 0].max() - x0) + 8; side_h = int(allp[:, 1].max() - y0) + 8
    cv = np.full((side_h, side_w, 3), (35, 55, 35), np.uint8)
    for _, cell, (R, t) in polys:
        t2 = t + np.array([-x0, -y0])
        cx0 = max(0, int(cell[:, 0].min())); cy0 = max(0, int(cell[:, 1].min()))
        cx1 = min(orig.shape[1], int(cell[:, 0].max()) + 1); cy1 = min(orig.shape[0], int(cell[:, 1].max()) + 1)
        crop = orig[cy0:cy1, cx0:cx1]; m = np.zeros(crop.shape[:2], np.uint8)
        cv2.fillPoly(m, [np.round(cell - [cx0, cy0]).astype(np.int32)], 255)
        sh = R @ np.array([cx0, cy0], float) + t2
        M = np.array([[R[0, 0], R[0, 1], sh[0]], [R[1, 0], R[1, 1], sh[1]]], float)
        wc = cv2.warpAffine(crop, M, (side_w, side_h)); wm = cv2.warpAffine(m, M, (side_w, side_h))
        cv[wm > 0] = wc[wm > 0]
    return cv


def main():
    polys = dense_contours(2.5)
    print("碎片数", len(polys), flush=True)
    if len(polys) != 2:
        print("不是两片, 退出"); return
    A, B = polys
    t0 = time.time()
    RT, contact = join(A, B)
    print(f"拼合: 撕痕贴合点数={contact}, {time.time()-t0:.1f}s", flush=True)
    if RT is None:
        print("没找到不重叠的贴合位姿"); return
    orig = rd(tmp / "two.jpg")
    joined = render(A, B, RT, orig)
    wr(out / "two_joined.png", joined)
    # 并排对比：原图 | 拼合
    src = cv2.resize(orig, (860, int(860 * orig.shape[0] / orig.shape[1])))
    jj = cv2.resize(joined, (860, int(860 * joined.shape[0] / joined.shape[1])))
    H = max(src.shape[0], jj.shape[0]) + 64; W = 860 * 2 + 48
    cv = np.full((H, W, 3), 20, np.uint8)
    def txt(s, o, sc, c, th=1): cv2.putText(cv, s, o, cv2.FONT_HERSHEY_SIMPLEX, sc, c, th, cv2.LINE_AA)
    txt("Two torn pieces  |  fit along torn seam (no overlap)", (16, 28), 0.62, (120, 245, 210), 2)
    txt(f"LEFT: original photo     RIGHT: joined  (seam contact pts={contact})", (16, 52), 0.5, (165, 165, 175), 1)
    cv[64:64 + src.shape[0], 16:16 + 860] = src
    cv[64:64 + jj.shape[0], 900:900 + 860] = jj
    wr(out / "two_joined_compare.png", cv)
    print("saved two_joined_compare.png", flush=True)
    os.startfile(os.path.abspath(str(out / "two_joined_compare.png")))


if __name__ == "__main__":
    main()
