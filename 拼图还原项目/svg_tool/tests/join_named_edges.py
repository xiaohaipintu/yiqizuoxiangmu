"""按用户指定对应关系拼两片：A 的 A6 边 贴 到 B 的 B2+B3 弧（不重叠），高亮这两处边并标注。"""
import sys, json, pathlib, tempfile, os
import numpy as np
import cv2

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import jigsaw_torn_solve as T
tmp = pathlib.Path(tempfile.gettempdir()) / "two"
out = ROOT / "svg_tool" / "test_out"


def rd(p): return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
def wr(p, im): cv2.imencode(".png", im)[1].tofile(str(p))


def start_topmost(poly):
    poly = np.asarray(poly, float)
    idx = np.lexsort((poly[:, 0], poly[:, 1]))[0]
    return np.roll(poly, -idx, axis=0)


def fit2(q0, q1, P0, P1):
    ang = np.arctan2(P1[1] - P0[1], P1[0] - P0[0]) - np.arctan2(q1[1] - q0[1], q1[0] - q0[0])
    R = T.rot(ang); t = P0 - R @ q0
    return R, t


def dense_contours(ds=2.5, minarea=3000):
    mask = cv2.imdecode(np.fromfile(str(tmp / "seg" / "01_mask.png"), np.uint8), cv2.IMREAD_GRAYSCALE)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 127).astype(np.uint8), 8)
    polys = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < minarea:
            continue
        cnts, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(float)
        polys.append(T.resample(c, ds))
    return polys


def edge_label_of(P, poly):
    """每个 dense 点最近属于简化多边形的哪条边(索引)。"""
    m = len(poly); best = np.full(len(P), 1e18); lab = np.zeros(len(P), int)
    for k in range(m):
        a = poly[k]; b = poly[(k + 1) % m]; ab = b - a; denom = ab @ ab + 1e-12
        t = np.clip(((P - a) @ ab) / denom, 0, 1)
        proj = a + t[:, None] * ab
        d = np.linalg.norm(P - proj, axis=1)
        upd = d < best; best[upd] = d[upd]; lab[upd] = k
    return lab


def main():
    doc = json.loads((tmp / "seg" / "pieces.json").read_text(encoding="utf-8"))
    ps = sorted(doc["pieces"], key=lambda p: (round(p["centroid"][1] / 200), p["centroid"][0]))
    A = start_topmost(np.array(ps[0]["polygon"], float))
    B = start_topmost(np.array(ps[1]["polygon"], float))
    nA, nB = len(A), len(B)
    print(f"A {nA}边  B {nB}边  |  A6={np.linalg.norm(A[6%nA]-A[5]):.0f}  "
          f"B2={np.linalg.norm(B[2]-B[1]):.0f} B3={np.linalg.norm(B[3]-B[2]):.0f}", flush=True)

    dense = dense_contours(2.5)
    dA, dB = dense[0], dense[1]
    a6_arc = dA[edge_label_of(dA, A) == 5] if nA > 6 else dA  # A6 = 第6条边(index5)
    print("A6 稠密弧点数", len(a6_arc), flush=True)
    from scipy.spatial import cKDTree
    kdA6 = cKDTree(a6_arc); kdAfull = cKDTree(dA)

    def overlap(pa, pb):
        from shapely.geometry import Polygon
        a = Polygon(pa); b = Polygon(pb)
        if not a.is_valid: a = a.buffer(0)
        if not b.is_valid: b = b.buffer(0)
        return a.intersection(b).area

    areaB = T.rd_poly_area(B)
    # A6 端点: A[5] -, A[6]; B 弧端点: B[1], B[3]
    best = None
    for (P0, P1) in [((A[6 % nA]), A[5]), (A[5], A[6 % nA])]:
        R, t = fit2(B[1], B[3], P0, P1)
        # ICP 精修：只用落到 A6 弧附近的 B 点
        for _ in range(12):
            d, idx = kdA6.query(dB @ R.T + t, distance_upper_bound=6.0)
            m = np.isfinite(d)
            if m.sum() < 15:
                break
            R, t = T.procrustes(dB[m], a6_arc[idx[m]], np.ones(int(m.sum())))
        d, _ = kdAfull.query(dB @ R.T + t, distance_upper_bound=3.0)
        contact = int(np.isfinite(d).sum())
        ov = overlap(B @ R.T + t, A)
        print(f"  拟合: A6<-B2+B3  contact={contact}  overlap={ov/areaB*100:.2f}%", flush=True)
        if ov < 0.03 * areaB and (best is None or contact > best[0]):
            best = (contact, R, t)
    if best is None:
        print("没有不重叠的贴合位姿"); return
    contact, R, t = best
    print(f"采用: contact={contact}", flush=True)

    # 渲染拼合（原图像素）+ 高亮 A6 / B2+B3
    # 稠密真实轮廓用于裁片 → 保留锯齿撕边（不用简化多边形的直边）
    dense_fine = dense_contours(1.5)
    dAf, dBf = dense_fine[0], dense_fine[1]

    def render(hl=True):
        polys = [(dAf, np.eye(2), np.zeros(2), (255, 120, 40), "A"), (dBf, R, t, (60, 60, 255), "B")]
        allp = np.vstack([p @ RR.T + tt for p, RR, tt, _c, _n in polys])
        x0 = int(allp[:, 0].min()) - 6; y0 = int(allp[:, 1].min()) - 6
        W = int(allp[:, 0].max() - x0) + 12; H = int(allp[:, 1].max() - y0) + 12
        cv = np.full((H, W, 3), (40, 55, 40), np.uint8)
        orig = rd(tmp / "two.jpg")
        for cell, RR, tt, col, nm in polys:
            tt2 = tt + np.array([-x0, -y0])
            cx0 = max(0, int(cell[:, 0].min())); cy0 = max(0, int(cell[:, 1].min()))
            cx1 = min(orig.shape[1], int(cell[:, 0].max()) + 1); cy1 = min(orig.shape[0], int(cell[:, 1].max()) + 1)
            crop = orig[cy0:cy1, cx0:cx1]; m = np.zeros(crop.shape[:2], np.uint8)
            cv2.fillPoly(m, [np.round(cell - [cx0, cy0]).astype(np.int32)], 255)
            sh = RR @ np.array([cx0, cy0], float) + tt2
            M = np.array([[RR[0, 0], RR[0, 1], sh[0]], [RR[1, 0], RR[1, 1], sh[1]]], float)
            wc = cv2.warpAffine(crop, M, (W, H)); wm = cv2.warpAffine(m, M, (W, H))
            cv[wm > 0] = wc[wm > 0]
        if hl:
            def draw_edge(cell, RR, tt, k, col, lab):
                a = cell[k] @ RR.T + tt + np.array([-x0, -y0]); b = cell[(k + 1) % len(cell)] @ RR.T + tt + np.array([-x0, -y0])
                cv2.line(cv, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), col, 3)
                mid = (a + b) / 2
                cv2.putText(cv, lab, (int(mid[0]) - 10, int(mid[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
                cv2.putText(cv, lab, (int(mid[0]) - 10, int(mid[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
            draw_edge(A, np.eye(2), np.zeros(2), 5, (0, 255, 255), "A6")      # A6
            draw_edge(B, R, t, 1, (0, 255, 0), "B2")                          # B2
            draw_edge(B, R, t, 2, (0, 255, 0), "B3")                          # B3
            cv2.putText(cv, "A6  <->  B2 + B3", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        return cv

    joined = render(True)
    wr(out / "two_joined_A6_B23.png", joined)
    clean = render(False)
    wr(out / "two_joined_clean.png", clean)
    print("saved 标注版  two_joined_A6_B23.png ; 无标注版 two_joined_clean.png", clean.shape)
    os.startfile(os.path.abspath(str(out / "two_joined_clean.png")))


if __name__ == "__main__":
    main()
