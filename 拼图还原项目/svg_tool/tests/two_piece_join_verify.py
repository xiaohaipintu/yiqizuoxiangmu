"""无答案验证：自动搜两片拼合(不给边对应) → 把找到的撕痕映射回边名 → 看是否=A6↔B2+B3。"""
import sys, json, pathlib, tempfile, os
import numpy as np
import cv2

TESTS = pathlib.Path(__file__).resolve().parent
ROOT = TESTS.parents[1]
sys.path.insert(0, str(TESTS)); sys.path.insert(0, str(ROOT / "src"))
import jigsaw_torn_solve as T
import two_piece_join as TP

tmp = pathlib.Path(tempfile.gettempdir()) / "two"
out = ROOT / "svg_tool" / "test_out"


def rd(p): return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
def wr(p, im): cv2.imencode(".png", im)[1].tofile(str(p))


def start_topmost(poly):
    poly = np.asarray(poly, float)
    idx = np.lexsort((poly[:, 0], poly[:, 1]))[0]
    return np.roll(poly, -idx, axis=0)


def edge_label(P, simpl):
    m = len(simpl); best = np.full(len(P), 1e18); lab = np.zeros(len(P), int)
    for k in range(m):
        a = simpl[k]; b = simpl[(k + 1) % m]; ab = b - a; den = ab @ ab + 1e-12
        t = np.clip(((P - a) @ ab) / den, 0, 1); proj = a + t[:, None] * ab
        d = np.linalg.norm(P - proj, axis=1); u = d < best; best[u] = d[u]; lab[u] = k
    return lab


def main():
    doc = json.loads((tmp / "seg" / "pieces.json").read_text(encoding="utf-8"))
    ps = sorted(doc["pieces"], key=lambda p: (round(p["centroid"][1] / 200), p["centroid"][0]))
    A_s = start_topmost(np.array(ps[0]["polygon"], float))   # 7 顶点
    B_s = start_topmost(np.array(ps[1]["polygon"], float))   # 6 顶点

    dense = TP.dense_contours(2.5)
    # 把两片稠密轮廓按质心对到 A_s / B_s
    def cen(p): return p.mean(0)
    dcen = [cen(d) for d in dense]; acen = cen(A_s); bcen = cen(B_s)
    iA = int(np.argmin([np.linalg.norm(c - acen) for c in dcen]))
    iB = int(np.argmin([np.linalg.norm(c - bcen) for c in dcen if True])) if iA == 0 else 0
    iB = 1 - iA if len(dense) == 2 else iB
    dA, dB = dense[iA], dense[iB]

    print("自动搜索两片拼合(不给边对应) ...", flush=True)
    RT, contact = TP.join(dA, dB)   # 无答案：枚举边界点+切向
    if RT is None:
        print("没找到不重叠贴合"); return
    R, t = RT
    print(f"自动拼合: 贴合点数={contact}", flush=True)

    # 撕痕映射回边名
    from scipy.spatial import cKDTree
    kdA = cKDTree(dA)
    mp = dB @ R.T + t
    d, _ = kdA.query(mp, distance_upper_bound=3.0)
    seam = np.isfinite(d)
    bpts = dB[seam]; apts = mp[seam]
    eA = edge_label(apts, A_s) + 1   # -> A1.. coordinate
    eB = edge_label(bpts, B_s) + 1   # -> B1..
    from collections import Counter
    cA = Counter(eA.tolist()); cB = Counter(eB.tolist())
    print("撕痕对应到 A 的边:", sorted(cA.items(), key=lambda x: -x[1])[:3], flush=True)
    print("撕痕对应到 B 的边:", sorted(cB.items(), key=lambda x: -x[1])[:4], flush=True)
    print("==> 自动找到:", "A"+ str(cA.most_common(1)[0][0]), "~", 
          "B"+ str(cB.most_common(2)[0][0]), "+B"+str(cB.most_common(2)[1][0]) if len(cB)>=2 else "", flush=True)
    print("（人工给的答案: A6 ~ B2+B3）", flush=True)

    # 渲染无答案拼合(稠密锯齿边)
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
    wr(out / "two_noanswer_joined.png", cv)
    print("saved two_noanswer_joined.png", cv.shape, flush=True)
    os.startfile(os.path.abspath(str(out / "two_noanswer_joined.png")))


if __name__ == "__main__":
    main()
