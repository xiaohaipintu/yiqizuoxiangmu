"""对真实手撕照片做端到端尝试：掩膜 → 稠密轮廓 → 容错拼装 → 渲染。"""
import sys, pathlib, tempfile, time, os
import numpy as np
import cv2

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import jigsaw_torn_solve as T

tmp = pathlib.Path(tempfile.gettempdir()) / "tornreal"
out = ROOT / "svg_tool" / "test_out"
SRC = r"H:/eb0bf8ee40415bc6c728ceb66879b3ac.jpg"


def rd(p): return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
def wr(p, im): cv2.imencode(".png", im)[1].tofile(str(p))


def main():
    import shutil, subprocess
    if not (tmp / "seg" / "01_mask.png").exists():
        tmp.mkdir(parents=True, exist_ok=True)
        img = tmp / "torn.jpg"; shutil.copyfile(SRC, img)
        subprocess.run([sys.executable, "src/segment.py", "--image", str(img), "--out",
                        str(tmp / "seg"), "--concave", "--min-area", "2500"],
                       cwd=str(ROOT), capture_output=True, text=True, timeout=180)
    mask = cv2.imdecode(np.fromfile(str(tmp / "seg" / "01_mask.png"), np.uint8), cv2.IMREAD_GRAYSCALE)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 127).astype(np.uint8), 8)
    polys = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 3000:
            continue
        comp = (lab == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(float)
        if len(c) < 12:
            continue
        polys.append(c)
    print("稠密轮廓碎片数:", len(polys), " 顶点数样例:", [len(p) for p in polys[:8]])

    # 试几组容错参数
    best = None
    for mv, tl, rt in [(20, 2.0, 2.5), (15, 2.5, 2.0), (25, 1.5, 1.8)]:
        t0 = time.time()
        pose, placed, pp = T.solve_v3(polys, min_verts=mv, tol_len=tl, res_tol=rt, seed=3)
        m = T.evaluate(pp)
        print(f"  mv={mv} tol={tl} res={rt}: placed {placed}/{len(polys)} hull={m.get('hull_frac')} "
              f"ov%={m.get('overlap_pct')} {time.time()-t0:.1f}s", flush=True)
        key = (placed, m.get("hull_frac") or 0)
        if best is None or key > best[0]:
            best = (key, pose, placed, pp)
    _, pose, placed, pp = best

    # 渲染拼装（用原照片像素）
    orig = rd(tmp / "torn.jpg") if (tmp / "torn.jpg").exists() else rd(SRC)
    allp = np.vstack([pp[k] for k in pp])
    x0 = int(np.floor(allp[:, 0].min())) - 3; y0 = int(np.floor(allp[:, 1].min())) - 3
    side = int(np.ceil(max(allp[:, 0].max() - x0, allp[:, 1].max() - y0))) + 6
    canvas = np.full((side, side, 3), (40, 60, 40), np.uint8)
    polys_ccw = [p if T.signed_area(p) > 0 else p[::-1] for p in polys]
    sh = np.array([-x0, -y0], float)
    # 用原始稠密多边形在原图上裁片，按位姿贴回
    for k in range(len(polys)):
        if pose[k] is None:
            continue
        R, t = pose[k]
        t = t + sh
        cell = polys_ccw[k]
        cx0 = max(0, int(cell[:, 0].min())); cy0 = max(0, int(cell[:, 1].min()))
        cx1 = min(orig.shape[1], int(cell[:, 0].max()) + 1); cy1 = min(orig.shape[0], int(cell[:, 1].max()) + 1)
        crop = orig[cy0:cy1, cx0:cx1]
        m = np.zeros(crop.shape[:2], np.uint8)
        cv2.fillPoly(m, [np.round(cell - [cx0, cy0]).astype(np.int32)], 255)
        shf = R @ np.array([cx0, cy0], float) + t
        M = np.array([[R[0, 0], R[0, 1], shf[0]], [R[1, 0], R[1, 1], shf[1]]], float)
        wc = cv2.warpAffine(crop, M, (side, side)); wm = cv2.warpAffine(m, M, (side, side))
        canvas[wm > 0] = wc[wm > 0]
    wr(out / "real_torn_assembled.png", canvas)
    wr(out / "real_torn_pieces_preview.png", rd(tmp / "seg" / "03_cutouts.png"))
    print("saved real_torn_assembled.png", canvas.shape)
    os.startfile(os.path.abspath(str(out / "real_torn_assembled.png")))


if __name__ == "__main__":
    main()
