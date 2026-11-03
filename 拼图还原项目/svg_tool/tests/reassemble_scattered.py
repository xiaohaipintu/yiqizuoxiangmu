"""从散落图分割出碎片多边形 → 用 jigsaw_poly_solve.solve 重拼（容差+纹理+形状）。"""
import sys, time
import numpy as np
import cv2

sys.path.insert(0, "拼图还原项目/src")
import jigsaw_poly_solve as J

P = r"C:/Users/Administrator/AppData/Local/Temp/reasonix-pasted-images/reasonix-pasted-1793684936977.png"
OUT = "拼图还原项目/svg_tool/test_out/reasm"


def main():
    img = cv2.imdecode(np.fromfile(P, np.uint8), cv2.IMREAD_COLOR)
    H, W = img.shape[:2]
    mask = (img.max(2) > 60).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    polys = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 300:
            continue
        cnts, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cnts, key=cv2.contourArea)
        peri = cv2.arcLength(c, True)
        ap = cv2.approxPolyDP(c, 0.010 * peri, True).reshape(-1, 2).astype(float)
        if len(ap) < 3:
            continue
        polys.append(ap if J._signed_area(ap) > 0 else ap[::-1])
    print(f"分割出 {len(polys)} 片，开始重拼 ...", flush=True)

    t0 = time.time()
    pose, placed, pp = J.solve(polys, len_tol=6.0, scat=img, shape_tol=12.0, tex_tol=55.0)
    dt = time.time() - t0
    m = J.evaluate(pp, max(H, W))
    print(f"solve {dt:.1f}s  placed {placed}/{len(polys)}  hull_frac={m.get('hull_frac')}  overlap%={m.get('overlap_pct')}", flush=True)

    # 渲染复原图
    if placed >= 2:
        allp = np.vstack([pp[k] for k in pp])
        x0 = int(allp[:, 0].min()) - 4; y0 = int(allp[:, 1].min()) - 4
        side = int(max(allp[:, 0].max() - x0, allp[:, 1].max() - y0)) + 8
        ps = [None] * len(polys)
        for k in pp:
            R, t = pose[k]; ps[k] = (R, t + np.array([-x0, -y0]))
        asm = J.render_assembled(img, max(H, W), polys, ps, side)
        cv2.imencode(".png", asm)[1].tofile(OUT + "_assembled.png")
        print("saved", OUT + "_assembled.png", asm.shape, flush=True)
        import os
        os.startfile(os.path.abspath(OUT + "_assembled.png"))


if __name__ == "__main__":
    main()
