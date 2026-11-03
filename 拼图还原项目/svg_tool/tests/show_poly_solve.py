"""把 jigsaw_poly_solve.py 的结果摆正并生成「散落 | 复原 | 原图」对比图。"""
import sys
import pathlib

import cv2
import numpy as np

BASE = pathlib.Path("拼图还原项目/svg_tool/test_out")
SRC = BASE / "poly_solve"


def rd(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def wr(p, im):
    cv2.imencode(".png", im)[1].tofile(str(p))


def order_pts(b):
    s = b.sum(1); d = np.diff(b, axis=1).ravel()
    return np.array([b[np.argmin(s)], b[np.argmin(d)], b[np.argmax(s)], b[np.argmax(d)]], np.float32)


def upright(asm, S=760):
    g = cv2.cvtColor(asm, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(g > 12)
    if len(xs) < 10:
        return cv2.resize(asm, (S, S))
    rect = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
    box = cv2.boxPoints(rect)
    src = order_pts(box)
    dst = np.float32([[0, 0], [S, 0], [S, S], [0, S]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(asm, M, (S, S))


def main():
    asm = rd(SRC / "assembled.png")
    up = upright(asm)
    orig = cv2.resize(rd(BASE / "letter_h5.png"), (760, 760), interpolation=cv2.INTER_AREA)
    # 从 4 个旋转里挑最贴原图的那一版（消除整体朝向歧义）
    best, bestd = up, 1e9
    for k in range(4):
        cand = np.rot90(up, k).copy()
        d = np.abs(cv2.cvtColor(cand, cv2.COLOR_BGR2GRAY).astype(np.int16) -
                   cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY).astype(np.int16)).mean()
        if d < bestd:
            bestd, best = d, cand
    wr(SRC / "assembled_upright.png", best)
    print("复原图 vs 原图 平均灰度差 =", round(float(bestd), 2), "(越小越像)")

    scat = cv2.resize(rd(SRC / "scattered.png"), (760, 760), interpolation=cv2.INTER_AREA)
    S = 760; GAP = 16; TOP = 64
    W = S * 3 + GAP * 4; H = TOP + S + GAP * 2
    cv = np.full((H, W, 3), 20, np.uint8)

    def txt(t, o, sc, col, th=1):
        cv2.putText(cv, t, o, cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)

    txt("Polygon-jigsaw solver: 60 pieces, arbitrary rotation -> reassembled", (GAP + 2, 26), 0.62, (120, 245, 210), 2)
    txt("LEFT: scattered input     MIDDLE: reassembled (solver)     RIGHT: original letter", (GAP + 2, 50), 0.5, (160, 160, 172), 1)
    y0 = TOP
    for i, (img, lab) in enumerate([(scat, "SCATTERED (60)"), (best, "REASSEMBLED"), (orig, "ORIGINAL")]):
        x = GAP + i * (S + GAP)
        cv[y0:y0 + S, x:x + S] = img
        cv2.rectangle(cv, (x, y0), (x + S, y0 + S), (70, 70, 80), 1)
        txt(lab, (x + 8, y0 + 24), 0.6, (255, 255, 255), 1)
    out = BASE / "poly_solve_compare.png"
    wr(out, cv)
    print("saved", out, cv.shape)
    import os
    os.startfile(os.path.abspath(str(out)))


if __name__ == "__main__":
    main()
