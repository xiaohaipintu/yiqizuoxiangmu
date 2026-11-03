"""生成一张新图 → jigsaw_poly_solve 切 60 片并求解 → 出「散落|复原|原图」对比图。"""
import sys, pathlib, subprocess, os
import numpy as np
import cv2

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "svg_tool" / "test_out"


def rd(p): return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
def wr(p, im): cv2.imencode(".png", im)[1].tofile(str(p))


def make_new_source(S=760):
    rng = np.random.default_rng(2026)
    c1 = rng.integers(20, 150, 3).astype(float); c2 = rng.integers(120, 255, 3).astype(float)
    gx = np.linspace(0, 1, S)[None, :, None]
    base = c1[None, None, :] * (1 - gx) + c2[None, None, :] * gx     # 竖向渐变
    img = np.clip(np.broadcast_to(base, (S, S, 3)), 0, 255).astype(np.uint8).copy()
    # 随机彩色形状，制造丰富纹理（拼对时接缝两侧内容应连续）
    for _ in range(70):
        col = tuple(int(v) for v in rng.integers(0, 255, 3))
        cx, cy = int(rng.integers(0, S)), int(rng.integers(0, S)); r = int(rng.integers(14, 60))
        if rng.random() < 0.5:
            cv2.circle(img, (cx, cy), r, col, -1)
        else:
            k = int(rng.integers(3, 6)); ang = rng.uniform(0, 6.28, k)
            pts = np.array([[cx + r*np.cos(a), cy + r*np.sin(a)] for a in ang], np.int32)
            cv2.fillPoly(img, [pts], col)
    for i in range(6):                                               # 一些文字/线
        cv2.line(img, (20, 90 + i*110), (S-20, 90 + i*110), (240, 240, 240), 2)
        cv2.putText(img, "60-PIECE  JIGSAW  TEST", (40, 80 + i*110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    return img


def upright(asm, ref, S=760):
    g = cv2.cvtColor(asm, cv2.COLOR_BGR2GRAY); ys, xs = np.where(g > 12)
    if len(xs) < 10:
        return cv2.resize(asm, (S, S))
    rect = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32)); box = cv2.boxPoints(rect)
    s = box.sum(1); d = np.diff(box, axis=1).ravel()
    src = np.array([box[np.argmin(s)], box[np.argmin(d)], box[np.argmax(s)], box[np.argmax(d)]], np.float32)
    M = cv2.getPerspectiveTransform(src, np.float32([[0, 0], [S, 0], [S, S], [0, S]]))
    up = cv2.warpPerspective(asm, M, (S, S))
    best, bd = up, 1e9
    for k in range(4):
        c = np.rot90(up, k).copy()
        diff = np.abs(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(int) - cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(int)).mean()
        if diff < bd: bd, best = diff, c
    return best, float(bd)


def main():
    S = 760
    src = make_new_source(S); wr(OUT / "new_src.png", src)
    outdir = OUT / "poly_new60"
    print("新图已生成 → 切 60 片求解 ...", flush=True)
    p = subprocess.run([sys.executable, "src/jigsaw_poly_solve.py", "--n", "60", "--seed", "42",
                        "--image", str(OUT / "new_src.png"), "--out", str(outdir)],
                       cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8")
    for l in p.stdout.splitlines():
        if '检测' in l or '"placed"' in l or 'hull_frac' in l or '"total"' in l:
            print("  " + l.strip(), flush=True)

    scat = rd(outdir / "scattered.png"); asm = rd(outdir / "assembled.png")
    asm_up, diff = upright(asm, src, S)
    print(f"复原 vs 新原图 灰度差 = {diff:.2f}", flush=True)

    GAP = 16; TOP = 64
    a = cv2.resize(scat, (S, S)); b = asm_up; c = src
    W = S * 3 + GAP * 4; H = TOP + S + GAP * 2
    cv = np.full((H, W, 3), 18, np.uint8)
    def t(s, o, sc, col, th=1): cv2.putText(cv, s, o, cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)
    t("New random image  ->  cut into 60 pieces (seed 42)  ->  auto-reassembled", (GAP + 2, 26), 0.62, (120, 245, 210), 2)
    t("LEFT: scattered 60 pieces    MIDDLE: reassembled    RIGHT: original new image", (GAP + 2, 50), 0.5, (165, 165, 175), 1)
    for i, (img, lab, col) in enumerate([(a, "SCATTERED 60", (235, 235, 235)), (b, "REASSEMBLED", (120, 245, 210)), (c, "ORIGINAL", (235, 235, 235))]):
        x = GAP + i * (S + GAP); cv[TOP:TOP + S, x:x + S] = img
        cv2.rectangle(cv, (x, TOP), (x + S, TOP + S), (70, 70, 80), 1)
        t(lab, (x + 8, TOP + 24), 0.6, col, 1)
    outp = OUT / "poly_new60_compare.png"; wr(outp, cv)
    print("saved", outp, cv.shape, flush=True)
    os.startfile(os.path.abspath(str(outp)))


if __name__ == "__main__":
    main()
