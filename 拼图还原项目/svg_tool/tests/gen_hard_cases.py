"""生成「实拍抠组件」硬核测试集（已知真值），供 run_hard_cases.py 回归。

每个 case 输出 <name>.jpg + <name>.json（真值：碎片数 / 总前景面积占比 / 说明）。
覆盖：绿幕/白底、阴影、渐变光、印刷暗点、低对比、近距离碎片、曲线拼图块、多碎片、噪声。

用法:
    python svg_tool/tests/gen_hard_cases.py --out svg_tool/test_out/hard
"""
from __future__ import annotations

import argparse
import json
import pathlib

import cv2
import numpy as np

W = H = 840


# ---------------- 形状 ----------------
def hexagon(cx, cy, r):
    return [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 7)[:-1]]


def _edge_bump(p0, p1, bump, n=24, inward=False):
    """沿 p0→p1 的边做一条正弦凸起/凹陷（模拟拼图圆头/缺口）。"""
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    L = np.hypot(dx, dy)
    nx, ny = -dy / L, dx / L  # 法向
    sgn = -1.0 if inward else 1.0
    pts = []
    for t in np.linspace(0, 1, n):
        b = bump * np.sin(np.pi * t)
        pts.append((x0 + dx * t + nx * b * sgn, y0 + dy * t + ny * b * sgn))
    return pts


def jigsaw(cx, cy, s):
    """方形 + 顶边外凸圆头 + 右边内凹缺口 → 曲线拼图块（含凹角）。"""
    p = [(cx - s, cy - s), (cx + s, cy - s), (cx + s, cy + s), (cx - s, cy + s)]
    top = _edge_bump(p[0], p[1], 0.5 * s, inward=False)
    right = _edge_bump(p[1], p[2], 0.4 * s, inward=True)
    left = _edge_bump(p[3], p[0], 0.0, inward=False)  # 直边
    return top + right + [(p[2]), (p[3])] + left


def triangles():
    return [
        [(150, 180), (430, 160), (300, 430)],
        [(520, 150), (760, 200), (640, 430)],
        [(150, 560), (400, 520), (300, 800)],
    ]


def many_small():
    out = []
    base = [(120, 120), (360, 130), (600, 120)]
    for r, y in enumerate([120, 380, 640]):
        for c, x in enumerate([120, 380, 640]):
            out.append([(x, y), (x + 150, y + 10), (x + 120, y + 140), (x + 20, y + 120)])
    return out


def grid_squares(n, side, margin):
    """n×n 个方格碎片（测碎片数量扩展）。"""
    step = (W - 2 * margin) / n
    out = []
    for r in range(n):
        for c in range(n):
            cx = margin + step * (c + 0.5)
            cy = margin + step * (r + 0.5)
            h = side / 2
            out.append([(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)])
    return out


def ragged(poly, amp=8.0, nper=8):
    """把多边形边缘改成手撕般的锯齿/纤维状（法向随机+正弦抖动）。"""
    poly = np.array(poly, float)
    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        d = b - a
        L = np.hypot(d[0], d[1]) or 1.0
        nx, ny = -d[1] / L, d[0] / L
        for k in range(nper):
            t = k / nper
            px, py = a + (b - a) * t
            off = np.random.uniform(-amp, amp) + amp * 0.5 * np.sin((i + t) * 5.1)
            out.append((px + nx * off, py + ny * off))
    return out


def add_hairs(img, poly, color, n=30, length=(3, 13)):
    """在边缘画细短毛刺（手撕纸的纤维），测形态学能不能清掉、别把碎片切碎。"""
    poly = np.array(poly, float)
    m = len(poly)
    for _ in range(n):
        i = np.random.randint(m)
        a, b = poly[i], poly[(i + 1) % m]
        t = np.random.rand()
        p = a + (b - a) * t
        d = b - a
        L = np.hypot(d[0], d[1]) or 1.0
        nx, ny = -d[1] / L, d[0] / L
        sgn = 1 if np.random.rand() > 0.5 else -1
        ln = np.random.uniform(*length)
        q = p + np.array([nx, ny]) * sgn * ln
        cv2.line(img, tuple(np.round(p).astype(int)), tuple(np.round(q).astype(int)), color, 1)


def palette(n):
    """远离背景绿的高对比配色（避免碎片颜色撞背景导致整块隐身）。"""
    base = [
        (210, 50, 50), (245, 200, 50), (60, 90, 220), (200, 70, 200),
        (245, 140, 50), (50, 190, 210), (150, 80, 220), (235, 95, 150),
        (120, 45, 200), (225, 170, 65), (70, 120, 235), (210, 60, 120),
    ]
    return [base[i % len(base)] for i in range(n)]


def draw_letter(w, h):
    """画一张「手写的信」：米白信纸 + 深色手写笔迹（随机连笔波浪线）。"""
    img = np.zeros((h, w, 3), np.float32)
    img[:] = (250, 249, 242)
    img += np.random.normal(0, 2.5, img.shape)      # 纸张细微纹理
    ink = (24, 26, 34)                               # 深色笔迹（相对蓝桌布仍是前景）
    y = 108
    while y < h - 70:
        x = 55
        while x < w - 90:
            wl = np.random.randint(55, 150)          # 一个「词」的宽度
            pts = []
            for k in range(18):
                t = k / 18
                pts.append((x + wl * t,
                            y + np.sin(t * np.pi * 2.3 + np.random.rand()) * np.random.uniform(4, 13)))
            cv2.polylines(img, [np.round(np.array(pts)).astype(np.int32)], False, ink, 3, cv2.LINE_AA)
            x += wl + np.random.randint(16, 38)      # 词间距
        y += 76
    return img


def tear_letter_into_scene(letter, scene, nx, ny, gap=4):
    """把信纸按 nx×ny 撕成小块（抖动边 + 腐蚀出缝隙），笔迹随碎片一起被撕开。返回块数。"""
    H2, W2 = letter.shape[:2]
    ox = (scene.shape[1] - W2) // 2
    oy = (scene.shape[0] - H2) // 2
    sub = scene[oy:oy + H2, ox:ox + W2]
    kern = np.ones((2 * gap + 1, 2 * gap + 1), np.uint8)
    n = 0
    for gy in range(ny):
        for gx in range(nx):
            x0, x1 = W2 * gx / nx, W2 * (gx + 1) / nx
            y0, y1 = H2 * gy / ny, H2 * (gy + 1) / ny
            cell = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            rg = ragged(cell, amp=2.5, nper=5)       # 手撕般的抖动边
            m = np.zeros((H2, W2), np.uint8)
            cv2.fillPoly(m, [np.round(np.array(rg)).astype(np.int32)], 255)
            m = cv2.erode(m, kern)                    # 缩出缝隙 → 碎片彼此分离
            sub[m > 0] = letter[m > 0]               # 把信纸上这块的像素贴过来（笔迹连续）
            n += 1
    return n


# ---------------- 合成 ----------------
def new_bg(color, grad=False):
    img = np.zeros((H, W, 3), np.float32)
    img[:] = color
    if grad:
        gx = np.linspace(0.68, 1.16, W, dtype=np.float32)[None, :, None]
        img = img * gx
    return img


def draw(img, polys, colors):
    for poly, col in zip(polys, colors):
        cv2.fillPoly(img, [np.round(np.array(poly)).astype(np.int32)], col)
    return img


def add_shadow(img, polys, dx=14, dy=20, strength=0.55):
    sh = np.zeros((H, W), np.float32)
    for poly in polys:
        cv2.fillPoly(sh, [np.round(np.array(poly) + [dx, dy]).astype(np.int32)], 1.0)
    sh = cv2.GaussianBlur(sh, (0, 0), 22)
    return img * (1.0 - strength * sh[..., None])


def add_marks(img, polys, color=(30, 30, 30), n=12):
    for poly in polys:
        poly = np.array(poly)
        x0, y0 = poly.min(0)
        x1, y1 = poly.max(0)
        for _ in range(n):
            cx = np.random.uniform(x0 + 40, x1 - 40)
            cy = np.random.uniform(y0 + 40, y1 - 40)
            cv2.circle(img, (int(cx), int(cy)), int(np.random.uniform(4, 11)), color, -1)
    return img


def save(name, img, gt, note, outdir):
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 94])[1].tofile(str(outdir / f"{name}.jpg"))
    np.random.seed(0)
    (outdir / f"{name}.json").write_text(
        json.dumps({"name": name, "pieces": gt, "note": note}, ensure_ascii=False), encoding="utf-8"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="svg_tool/test_out/hard")
    outdir = pathlib.Path(ap.parse_args().out)
    outdir.mkdir(parents=True, exist_ok=True)
    np.random.seed(0)
    tri = triangles()
    tri_cols = [(215, 70, 60), (240, 205, 70), (90, 120, 230)]

    # 1. 绿幕基础
    img = new_bg((60, 190, 70))
    save("green_basic", draw(img, tri, tri_cols), 3, "绿幕 + 3 块（基线）", outdir)

    # 2. 白底
    img = new_bg((245, 246, 248))
    save("white_bg", draw(img, tri, tri_cols), 3, "白色背景（走 Lab ΔE）", outdir)

    # 3. 绿幕 + 阴影 + 渐变光
    img = new_bg((60, 190, 70), grad=True)
    img = add_shadow(img, tri)
    img = draw(img, tri, tri_cols)
    save("shadow_green", img, 3, "绿幕 + 强阴影 + 渐变光（阴影/光照只改亮度）", outdir)

    # 4. 印刷暗点（碎片上有黑色印刷点，不能把碎片抠出洞/切碎）
    img = new_bg((60, 190, 70))
    img = draw(img, tri, tri_cols)
    img = add_marks(img, tri, color=(25, 25, 25), n=14)
    save("printed_marks", img, 3, "碎片上有黑色印刷点（应仍是 3 块）", outdir)

    # 5. 低对比（碎片颜色接近背景绿，默认 ΔE=40 会漏，需调低阈值）
    img = new_bg((70, 190, 80), grad=True)
    low = [(95, 170, 95), (120, 200, 120), (100, 175, 105)]  # 相对背景 dE≈22/20/24（低但可分）
    save("low_contrast", draw(img, tri, low), 3, "低对比（碎片≈背景色，需 deltaE≈15）", outdir)

    # 6. 近距离碎片（留 ~18px 缝，应仍分开为 3 块）
    close = [[(150, 170), (410, 160), (280, 400)],
             [(440, 160), (700, 180), (570, 410)],
             [(300, 470), (560, 460), (430, 760)]]
    img = new_bg((60, 190, 70))
    save("close_pieces", draw(img, close, [c for c in tri_cols[:3]]), 3, "近距离碎片（约 18px 缝）", outdir)

    # 7. 曲线拼图块（圆头 + 缺口，测曲线轮廓面积精度）
    jp = jigsaw(420, 420, 220)
    img = new_bg((60, 190, 70))
    save("curved_jigsaw", draw(img, [jp], [(210, 90, 70)]), 1, "曲线拼图块（圆头+缺口）", outdir)

    # 8. 多碎片（9 块）
    img = new_bg((60, 190, 70))
    ms = many_small()
    cols = [(200 + (i * 37) % 55, 70 + (i * 53) % 150, 60 + (i * 91) % 180) for i in range(len(ms))]
    save("many_small", draw(img, ms, cols), 9, "9 块小碎片", outdir)

    # 9. 噪声（盐噪 + 高斯噪）
    img = new_bg((60, 190, 70))
    img = draw(img, tri, tri_cols)
    noise = np.random.normal(0, 14, img.shape).astype(np.float32)
    img = img + noise
    sp = np.random.rand(H, W)
    img[sp < 0.0016] = 235
    img[sp > 0.9984] = 20
    save("noise", img, 3, "椒盐 + 高斯噪声（形态学应清掉）", outdir)

    # 10. 多碎片 4×4 = 16 块
    img = new_bg((60, 190, 70))
    g16 = grid_squares(4, 150, 60)
    cols = palette(len(g16))
    save("pieces_16", draw(img, g16, cols), 16, "16 块（4×4 方格）", outdir)

    # 11. 多碎片 5×5 = 25 块
    img = new_bg((60, 190, 70))
    g25 = grid_squares(5, 120, 50)
    cols = palette(len(g25))
    save("pieces_25", draw(img, g25, cols), 25, "25 块（5×5 方格）", outdir)

    # 12. 手撕毛边（锯齿 + 纤维 + 软边模糊）
    torn = [ragged(p, amp=9, nper=8) for p in triangles()]
    img = new_bg((60, 190, 70))
    img = draw(img, torn, tri_cols)
    for p, col in zip(torn, tri_cols):
        add_hairs(img, p, col, n=36)
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    save("torn_edges", img, 3, "手撕毛边（锯齿+纤维+软边）", outdir)

    # 13. 手撕 + 多碎片（9 块）
    base = many_small()
    torn9 = [ragged(p, amp=5, nper=6) for p in base]
    img = new_bg((60, 190, 70))
    cols = palette(len(torn9))
    img = draw(img, torn9, cols)
    for p, col in zip(torn9, cols):
        add_hairs(img, p, col, n=18, length=(2, 9))
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    save("torn_many", img, 9, "手撕毛边 + 9 块", outdir)

    # 14. 手写的信被撕成特别碎的小碎片（7×7 = 49 块，笔迹随碎片撕开）
    letter = draw_letter(760, 760)
    img = new_bg((50, 80, 130))   # 蓝灰桌布背景
    nfrag = tear_letter_into_scene(letter, img, nx=7, ny=7, gap=4)
    save("letter_torn", img, nfrag, f"手写信撕成 {nfrag} 块小碎片（7×7）", outdir)

    # 15. 撕得更碎（10×10 = 100 块，极限压测分辨率/最小面积）
    letter2 = draw_letter(780, 780)
    img = new_bg((50, 80, 130))
    nf2 = tear_letter_into_scene(letter2, img, nx=10, ny=10, gap=6)
    save("letter_torn_fine", img, nf2, f"手写信撕得更碎 {nf2} 块（10×10）", outdir)

    print(f"已生成 15 个 case 到 {outdir}")
    for f in sorted(outdir.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        print(f"  {d['name']:<16} gt={d['pieces']}  {d['note']}")


if __name__ == "__main__":
    main()
