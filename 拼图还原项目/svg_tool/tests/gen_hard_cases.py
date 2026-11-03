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

    print(f"已生成 9 个 case 到 {outdir}")
    for f in sorted(outdir.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        print(f"  {d['name']:<16} gt={d['pieces']}  {d['note']}")


if __name__ == "__main__":
    main()
