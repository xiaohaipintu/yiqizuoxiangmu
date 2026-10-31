"""第三步：把求解结果画出来。

输出:
    out/03_assembly.png       拼好后的图形（每块不同颜色 + 编号 + 顶点）
    out/04_before_after.png   左边原始照片、右边拼装结果，同色对应
    out/05_assembly.gif       碎片从照片位置滑到正确位置的动画（可选）
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import geometry as G  # noqa: E402

PALETTE = [
    (56, 56, 224), (0, 140, 255), (60, 180, 75), (200, 120, 20),
    (200, 60, 200), (30, 200, 200), (140, 60, 160), (0, 90, 200),
]


def draw_assembly(sol: dict, size: int = 1200, margin: int = 60) -> np.ndarray:
    polys = [np.array(p["polygon_world"], dtype=float) for p in sol["pieces"]]
    allp = np.vstack(polys)
    w = allp[:, 0].max() - allp[:, 0].min()
    h = allp[:, 1].max() - allp[:, 1].min()
    s = (size - 2 * margin) / max(w, h)
    canvas = np.full((int(h * s) + 2 * margin, int(w * s) + 2 * margin, 3), 255, np.uint8)

    for p, poly in zip(sol["pieces"], polys):
        q = (poly - [allp[:, 0].min(), allp[:, 1].min()]) * s + margin
        qi = q.astype(np.int32)
        col = PALETTE[(p["id"] - 1) % len(PALETTE)]
        cv2.fillPoly(canvas, [qi], col)
        cv2.polylines(canvas, [qi], True, (40, 40, 40), 2, cv2.LINE_AA)
        for v in qi:
            cv2.circle(canvas, tuple(int(x) for x in v), 3, (30, 30, 30), -1, cv2.LINE_AA)
        c = q.mean(axis=0).astype(int)
        txt = f"#{p['id']}"
        cv2.putText(canvas, txt, (c[0] - 14, c[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(canvas, txt, (c[0] - 14, c[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def draw_source(img: np.ndarray, pieces: list[dict]) -> np.ndarray:
    out = img.copy()
    for p in pieces:
        poly = np.array(p["polygon"], dtype=np.int32)
        col = PALETTE[(p["id"] - 1) % len(PALETTE)]
        overlay = out.copy()
        cv2.fillPoly(overlay, [poly], col)
        out = cv2.addWeighted(overlay, 0.75, out, 0.25, 0)
        cv2.polylines(out, [poly], True, (255, 255, 255), 2, cv2.LINE_AA)
        c = tuple(int(v) for v in p["centroid"])
        cv2.putText(out, f"#{p['id']}", (c[0] - 18, c[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(out, f"#{p['id']}", (c[0] - 18, c[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 1, cv2.LINE_AA)
    return out


def make_gif(img_path: str, src_pieces: list[dict], sol: dict, gif_path: pathlib.Path, frames: int = 22, size: int = 900):
    img = cv2.imread(img_path)
    target = draw_assembly(sol, size=size, margin=40)
    th, tw = target.shape[:2]
    W = max(img.shape[1], tw) + 40
    H = max(img.shape[0], th) + 40

    # 源/目标坐标对应：源 = 照片中的多边形（按屎心对齐），目标 = 拼装坐标
    src_polys = {p["id"]: np.array(p["polygon"], float) for p in src_pieces}
    dst_polys = {p["id"]: np.array(p["polygon_world"], float) for p in sol["pieces"]}
    all_dst = np.vstack(list(dst_polys.values()))
    dw, dh = all_dst[:, 0].max() - all_dst[:, 0].min(), all_dst[:, 1].max() - all_dst[:, 1].min()
    scale_dst = (size - 80) / max(dw, dh)

    out_frames = []
    for f in range(frames + 1):
        t = f / frames
        t = t * t * (3 - 2 * t)  # 缓动
        canvas = np.full((H, W, 3), 255, np.uint8)
        # 淡出照片
        ghost = img.copy()
        cv2.putText(ghost, "before", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.addWeighted(ghost, 1.0 - 0.6 * t, canvas[: img.shape[0], : img.shape[1]], 0.6 * t, 0, canvas[: img.shape[0], : img.shape[1]])

        for pid, sp in src_polys.items():
            dp = dst_polys[pid]
            dq = (dp - [all_dst[:, 0].min(), all_dst[:, 1].min()]) * scale_dst + 40
            # 源多边形平移到与目标尺度一致的画布位置
            sq = sp * 0.55
            mix = (1 - t) * sq + t * dq
            col = PALETTE[(pid - 1) % len(PALETTE)]
            pts = mix.astype(np.int32)
            cv2.fillPoly(canvas, [pts], col)
            cv2.polylines(canvas, [pts], True, (40, 40, 40), 2, cv2.LINE_AA)
        if t > 0.6:
            cv2.putText(canvas, "after", (W - 160, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        out_frames.append(canvas)

    gif_frames = [cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) for fr in out_frames]
    try:
        import imageio.v2 as imageio

        imageio.mimsave(str(gif_path), gif_frames, duration=0.09, loop=0)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"GIF 生成失败（可能缺 imageio）：{exc}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pieces", default="out/pieces.json")
    ap.add_argument("--solution", default="out/solution.json")
    ap.add_argument("--image", default="input/pieces.jpg")
    ap.add_argument("--out", default="out")
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args()

    outdir = pathlib.Path(args.out)
    data = json.loads(pathlib.Path(args.pieces).read_text(encoding="utf-8"))
    pieces = data["pieces"]
    sol = json.loads(pathlib.Path(args.solution).read_text(encoding="utf-8"))

    assembly = draw_assembly(sol)
    cv2.imwrite(str(outdir / "03_assembly.png"), assembly)
    print(f"写出 {outdir / '03_assembly.png'}  ({assembly.shape[1]}x{assembly.shape[0]})")

    img = cv2.imread(args.image)
    src = draw_source(img, pieces)
    # 横向拼成 before/after
    h = 900
    src_s = cv2.resize(src, (int(src.shape[1] * h / src.shape[0]), h))
    asm_s = cv2.resize(assembly, (int(assembly.shape[1] * h / assembly.shape[0]), h))
    gap = np.full((h, 24, 3), 255, np.uint8)
    combo = np.hstack([src_s, gap, asm_s])
    cv2.putText(combo, "BEFORE", (24, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(combo, "AFTER", (src_s.shape[1] + 48, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.imwrite(str(outdir / "04_before_after.png"), combo)
    print(f"写出 {outdir / '04_before_after.png'}  ({combo.shape[1]}x{combo.shape[0]})")

    if not args.no_gif:
        ok = make_gif(args.image, pieces, sol, outdir / "05_assembly.gif")
        if ok:
            print(f"写出 {outdir / '05_assembly.gif'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
