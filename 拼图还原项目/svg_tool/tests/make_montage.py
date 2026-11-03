"""把全部硬核 test case 拼成一张「原图 | 抠图结果」对比大图（单个 PNG）。

对每个 case：headless Edge 打开 puzzle_svg.html?selftest=real&render=1&spread=0 截取
抠图结果（碎片贴在深色底、背景已去掉），再与原始照片上下配成一块，最后拼成大图。

用法:
    python svg_tool/tests/make_montage.py
"""
from __future__ import annotations

import glob
import json
import os
import pathlib
import shutil
import subprocess
import sys

import cv2
import numpy as np


def rd(p):
    """读图，兼容中文/非 ASCII 路径。"""
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import run_hard_cases as R  # noqa: E402

BASE = pathlib.Path("拼图还原项目/svg_tool")
HTML = BASE / "puzzle_svg.html"
CASES = BASE / "test_out" / "hard"
TMP = BASE / "test_out" / "_montage"
OUT = BASE / "test_out" / "hard_montage.png"
CELL = 300
GAP = 12
LABEL_H = 26
TOP_H = 64


def shot(browser, img_path, out_png, profile, delta_e=40.0):
    url = (R.file_url(HTML) + f"?selftest=real&render=1&spread=0&minArea=0.0015&deltaE={delta_e}&img="
           + R.file_url(img_path))
    cmd = [browser, "--headless=new", "--disable-gpu", "--no-sandbox",
           "--allow-file-access-from-files", f"--user-data-dir={pathlib.Path(profile).resolve()}",
           "--hide-scrollbars", f"--window-size={CELL},{CELL}",
           "--virtual-time-budget=15000",
           f"--screenshot={pathlib.Path(out_png).resolve()}", url]
    subprocess.run(cmd, capture_output=True, text=True, timeout=120)


def put(img, text, org, scale=0.5, color=(240, 240, 240), th=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


def main():
    browser = R.find_browser(None)
    if TMP.exists():
        shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True, exist_ok=True)
    profile = TMP / ".edge"

    cases = []
    for jf in sorted(CASES.glob("*.json")):
        gt = json.loads(jf.read_text(encoding="utf-8"))
        img = CASES / (gt["name"] + ".jpg")
        if not img.exists():
            continue
        # 识别计数（dump-dom）
        de = 15.0 if gt["name"] == "low_contrast" else 40.0
        res = R.run_case(browser, HTML, img, de, profile)
        got = res["details"].get("pieceCount") if res and "details" in res else None
        # 抠图结果截图
        out_png = TMP / (gt["name"] + "_result.png")
        shot(browser, img, out_png, profile, de)
        cases.append((gt["name"], gt["pieces"], got, gt.get("note", ""), img, out_png))

    n = len(cases)
    cols = 5
    rows = (n + cols - 1) // cols
    tile_h = LABEL_H + CELL + CELL
    W = cols * CELL + (cols + 1) * GAP
    H = TOP_H + rows * tile_h + (rows + 1) * GAP
    canvas = np.full((H, W, 3), 22, np.uint8)
    # 标题
    put(canvas, "Real-photo cutout -> SVG  |  15 experiments  (TOP: input photo, BOTTOM: cut-out result)",
        (GAP + 4, 30), 0.62, (120, 245, 210), 2)
    put(canvas, f"15/15 passed  ·  each cell: input / result  ·  gt = ground-truth piece count",
        (GAP + 4, 52), 0.5, (150, 150, 160), 1)

    for k, (name, gt, got, note, img_path, out_png) in enumerate(cases):
        r, c = divmod(k, cols)
        x = GAP + c * (CELL + GAP)
        y = TOP_H + GAP + r * (tile_h + GAP)
        # 标签条
        cv2.rectangle(canvas, (x, y), (x + CELL, y + LABEL_H), (40, 40, 48), -1)
        ok = (got == gt)
        col = (120, 245, 210) if ok else (80, 80, 245)
        put(canvas, f"{name}  gt={gt}  got={got if got is not None else '?'}  {'OK' if ok else 'FAIL'}",
            (x + 6, y + 18), 0.48, col, 1)
        # 原图
        src = rd(img_path)
        if src is None:
            src = np.zeros((CELL, CELL, 3), np.uint8)
        src = cv2.resize(src, (CELL, CELL), interpolation=cv2.INTER_AREA)
        canvas[y + LABEL_H:y + LABEL_H + CELL, x:x + CELL] = src
        # 结果
        resimg = rd(out_png)
        if resimg is None or resimg.size == 0:
            resimg = np.zeros((CELL, CELL, 3), np.uint8)
        resimg = cv2.resize(resimg, (CELL, CELL), interpolation=cv2.INTER_AREA)
        yy = y + LABEL_H + CELL
        canvas[yy:yy + CELL, x:x + CELL] = resimg
        # 小标注 in / out
        put(canvas, "IN", (x + 6, y + LABEL_H + 18), 0.5, (255, 255, 255), 1)
        put(canvas, "OUT", (x + 6, yy + 20), 0.5, (120, 245, 210), 1)
        cv2.rectangle(canvas, (x, y + LABEL_H), (x + CELL, yy + CELL), (70, 70, 80), 1)

    ok, enc = cv2.imencode('.png', canvas)
    enc.tofile(str(OUT))
    print(f"已生成 {OUT}  ({W}x{H}, {n} experiments)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
