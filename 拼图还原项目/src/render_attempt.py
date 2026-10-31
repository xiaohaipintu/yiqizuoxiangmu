"""把闭环搜索找到的（不完美的）拼法画出来，标注空穴率。

用法: python src/render_attempt.py --solutions out3/solutions.json --out out3 --top 3
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from assemble import draw  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solutions", default="out3/solutions.json")
    ap.add_argument("--out", default="out3")
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()

    outdir = pathlib.Path(args.out)
    data = json.loads(pathlib.Path(args.solutions).read_text(encoding="utf-8"))
    sols = data["solutions"][: args.top]
    if not sols:
        print("没有完整拼法可画")
        return 1

    tiles = []
    seen = set()
    for i, sol in enumerate(sols, 1):
        polys = [np.array(p, float) for p in sol["polys"]]
        ids = [r[0] for r in sol["placed"]]
        key = hash(np.round(np.vstack(polys), -1).tobytes())
        if key in seen:          # 同一套摆法（只是生成顺序不同）
            continue
        seen.add(key)
        im = draw(polys, ids, size=460, margin=14)
        h, w = im.shape[:2]
        canvas = np.full((h + 82, max(w + 20, 470), 3), 255, np.uint8)
        canvas[40:40 + h, 10:10 + w] = im
        cv2.putText(canvas, f"attempt {i}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"holes {sol['gap_pct']:.1f}%  (not seamless)", (10, 40 + h + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 60, 170), 1, cv2.LINE_AA)
        tiles.append(canvas)

    W = sum(t.shape[1] for t in tiles) + 12 * len(tiles)
    H = max(t.shape[0] for t in tiles) + 20
    sheet = np.full((H + 70, W + 16, 3), 255, np.uint8)
    cv2.putText(sheet, "equal-length-only closure attempts (edge lengths within 4%)", (14, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    x = 8
    for t in tiles:
        sheet[62:62 + t.shape[0], x:x + t.shape[1]] = t
        x += t.shape[1] + 12
    cv2.imwrite(str(outdir / "40_attempts.png"), sheet)
    print(f"写出 {outdir / '40_attempts.png'} {sheet.shape[1]}x{sheet.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

