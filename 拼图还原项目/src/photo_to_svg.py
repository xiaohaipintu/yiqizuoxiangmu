"""实拍照片 → 逐块 SVG（真实抠图 + 轮廓导出）。

复用 segment.py 里已经调到很稳的抠像 / 多边形拟合逻辑，把每块碎片导出成
「真实纹理裁剪 + 真实轮廓」的 SVG，既给整图 SVG，也给逐块 SVG。

用法:
    python src/photo_to_svg.py --image input/pieces5.jpg --out out5 --concave
    python src/photo_to_svg.py --image 拼图软件测试/实验1测试图.jpg --out /tmp/x

输出（--out 目录下）:
    photo_puzzle.svg      整图：深色底 + 每块按真实轮廓裁剪原图 + 青色描边
    piece_<id>.svg         逐块：viewBox 正好是这块的外接框，透明底
    photo_svg_report.json  统计（块数 / 顶点数 / 面积误差）
"""
from __future__ import annotations

import argparse
import base64
import json
import pathlib
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import segment as S  # noqa: E402
import geometry as G  # noqa: E402


def read_color(path: str) -> np.ndarray | None:
    """读取图片，兼容中文/非 ASCII 路径（cv2.imread 在 Windows 上会失败）。"""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def pick_mask(bgr: np.ndarray, args) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """按 segment.py 的自适应逻辑选出前景掩膜，返回 (mask, alpha, bg)。"""
    delta, bg = S.background_delta(bgr)
    if args.bg:
        manual = S.parse_bg(args.bg)
        if manual is not None:
            bg = manual
    use_chroma = S.is_chroma_bg(bg) if args.mode == "auto" else (args.mode == "chroma")

    def chroma_mask() -> np.ndarray:
        d = S.chroma_delta(bgr, bg)
        m = (d > args.chroma_hi).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

    def lab_mask() -> np.ndarray:
        return S.auto_mask(bgr, args.delta_e)

    if args.mode == "red":
        mask = S.red_mask(bgr)
        use_chroma = False
    elif args.mode == "chroma":
        mask = chroma_mask()
        use_chroma = True
    elif args.mode == "auto":
        primary, alt = (chroma_mask(), lab_mask()) if use_chroma else (lab_mask(), chroma_mask())
        if S.mask_quality(alt, args.min_area) > S.mask_quality(primary, args.min_area) + 0.5:
            mask, use_chroma = alt, not use_chroma
        else:
            mask = primary
    else:
        mask = lab_mask()
        use_chroma = False

    mask, _ns, _na = S.drop_specks(mask, args.min_area)
    if use_chroma:
        alpha = S.chroma_alpha(bgr, bg, args.chroma_lo, args.chroma_hi, mask)
    else:
        d, _ = S.background_delta(bgr)
        alpha = S.soft_alpha(mask, d)
    return mask, alpha, bg


def extract_pieces(bgr: np.ndarray, mask: np.ndarray, args) -> list[dict]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    pieces = []
    for lbl in range(1, n):
        area = float(stats[lbl, cv2.CC_STAT_AREA])
        if area < args.min_area:
            continue
        comp = (labels == lbl).astype(np.uint8) * 255
        comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        res = S.polygon_of_component(comp, area, allow_concave=args.concave)
        if res is None:
            continue
        poly, _fits = res
        poly = np.asarray(poly, dtype=float)
        pieces.append({
            "polygon": poly,
            "pixel_area": area,
            "area": float(G.polygon_area(poly)),
            "centroid": np.asarray(G.centroid(poly), dtype=float),
        })
    pieces.sort(key=lambda p: (round(p["centroid"][1] / 120.0), p["centroid"][0]))
    for i, p in enumerate(pieces, start=1):
        p["id"] = i
    return pieces


def polygon_d(poly: np.ndarray, sx: float, sy: float, tx: float, ty: float) -> str:
    pts = [(x * sx + tx, y * sy + ty) for x, y in poly]
    return "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in pts) + " Z"


def build_svgs(bgr: np.ndarray, pieces: list[dict], tex_max: int) -> tuple[str, dict[int, str], int, int]:
    h, w = bgr.shape[:2]
    s = min(1.0, tex_max / max(h, w))
    tw, th = max(1, int(round(w * s))), max(1, int(round(h * s)))
    tex = cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_AREA) if s < 1.0 else bgr
    ok, enc = cv2.imencode(".jpg", tex, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    data_url = "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode("ascii")
    sx, sy = tw / w, th / h

    # 整图 SVG：深色底 + 每块裁剪纹理 + 描边
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {tw} {th}" width="{tw}" height="{th}">',
        "  <defs>",
    ]
    for p in pieces:
        parts.append(
            f'    <clipPath id="cp-{p["id"]}" clipPathUnits="userSpaceOnUse">'
            f'<path d="{polygon_d(p["polygon"], sx, sy, 0, 0)}"/></clipPath>'
        )
    parts.append("  </defs>")
    parts.append(f'  <rect width="{tw}" height="{th}" fill="#0f0f13"/>')
    for p in pieces:
        d = polygon_d(p["polygon"], sx, sy, 0, 0)
        parts.append("  <g>")
        parts.append(f'    <image href="{data_url}" x="0" y="0" width="{tw}" height="{th}" '
                     f'preserveAspectRatio="none" clip-path="url(#cp-{p["id"]})"/>')
        parts.append(f'    <path d="{d}" fill="none" stroke="rgba(94,231,196,0.85)" stroke-width="1.5"/>')
        parts.append("  </g>")
    parts.append("</svg>")
    combined = "\n".join(parts)

    # 逐块 SVG：viewBox = 该块外接框
    each: dict[int, str] = {}
    for p in pieces:
        xs = p["polygon"][:, 0] * sx
        ys = p["polygon"][:, 1] * sy
        minx, miny = float(xs.min()), float(ys.min())
        bw = max(1, int(np.ceil(xs.max() - minx)))
        bh = max(1, int(np.ceil(ys.max() - miny)))
        d = polygon_d(p["polygon"], sx, sy, -minx, -miny)
        each[p["id"]] = "\n".join([
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {bw} {bh}" width="{bw}" height="{bh}">',
            '  <defs><clipPath id="cp" clipPathUnits="userSpaceOnUse">'
            f'<path d="{d}"/></clipPath></defs>',
            f'  <image href="{data_url}" x="{-minx:.2f}" y="{-miny:.2f}" width="{tw}" height="{th}" '
            'preserveAspectRatio="none" clip-path="url(#cp)"/>',
            f'  <path d="{d}" fill="none" stroke="rgba(94,231,196,0.9)" stroke-width="1.5"/>',
            "</svg>",
        ])
    return combined, each, tw, th


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="out_svg")
    ap.add_argument("--min-area", type=float, default=1500.0)
    ap.add_argument("--mode", choices=("auto", "red", "chroma", "lab"), default="auto")
    ap.add_argument("--delta-e", type=float, default=40.0)
    ap.add_argument("--chroma-lo", type=float, default=10.0)
    ap.add_argument("--chroma-hi", type=float, default=22.0)
    ap.add_argument("--bg", default=None)
    ap.add_argument("--concave", action="store_true")
    ap.add_argument("--tex-max", type=int, default=1600, help="内嵌纹理长边上限（控制 SVG 体积）")
    args = ap.parse_args()

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    bgr = read_color(args.image)
    if bgr is None:
        print(f"读不到图片: {args.image}")
        return 1
    print(f"图像尺寸: {bgr.shape[1]}x{bgr.shape[0]}")

    mask, _alpha, bg = pick_mask(bgr, args)
    if bg is not None:
        print("背景色 BGR %s" % np.asarray(bg).astype(int).tolist())
    pieces = extract_pieces(bgr, mask, args)
    if not pieces:
        print("没抠到任何碎片（试试调 --delta-e / --min-area / --concave）")
        return 1

    combined, each, tw, th = build_svgs(bgr, pieces, args.tex_max)
    (outdir / "photo_puzzle.svg").write_text(combined, encoding="utf-8")
    for pid, svg in each.items():
        (outdir / f"piece_{pid}.svg").write_text(svg, encoding="utf-8")

    # 自检：每个 SVG 必须是良构 XML
    def well_formed(text: str) -> bool:
        try:
            ET.fromstring(text)
            return True
        except ET.ParseError:
            return False

    report = {
        "image": args.image,
        "texture": [tw, th],
        "pieceCount": len(pieces),
        "combinedWellFormed": well_formed(combined),
        "piecesSVGWellFormed": all(well_formed(s) for s in each.values()),
        "pieces": [
            {
                "id": p["id"],
                "vertices": int(len(p["polygon"])),
                "area": round(p["area"], 1),
                "pixelArea": round(p["pixel_area"], 1),
                "areaErrPct": round(abs(p["area"] - p["pixel_area"]) / max(p["pixel_area"], 1) * 100, 2),
            }
            for p in pieces
        ],
    }
    (outdir / "photo_svg_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"抠出碎片 {len(pieces)} 块 → photo_puzzle.svg + piece_*.svg（纹理 {tw}x{th}）")
    for row in report["pieces"]:
        print(f"  #{row['id']:>2}  顶点 {row['vertices']:>2}  面积误差 {row['areaErrPct']:>5.1f}%")
    print(f"自检：整图 SVG 良构={report['combinedWellFormed']}，逐块全良构={report['piecesSVGWellFormed']}")
    return 0 if report["combinedWellFormed"] and report["piecesSVGWellFormed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
