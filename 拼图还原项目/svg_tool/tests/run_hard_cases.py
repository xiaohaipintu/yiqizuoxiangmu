"""对 gen_hard_cases.py 生成的硬核测试集跑「实拍抠组件」回归。

用 headless Edge 打开 puzzle_svg.html?selftest=real，抓 #report 里的 SELFTEST_RESULT，
把识别出的碎片数和 json 真值对比，打印通过/失败表。

用法:
    python svg_tool/tests/gen_hard_cases.py --out svg_tool/test_out/hard
    python svg_tool/tests/run_hard_cases.py --cases svg_tool/test_out/hard
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import subprocess
import sys

EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "msedge",
    "chrome",
    "google-chrome",
]


def find_browser(explicit: str | None) -> str:
    if explicit:
        return explicit
    for c in EDGE_CANDIDATES:
        if pathlib.Path(c).exists() or c in ("msedge", "chrome", "google-chrome"):
            return c
    raise SystemExit("没找到 Edge/Chrome，可 --browser 指定路径")


def file_url(p: pathlib.Path) -> str:
    return "file:///" + p.resolve().as_posix()


def run_case(browser: str, html: pathlib.Path, img: pathlib.Path, delta_e: float,
             profile: pathlib.Path, min_area: float = 0.0015) -> dict | None:
    url = (file_url(html) + f"?selftest=real&deltaE={delta_e}&minArea={min_area}"
           f"&img=" + file_url(img))
    cmd = [browser, "--headless=new", "--disable-gpu", "--no-sandbox",
           "--allow-file-access-from-files", f"--user-data-dir={profile}",
           "--virtual-time-budget=25000", "--dump-dom", url]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout
    except subprocess.TimeoutExpired:
        return None
    key = "SELFTEST_RESULT "
    i = out.find(key)
    if i < 0:
        return None
    j = out.find(" END_SELFTEST", i)
    try:
        return json.loads(out[i + len(key):j])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="svg_tool/test_out/hard")
    ap.add_argument("--html", default="svg_tool/puzzle_svg.html")
    ap.add_argument("--browser", default=None)
    ap.add_argument("--deltaE", type=float, default=40.0)
    args = ap.parse_args()

    browser = find_browser(args.browser)
    html = pathlib.Path(args.html)
    cases_dir = pathlib.Path(args.cases)
    profile = cases_dir / ".edge_iter"
    if profile.exists():
        import shutil
        shutil.rmtree(profile, ignore_errors=True)

    # 每个 case 的阈值覆盖（低对比需要更小 deltaE）
    delta_override = {"low_contrast": 15.0}  # 低对比碎片用更低阈值才抠得出

    rows = []
    npass = 0
    for jf in sorted(cases_dir.glob("*.json")):
        gt = json.loads(jf.read_text(encoding="utf-8"))
        img = cases_dir / (gt["name"] + ".jpg")
        if not img.exists():
            continue
        de = delta_override.get(gt["name"], args.deltaE)
        res = run_case(browser, html, img, de, profile)
        if res is None or "details" not in res:
            rows.append((gt["name"], gt["pieces"], None, de, "ERROR", gt["note"]))
            continue
        d = res["details"]
        got = d.get("pieceCount")
        errs = [p["areaErrPct"] for p in d.get("pieces", []) if p.get("areaErrPct") is not None]
        maxerr = max(errs) if errs else None
        ok = (got == gt["pieces"]) and d.get("svgValid") and d.get("pieceSVGbad") == 0
        npass += 1 if ok else 0
        rows.append((gt["name"], gt["pieces"], got, de, "PASS" if ok else "FAIL",
                     f"maxAreaErr={maxerr}"))

    total = len(rows)
    print(f"\n{'case':<16}{'gt':>4}{'got':>5}{'dE':>6}  {'status':<6}  detail")
    print("-" * 78)
    for name, gt, got, de, status, detail in rows:
        print(f"{name:<16}{gt:>4}{str(got):>5}{de:>6.0f}  {status:<6}  {detail}")
    print("-" * 78)
    print(f"通过 {npass}/{total}")
    return 0 if npass == total else 1


if __name__ == "__main__":
    sys.exit(main())
