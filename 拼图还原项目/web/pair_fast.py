"""把 pieces 转成 pairsearch 的文本输入，跑编译好的 C++ 求解器，产出 solutions JSON。

- 边长匹配只当信息用：容差 max(rel×短边, abs_px)，一条边可以与多条边匹配
- 不改任何缩放（piece 几何原样传进去）
- 输出格式与 beam_solutions.json 一致（gap_pct / placed / polys），上层不用改
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BIN = HERE.parent / "bin" / "pairsearch"


def to_text(data: dict) -> str:
    lines = [str(len(data["pieces"]))]
    for p in data["pieces"]:
        poly = p["polygon"]
        rows = [f"{int(p['id'])} {float(p['area']):.6f} {len(poly)}"]
        rows += [f"{float(x):.4f} {float(y):.4f}" for x, y in poly]
        lines.extend(rows)
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pieces", required=True)
    ap.add_argument("--out", required=True, help="输出目录（写 pair_solutions.json）")
    ap.add_argument("--rel", type=float, default=0.15, help="边长相对容差")
    ap.add_argument("--abs", type=float, default=9.0, help="边长绝对容差（px）")
    ap.add_argument("--beam", type=int, default=3000)
    ap.add_argument("--tops", type=int, default=50)
    ap.add_argument("--nodes", type=int, default=4000000)
    ap.add_argument("--time", type=float, default=20.0)
    args = ap.parse_args()

    if not BIN.exists():
        print(f"编译好的求解器不存在：{BIN}", file=sys.stderr)
        return 3

    data = json.loads(Path(args.pieces).read_text(encoding="utf-8"))
    txt = to_text(data)
    cmd = [str(BIN), "--rel", str(args.rel), "--abs", str(args.abs),
           "--beam", str(args.beam), "--tops", str(args.tops), "--nodes", str(args.nodes)]
    try:
        proc = subprocess.run(cmd, input=txt, capture_output=True, text=True,
                              timeout=args.time)
    except subprocess.TimeoutExpired:
        print(f"求解器超时（{args.time}s）", file=sys.stderr)
        return 4
    if proc.returncode != 0 or not proc.stdout.strip():
        print(f"求解器失败 rc={proc.returncode}\n{proc.stderr[-800:]}", file=sys.stderr)
        return 5

    doc = json.loads(proc.stdout)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    sols = [{"gap_pct": s["gap_pct"], "placed": s["placed"], "polys": s["polys"]}
            for s in doc.get("solutions") or []]
    (outdir / "pair_solutions.json").write_text(
        json.dumps({"n": len(sols), "solutions": sols}, ensure_ascii=False), encoding="utf-8")
    best = sols[0] if sols else None
    if best:
        print(f"配对求解：{len(sols)} 个完整解，最好 缝 {best['gap_pct']:.2f}%")
    else:
        print("配对求解：没有完整解")
    for line in proc.stderr.strip().splitlines()[-3:]:
        print("  " + line)
    return 0 if sols else 6


if __name__ == "__main__":
    raise SystemExit(main())
