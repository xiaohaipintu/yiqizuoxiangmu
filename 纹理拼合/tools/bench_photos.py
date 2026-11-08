"""用 纹理拼合/照片/ 里的真实照片跑纹理拼合，统计成功率。

算法与 纹理拼合/index.html 完全一致：
    横缝代价 = mean((a.R - b.L)^2)      竖缝代价 = mean((a.B - b.T)^2)
    分支限界按行优先逐格填、超界剪枝
每张照片：切成 3×3 → 随机重编号 perm（求解器拿不到）→ 求解 → 用 perm 映射回原图判对错。

用法：
    python 纹理拼合/tools/bench_photos.py [照片目录]
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

N, SIZE = 3, 600
TILE = SIZE // N


def edges_of(t):
    """每侧取 2 列/行：外列算接缝差异，内列算局部自然变化（归一化基线）。"""
    return {
        "Lo": t[:, 0, :].astype(np.float32), "Li": t[:, 1, :].astype(np.float32),
        "Ro": t[:, -1, :].astype(np.float32), "Ri": t[:, -2, :].astype(np.float32),
        "To": t[0, :, :].astype(np.float32), "Ti": t[1, :, :].astype(np.float32),
        "Bo": t[-1, :, :].astype(np.float32), "Bi": t[-2, :, :].astype(np.float32),
    }


def _energy(u, v):
    d = u - v
    return float(np.mean(np.sum(d * d, axis=-1)))


def seam_lr(a, b):
    """a 在左、b 在右：接缝差异 / 两侧自身局部变化。与页面 seamLR() 一致。"""
    raw = _energy(a["Ro"], b["Lo"])
    base = 0.5 * (_energy(a["Ro"], a["Ri"]) + _energy(b["Li"], b["Lo"])) + 1.0
    return raw / base


def seam_tb(a, b):
    """a 在上、b 在下。与页面 seamTB() 一致。"""
    raw = _energy(a["Bo"], b["To"])
    base = 0.5 * (_energy(a["Bo"], a["Bi"]) + _energy(b["Ti"], b["To"])) + 1.0
    return raw / base


def solve(tiles):
    """只接收像素块，不接收任何位置信息。"""
    n = N * N
    e = [edges_of(t) for t in tiles]
    h = np.full((n, n), np.inf)
    v = np.full((n, n), np.inf)
    for a in range(n):
        for b in range(n):
            if a != b:
                h[a, b] = seam_lr(e[a], e[b])
                v[a, b] = seam_tb(e[a], e[b])

    best = [float("inf"), None]
    used = [False] * n
    pos = [-1] * n

    def dfs(k, cost):
        if cost >= best[0]:
            return
        if k == n:
            best[0], best[1] = cost, pos[:]
            return
        r, c = divmod(k, N)
        for t in range(n):
            if used[t]:
                continue
            add = (h[pos[k - 1], t] if c > 0 else 0.0) + (v[pos[k - N], t] if r > 0 else 0.0)
            if cost + add >= best[0]:
                continue
            used[t] = True
            pos[k] = t
            dfs(k + 1, cost + add)
            used[t] = False
            pos[k] = -1

    dfs(0, 0.0)
    return best[1], best[0], h, v


def load(path: Path) -> np.ndarray | None:
    im = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        return None
    if im.shape[0] != SIZE or im.shape[1] != SIZE:
        im = cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return im


def main() -> int:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "纹理拼合/照片")
    files = sorted(folder.glob("*.jpg"))
    if not files:
        print("没找到照片：", folder)
        return 1

    rng = random.Random(20261031)
    rows = []
    t_all = time.time()

    for f in files:
        img = load(f)
        if img is None:
            print("读不了：", f.name)
            continue
        raw = [img[(i // N) * TILE:(i // N + 1) * TILE,
                   (i % N) * TILE:(i % N + 1) * TILE].copy() for i in range(N * N)]

        perm = list(range(N * N))
        rng.shuffle(perm)                        # 随机重编号：perm[t] = 这块来自原图第几格
        tiles = [raw[p] for p in perm]           # 求解器只看到这个

        order, cost, h, v = solve(tiles)
        right = all(perm[t] == k for k, t in enumerate(order))

        # 纹理丰富度：每块四条边像素的标准差（越大越有纹理可配）
        edge_std = float(np.mean([np.std(edges_of(t)["Ro"]) for t in raw]))
        allc = np.concatenate([h[~np.isinf(h)], v[~np.isinf(v)]])
        margin = float(np.median(allc) / max(cost / 12.0, 1e-9))
        rows.append({"name": f.name, "right": right, "cost": cost,
                     "edge_std": edge_std, "margin": margin})

    ms = (time.time() - t_all) * 1000
    ok = [r for r in rows if r["right"]]
    bad = [r for r in rows if not r["right"]]

    print(f"共 {len(rows)} 张，成功 {len(ok)}，失败 {len(bad)}，"
          f"成功率 {100.0 * len(ok) / max(1, len(rows)):.1f}%  总耗时 {ms / 1000:.1f}s")
    print()
    if bad:
        print("失败清单（边缘越平坦越容易拼错）：")
        for r in sorted(bad, key=lambda x: x["edge_std"]):
            print(f"  {r['name']}  边像素std={r['edge_std']:6.2f}  代价={r['cost']:10.1f}  判别倍数={r['margin']:5.2f}×")
        print()
    print("成功样本的边缘标准差：",
          "min %.2f  中位 %.2f  max %.2f"
          % (min(r["edge_std"] for r in ok) if ok else 0,
             float(np.median([r["edge_std"] for r in ok])) if ok else 0,
             max(r["edge_std"] for r in ok) if ok else 0))
    if bad:
        print("失败样本的边缘标准差：",
              "min %.2f  中位 %.2f  max %.2f"
              % (min(r["edge_std"] for r in bad),
                 float(np.median([r["edge_std"] for r in bad])),
                 max(r["edge_std"] for r in bad)))
    print()
    print("判别倍数（随机硬拼代价中位数 / 正确拼接的平均缝代价）分布：")
    ds = sorted(r["margin"] for r in rows)
    for q in (0, 10, 25, 50, 75, 90, 100):
        idx = min(len(ds) - 1, int(round(q / 100 * (len(ds) - 1))))
        print(f"  p{q:<3} {ds[idx]:6.2f}×")
    return 0


if __name__ == "__main__":
    sys.exit(main())
