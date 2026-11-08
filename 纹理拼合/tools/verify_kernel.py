"""把 纹理拼合/index.html 里的真实函数抽出来，用 quickjs 真跑一遍。

为什么需要这个脚本：开发环境里没有浏览器、也没有 Node，所以页面的内联 JS
平时只做了「括号配对」这种静态检查，并没有真正执行过。本脚本用 quickjs
（`pip install quickjs`）把页面里的求解内核**逐字符抽出来**执行，用来回答
「它到底是靠纹理算出来的，还是靠下标顺序蒙对的」。

抽取的函数：edgeOf / seamLR / seamTB / solveByTexture（原封不动，逐字符来自文件）
不抽取：任何 canvas / DOM / perm 相关逻辑 —— 求解器本来也碰不到它们。
脚本还会断言：抽出来的内核文本里不含 `perm`。

用法：
    python 纹理拼合/tools/verify_kernel.py
"""
import json
import re
import sys

import quickjs

html = open("纹理拼合/index.html", encoding="utf-8").read()
script = re.search(r"<script>(.*)</script>", html, re.S).group(1)

start = script.index("/* ---------------- 3) 取四边像素")
end = script.index("/* ---------------- 5) 打乱")
kernel = script[start:end]
print("抽出的求解内核：%d 行" % (kernel.count("\n") + 1))
assert "function solveByTexture" in kernel
assert "perm" not in kernel, "求解内核里不该出现 perm！"
print("检查：内核中不含 perm 变量 -> 通过")

harness = r"""
var N = 3, SIZE = 600, TILE = SIZE / N;

/* performance.now 在 quickjs 里没有，用计数器顶替（只用于计时展示） */
var performance = { now: (function () { var t = 0; return function () { t += 1; return t; }; })() };

function lcg(seed) {
  var s = seed >>> 0;
  return function () {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/* 程序生成一张 600x600 的正方形纹理图（不用 canvas），低频+高频叠加 */
function makeImage(seed) {
  var rnd = lcg(seed);
  var W = SIZE;
  var d = new Uint8ClampedArray(W * W * 4);
  var ph = [];
  for (var k = 0; k < 8; k++) ph.push(rnd() * 6.2831853);
  var fq = [];
  for (var k2 = 0; k2 < 8; k2++) fq.push(0.006 + rnd() * 0.28);
  for (var y = 0; y < W; y++) {
    for (var x = 0; x < W; x++) {
      var r = 128, g = 128, b = 128;
      r += 70 * Math.sin(x * fq[0] + ph[0]) + 45 * Math.sin((x + y) * fq[1] + ph[1]);
      g += 70 * Math.cos(y * fq[2] + ph[2]) + 45 * Math.sin((x - y) * fq[3] + ph[3]);
      b += 70 * Math.sin((x * fq[4]) + (y * fq[5]) + ph[4]) + 45 * Math.cos((x + 2 * y) * fq[6] + ph[5]);
      var i = (y * W + x) * 4;
      d[i] = r; d[i + 1] = g; d[i + 2] = b; d[i + 3] = 255;
    }
  }
  return { width: W, height: W, data: d };
}

function sliceTiles(src) {
  var out = [];
  for (var idx = 0; idx < N * N; idx++) {
    var r = Math.floor(idx / N), c = idx % N;
    var buf = new Uint8ClampedArray(TILE * TILE * 4);
    for (var y = 0; y < TILE; y++) {
      for (var x = 0; x < TILE; x++) {
        var so = ((r * TILE + y) * SIZE + (c * TILE + x)) * 4;
        var to = (y * TILE + x) * 4;
        buf[to] = src.data[so]; buf[to + 1] = src.data[so + 1];
        buf[to + 2] = src.data[so + 2]; buf[to + 3] = 255;
      }
    }
    out.push({ width: TILE, height: TILE, data: buf });
  }
  return out;
}

function randPerm(n, rnd) {
  var a = [];
  for (var i = 0; i < n; i++) a.push(i);
  for (var j = a.length - 1; j > 0; j--) {
    var k = Math.floor(rnd() * (j + 1));
    var t = a[j]; a[j] = a[k]; a[k] = t;
  }
  return a;
}

var results = [];
for (var trial = 1; trial <= 10; trial++) {
  var seed = 1000 + trial * 7919;
  var rnd = lcg(seed ^ 0x5bf03635);
  var img = makeImage(seed);
  var raw = sliceTiles(img);

  var perm = randPerm(N * N, rnd);          // perm[t] = 编号 t 来自原图第几格
  var tiles = [];
  for (var t = 0; t < N * N; t++) tiles.push(raw[perm[t]]);   // 只把像素交给求解器

  var res = solveByTexture(tiles);          // ← 页面里的原函数

  var right = true;
  for (var k = 0; k < N * N; k++) if (perm[res.order[k]] !== k) right = false;
  var identity = true;
  for (var k2 = 0; k2 < N * N; k2++) if (res.order[k2] !== k2) identity = false;

  results.push({
    trial: trial,
    perm: perm.join(""),
    out: res.order.join(""),
    right: right,
    identityOutput: identity,
    cost: typeof res.cost === "number" ? Math.round(res.cost) : null
  });
}

JSON.stringify(results);
"""

ctx = quickjs.Context()
ctx.eval(kernel)          # 只加载求解内核
out = ctx.eval(harness)
rows = json.loads(out)

ok = 0
ident = 0
print()
print("== 用 quickjs 执行页面里的真实 solveByTexture ==")
for r in rows:
    ok += 1 if r["right"] else 0
    ident += 1 if r["identityOutput"] else 0
    print("第 %2d 次  perm=%s  输出=%s  %s  输出恰好是0..8:%s  代价=%s"
          % (r["trial"], r["perm"], r["out"],
             "✓ 归位" if r["right"] else "✗ 错位",
             "是(可疑)" if r["identityOutput"] else "否",
             r["cost"]))
print()
print("归位：%d/%d" % (ok, len(rows)))
print("输出恰好等于 0..8 的次数：%d/%d" % (ident, len(rows)))
