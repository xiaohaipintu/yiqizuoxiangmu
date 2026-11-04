// 临时校验：抽出页面里的纯几何引擎，在 Node 里跑切割用例
const fs = require('fs');
const path = require('path');
const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
const m = html.match(/<script id="cut-engine">([\s\S]*?)<\/script>/);
if (!m) { console.error('FAIL: 找不到 #cut-engine 脚本块'); process.exit(1); }
const code = m[1];
new Function(code + '\nif(typeof CutEngine==="undefined")throw new Error("no CutEngine");')();
const CutEngine = new Function(code + '\nreturn CutEngine;')();

let fail = 0;
function ok(name, cond, extra) {
  console.log((cond ? 'OK   ' : 'FAIL ') + name + (extra ? '  ' + extra : ''));
  if (!cond) fail++;
}

// ---- 1. 不切割 → 1 块
{
  const ps = CutEngine.computePieces({ width: 100, height: 100, cuts: [], res: 600 });
  const a = ps.length === 1 ? Math.abs(CutEngine.area(ps[0].rings[0])) : 0;
  ok('无切割 → 1 块', ps.length === 1, `pieces=${ps.length}`);
  ok('无切割 → 面积≈10000', Math.abs(a - 10000) < 60, `area=${a.toFixed(1)}`);
  ok('无切割 → 矩形 4~8 顶点', ps[0] && ps[0].rings[0].length <= 8, `pts=${ps[0] && ps[0].rings[0].length}`);
}

// ---- 2. 一条竖线 → 2 块，面积各半
{
  const ps = CutEngine.computePieces({
    width: 100, height: 100, res: 900,
    cuts: [[[50, -1], [50, 101]]],
  });
  const areas = ps.map((p) => Math.abs(CutEngine.area(p.rings[0]))).sort((a, b) => a - b);
  ok('竖线 → 2 块', ps.length === 2, `pieces=${ps.length}`);
  ok('竖线 → 面积各半', Math.abs(areas[0] - 5000) < 120 && Math.abs(areas[1] - 5000) < 120,
     `areas=${areas.map((x) => x.toFixed(0)).join('/')}`);
  const bbs = ps.map((p) => `${p.bbox.minX.toFixed(1)}..${p.bbox.maxX.toFixed(1)}`).join(' , ');
  const left = ps.find((p) => p.bbox.minX < 1), right = ps.find((p) => p.bbox.maxX > 99);
  ok('竖线 → 两块 x 范围分开', left && right && left.bbox.maxX <= 50.6 && right.bbox.minX >= 49.4, bbs);
}

// ---- 3. 3×3 网格 → 9 块
{
  const cuts = [];
  for (let i = 1; i < 3; i++) cuts.push([[100 * i / 3, -1], [100 * i / 3, 101]]);
  for (let j = 1; j < 3; j++) cuts.push([[-1, 100 * j / 3], [101, 100 * j / 3]]);
  const t0 = Date.now();
  const ps = CutEngine.computePieces({ width: 100, height: 100, cuts, res: 900 });
  const ms = Date.now() - t0;
  const areas = ps.map((p) => Math.abs(CutEngine.area(p.rings[0])));
  const maxDev = Math.max(...areas.map((a) => Math.abs(a - 10000 / 9) / (10000 / 9)));
  ok('3×3 → 9 块', ps.length === 9, `pieces=${ps.length} ${ms}ms`);
  ok('3×3 → 每块面积≈1111', maxDev < 0.06, `maxDev=${(maxDev * 100).toFixed(1)}%`);
  const shapes = ps.map((p) => p.rings.length === 1 && p.rings[0].length <= 8);
  ok('3×3 → 每块仍是简单矩形', shapes.every(Boolean), `${shapes.filter(Boolean).length}/9`);
}

// ---- 4. 对角折线（45°）→ 2 块
{
  const ps = CutEngine.computePieces({
    width: 100, height: 100, res: 1200,
    cuts: [[[0, 0], [50, 50], [100, 100]]],
  });
  const areas = ps.map((p) => Math.abs(CutEngine.area(p.rings[0]))).sort((a, b) => a - b);
  ok('对角折线 → 2 块', ps.length === 2, `pieces=${ps.length}`);
  ok('对角折线 → 面积各半', Math.abs(areas[0] - 5000) < 300 && Math.abs(areas[1] - 5000) < 300,
     `areas=${areas.map((x) => x.toFixed(0)).join('/')}`);
}

// ---- 5. 内部闭合方框 → 2 块（外圈带洞）
{
  const box = [[30, 30], [70, 30], [70, 70], [30, 70], [30, 30]];
  const ps = CutEngine.computePieces({ width: 100, height: 100, cuts: [box], res: 900 });
  const outer = ps.find((p) => p.rings.length > 1);
  const inner = ps.find((p) => p.rings.length === 1 && Math.abs(CutEngine.area(p.rings[0])) < 3000);
  const innerA = inner ? Math.abs(CutEngine.area(inner.rings[0])) : 0;
  ok('闭合方框 → 2 块', ps.length === 2, `pieces=${ps.length}`);
  ok('闭合方框 → 内块面积≈1600', Math.abs(innerA - 1600) < 200, `area=${innerA.toFixed(0)}`);
  ok('闭合方框 → 外块含洞(2 环)', !!outer, outer ? `rings=${outer.rings.length}` : 'none');
}

// ---- 6. 所有环闭合 / 面积符号一致
{
  const cuts = [[[0, 20], [100, 20]], [[10, 0], [10, 100]], [[0, 60], [100, 60]], [[70, 0], [70, 100]]];
  const ps = CutEngine.computePieces({ width: 100, height: 100, cuts, res: 800 });
  const signs = new Set(ps.map((p) => Math.sign(CutEngine.area(p.rings[0]))));
  ok('多刀 → 9 块（3 横带 × 3 竖列）', ps.length === 9, `pieces=${ps.length}`);
  ok('所有外环同向（环闭合）', signs.size === 1, `signs=${[...signs].join(',')}`);
  const total = ps.reduce((s, p) => s + p.rings.reduce((t, r, i) => t + (i === 0 ? 1 : -1) * Math.abs(CutEngine.area(r)), 0), 0);
  ok('碎片总面积 ≈ 画布面积（切缝损耗 <2%）', Math.abs(total - 10000) / 10000 < 0.02,
     `total=${total.toFixed(0)}`);}

// ---- 7. 大图性能（1200×800，6×4 网格）
{
  const cuts = [];
  for (let i = 1; i < 6; i++) cuts.push([[1200 * i / 6, -1], [1200 * i / 6, 801]]);
  for (let j = 1; j < 4; j++) cuts.push([[-1, 800 * j / 4], [1201, 800 * j / 4]]);
  const t0 = Date.now();
  const ps = CutEngine.computePieces({ width: 1200, height: 800, cuts, res: 1400 });
  const ms = Date.now() - t0;
  ok('1200×800 6×4 → 24 块', ps.length === 24, `pieces=${ps.length} ${ms}ms`);
  ok('大图耗时 < 1500ms', ms < 1500, `${ms}ms`);
}

console.log(fail ? `\n${fail} FAILED` : '\nALL PASSED');
process.exit(fail ? 1 : 0);
