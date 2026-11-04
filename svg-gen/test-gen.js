// 生成器回归：DOM shim 里跑 buildArt，检查良构 / 元素丰富度 / 纹理层是否覆盖全图
const fs = require('fs');
const path = require('path');

const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
const code = html.match(/<script>([\s\S]*)<\/script>/)[1];

function mkStub(name) {
  const t = {
    _n: name, innerHTML: '', textContent: '', value: '', checked: true, options: [],
    addEventListener() {}, appendChild() {}, remove() {}, click() {},
    classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
    style: {}, dataset: {}, setAttribute() {}, getAttribute: () => null,
    querySelector: () => mkStub(name + '-q'), querySelectorAll: () => [],
  };
  return t;
}
const registry = {};
global.document = {
  getElementById: (id) => (registry[id] = registry[id] || mkStub(id)),
  querySelector: (s) => mkStub(s),
  querySelectorAll: () => [],
  addEventListener() {},
  createElement: () => mkStub('el'),
  body: mkStub('body'),
};
global.window = { addEventListener() {}, matchMedia: () => ({ matches: false, addEventListener() {} }) };
global.location = { search: '', href: 'file:///x/index.html' };
global.history = { replaceState() {} };
global.navigator = { clipboard: { writeText: () => Promise.resolve() } };
global.URL = { createObjectURL: () => '', revokeObjectURL() {} };
global.Blob = function () {};
global.Image = function () {};
global.requestAnimationFrame = (fn) => setTimeout(fn, 0);
global.localStorage = { getItem: () => null, setItem() {} };

// 只取 buildArt 相关：把 IIFE 末尾改成挂到 window 上再跑
const patched = code.replace(/\}\)\(\);\s*$/, '  window.__gen = { buildArt: buildArt, STYLES: STYLES };\n})();\n');
new Function('document', 'window', patched)(global.document, global.window);
const api = global.window.__gen;
if (!api) { console.error('FAIL: 拿不到 buildArt'); process.exit(1); }

const STYLES = Object.keys(api.STYLES);
const SEEDS = ['a1', 'zzz', '2026', 'k7', 'q9', 'seed-x', 'luna', '42'];
const SIZES = [[900, 900], [1200, 800], [800, 1200]];

let fail = 0, total = 0;
const rows = [];
for (const style of STYLES) {
  let worst = { elements: Infinity, size: 0, seed: '' };
  for (let si = 0; si < SEEDS.length; si++) {
    const seed = SEEDS[si];
    const size = SIZES[si % SIZES.length];
    const out = api.buildArt({ seed, style, size, grain: true });
    const svg = out.svg;
    total++;
    const problems = [];
    if (!svg.includes('xmlns="http://www.w3.org/2000/svg"')) problems.push('no xmlns');
    if (!/viewBox="0 0 \d+ \d+"/.test(svg)) problems.push('no viewBox');
    if (/undefined|NaN|Infinity/.test(svg)) problems.push('NaN/undefined');
    if ((svg.match(/</g) || []).length !== (svg.match(/>/g) || []).length) problems.push('tags unbalanced');
    const refs = [...svg.matchAll(/url\(#([\w-]+)\)/g)].map((m) => m[1]);
    for (const r of new Set(refs)) if (!new RegExp(`id="${r}"`).test(svg)) problems.push('missing def #' + r);

    // 纹理层必须存在且覆盖面足够细（结构上保证没有大面积连续纯色）
    if (!svg.includes('id="tx-hatch"')) problems.push('no hatch');
    if (!svg.includes('id="tx-grain"')) problems.push('no grain');
    const step = /pattern id="tx-hatch" width="([\d.]+)"/.exec(svg);
    const stepVal = step ? parseFloat(step[1]) : 1e9;
    const minSide = Math.min(size[0], size[1]);
    if (stepVal > 22.5) problems.push('hatch too coarse: ' + stepVal);
    const hatchAt = svg.indexOf('fill="url(#tx-hatch)"');
    if (hatchAt < 0) problems.push('hatch not applied');

    // 元素丰富度
    const shapes = (svg.match(/<(path|circle|rect|polygon|ellipse|line|polyline)\b/g) || []).length;
    if (shapes < 150) problems.push('too few elements: ' + shapes);
    const sizeKb = Math.round(svg.length / 1024);
    if (sizeKb > 160) problems.push('too heavy: ' + sizeKb + 'KB');

    if (shapes < worst.elements) { worst = { elements: shapes, size: sizeKb, seed }; }
    if (problems.length) {
      fail++;
      console.log(`FAIL ${style}/${seed}/${size.join('x')}: ${problems.join(', ')}`);
    }
  }
  rows.push([style, api.STYLES[style].name, worst.elements, worst.size, worst.seed]);
}

console.log('\n| 风格 | 名称 | 最少元素数 | 体积KB | 对应种子 |');
console.log('| --- | --- | --- | --- | --- |');
for (const r of rows) console.log(`| ${r[0]} | ${r[1]} | ${r[2]} | ${r[3]} | ${r[4]} |`);
console.log(`\n${total - fail}/${total} passed`);
process.exit(fail ? 1 : 0);
