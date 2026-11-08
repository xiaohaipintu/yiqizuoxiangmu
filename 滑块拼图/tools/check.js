// 临时校验脚本：抽取 index.html 里的 <script> 做语法检查 + 逻辑自测
const fs = require('fs');
const path = require('path');

const html = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');
const m = html.match(/<script>([\s\S]*?)<\/script>/);
if (!m) {
  console.error('没找到 script 标签');
  process.exit(1);
}
const code = m[1];

// 1) 语法检查
new Function(code);
console.log('✔ JS 语法通过');

// 2) 静态检查：DOM id 引用是否都存在
const ids = [...code.matchAll(/getElementById\('([^']+)'\)/g)].map((x) => x[1]);
const missing = ids.filter((id) => !html.includes('id="' + id + '"'));
console.log(missing.length ? '✘ 缺失的 id: ' + missing.join(', ') : '✔ 引用的 DOM id 全部存在 (' + ids.length + ' 个)');

// 3) 检查触摸/键盘/难度等关键绑定是否都在
const checks = [
  ['移动端 viewport', /name="viewport"[^>]*width=device-width/],
  ['安全区适配', /env\(safe-area-inset/],
  ['触摸滑动', /addEventListener\('touchstart'/],
  ['方向键', /ArrowLeft/],
  ['难度切换', /data-n="5"/],
  ['最佳纪录', /localStorage/],
  ['图片模式', /buildPicture/],
];
for (const [name, re] of checks) {
  console.log((re.test(html) ? '✔ ' : '✘ ') + name);
}

// 4) 逻辑自测：把 8-puzzle 核心算法抽出来跑一遍可解性
function neighbours(i, n) {
  const r = Math.floor(i / n), c = i % n, out = [];
  if (r > 0) out.push(i - n);
  if (r < n - 1) out.push(i + n);
  if (c > 0) out.push(i - 1);
  if (c < n - 1) out.push(i + 1);
  return out;
}
function solvable(place, n) {
  const seq = place.filter((v) => v !== 0);
  let inv = 0;
  for (let i = 0; i < seq.length; i++)
    for (let j = i + 1; j < seq.length; j++) if (seq[i] > seq[j]) inv++;
  if (n % 2 === 1) return inv % 2 === 0;
  const blankRowFromBottom = n - Math.floor(place.indexOf(0) / n);
  return (inv + blankRowFromBottom) % 2 === 1;
}
function simulate(n, rounds) {
  let bad = 0;
  for (let t = 0; t < rounds; t++) {
    const place = [];
    for (let i = 0; i < n * n - 1; i++) place.push(i + 1);
    place.push(0);
    let blank = n * n - 1, last = -1;
    for (let k = 0; k < n * n * 24; k++) {
      const nb = neighbours(blank, n).filter((i) => i !== last);
      const pick = nb[Math.floor(Math.random() * nb.length)];
      last = blank;
      place[blank] = place[pick];
      place[pick] = 0;
      blank = pick;
    }
    if (!solvable(place, n)) bad++;
  }
  return bad;
}
let totalBad = 0;
for (const n of [3, 4, 5]) {
  const bad = simulate(n, 300);
  totalBad += bad;
  console.log('✔ ' + n + '×' + n + ' 随机 300 局，不可解 ' + bad + ' 局');
}
process.exit(totalBad === 0 ? 0 : 1);
