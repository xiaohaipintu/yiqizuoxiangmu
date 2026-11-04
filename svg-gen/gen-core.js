var GenCore = (function () {
  'use strict';

  /* ---------------- 基础工具 ---------------- */
  const TAU = Math.PI * 2;
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const n1 = (v) => Math.round(v * 10) / 10;
  const n2 = (v) => Math.round(v * 100) / 100;

  const norm360 = (h) => ((h % 360) + 360) % 360;
  function hsl(h, s, l, a) {
    const H = Math.round(norm360(h));
    const S = Math.round(clamp(s, 0, 100));
    const L = Math.round(clamp(l, 0, 100));
    if (a === undefined || a >= 1) return `hsl(${H},${S}%,${L}%)`;
    return `hsla(${H},${S}%,${L}%,${n2(clamp(a, 0, 1))})`;
  }

  /* ---------------- 可复现随机数 ---------------- */
  function hashStr(str) {
    let h = 2166136261 >>> 0;
    for (let i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return h >>> 0;
  }
  function mulberry32(a) {
    return function () {
      a |= 0;
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function makeRand(seed) {
    const next = mulberry32(hashStr(String(seed)));
    const r = {
      next,
      float: (a = 0, b = 1) => a + (b - a) * next(),
      int: (a, b) => Math.floor(a + (b - a + 1) * next()),
      pick: (arr) => arr[Math.floor(next() * arr.length)],
      bool: (p = 0.5) => next() < p,
      sign: () => (next() < 0.5 ? -1 : 1),
    };
    return r;
  }

  /* ---------------- 配色 ---------------- */
  const SCHEMES = [
    [0, 24, 48, -24, -48],      // 邻近色
    [0, 180, 20, 200, -160],    // 补色
    [0, 120, 240, 60, 300],     // 三色
    [0, 150, 210, 30, -30],     // 分裂补色
    [0, 10, -10, 20, -20],      // 单色
    [0, 40, 80, 320, 280],      // 宽邻近
  ];

  function makePalette(r) {
    const baseHue = r.int(0, 359);
    const offsets = r.pick(SCHEMES);
    const satBase = r.float(58, 94);
    const isDark = r.bool(0.72);

    const colors = offsets.map((off) => {
      const h = norm360(baseHue + off);
      const s = clamp(satBase + r.float(-16, 6), 28, 98);
      const l = isDark ? r.float(46, 76) : r.float(26, 56);
      return { h, s, l };
    });

    const bgHue = norm360(baseHue + r.pick([0, 30, 180, 210, -150]));
    const bg = isDark
      ? hsl(bgHue, clamp(r.float(30, 60), 15, 70), r.float(6, 12))
      : hsl(bgHue, r.float(28, 60), r.float(93, 97));
    const bg2 = isDark
      ? hsl(norm360(bgHue + r.float(20, 70) * r.sign()), r.float(28, 62), r.float(10, 20))
      : hsl(norm360(bgHue + r.float(20, 70) * r.sign()), r.float(30, 62), r.float(84, 91));

    return { baseHue, colors, isDark, bg, bg2 };
  }

  /* ---------------- 风格：每种返回 { defs, body } ---------------- */

  // 1. 几何碎片（三角网格）
  function shards(r, { W, H, pal, cg }) {
    const cols = r.int(6, 11);
    const rows = Math.max(4, Math.round((cols * H) / W));
    const cw = W / cols;
    const ch = H / rows;
    const pts = [];
    for (let y = 0; y <= rows; y++) {
      const row = [];
      for (let x = 0; x <= cols; x++) {
        const isEdge = x === 0 || x === cols;
        const isTopBottom = y === 0 || y === rows;
        const jx = isEdge ? 0 : r.float(-0.34, 0.34) * cw;
        const jy = isTopBottom ? 0 : r.float(-0.34, 0.34) * ch;
        row.push([x * cw + jx, y * ch + jy]);
      }
      pts.push(row);
    }

    let body = '';
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        const a = pts[y][x], b = pts[y][x + 1], c = pts[y + 1][x], d = pts[y + 1][x + 1];
        const tris = r.bool() ? [[a, b, d], [a, d, c]] : [[a, b, c], [b, d, c]];
        for (const t of tris) {
          const col = pal.colors[r.int(0, pal.colors.length - 1)];
          const fill = cg && r.bool(0.5)
            ? cg[r.int(0, cg.length - 1)]
            : hsl(col.h + r.float(-5, 5), col.s + r.float(-10, 6), clamp(col.l + r.float(-16, 12), 12, 90));
          const p = t.map((pt) => `${n1(pt[0])},${n1(pt[1])}`).join(' ');
          body += `<polygon points="${p}" fill="${fill}" stroke="${fill}" stroke-width="0.7"/>`;
        }
      }
    }
    return { body };
  }

  // 2. 层叠波浪
  function waves(r, { W, H, pal, cg }) {
    const bands = r.int(9, 18);
    const asLine = r.bool(0.45);
    let body = '';

    for (let i = 0; i < bands; i++) {
      const t = (i + 0.5) / bands;
      const y0 = H * t + r.float(-H * 0.02, H * 0.02);
      const amp = H * r.float(0.02, 0.08);
      const segs = r.int(2, 4);
      const step = (W + 80) / segs;
      let d = `M -40 ${n1(y0)}`;
      for (let k = 0; k < segs; k++) {
        const x1 = -40 + step * k;
        const x2 = x1 + step;
        const c1 = y0 + r.float(-amp, amp);
        const c2 = y0 + r.float(-amp, amp);
        const ey = y0 + r.float(-amp * 0.5, amp * 0.5);
        d += ` C ${n1(x1 + step / 3)} ${n1(c1)}, ${n1(x1 + (step * 2) / 3)} ${n1(c2)}, ${n1(x2)} ${n1(ey)}`;
      }
      const col = pal.colors[i % pal.colors.length];
      const h = col.h + r.float(-6, 6);
      if (asLine) {
        body += `<path d="${d}" fill="none" stroke="${hsl(h, col.s, clamp(col.l + r.float(-6, 8), 20, 88), 0.85)}" stroke-width="${n1(Math.min(W, H) * r.float(0.004, 0.018))}" stroke-linecap="round"/>`;
      } else {
        const lShift = pal.isDark ? (bands - i) * 1.6 : -(bands - i) * 1.6;
        body += `<path d="${d} L ${W + 40} ${H + 40} L -40 ${H + 40} Z" fill="${cg && cg.length ? cg[i % cg.length] : hsl(h, col.s, clamp(col.l + lShift + r.float(-5, 5), 12, 92), 0.96)}" fill-opacity="0.95"/>`;
      }
    }
    return { body };
  }

  // 3. 点阵波纹
  function dots(r, { W, H, pal }) {
    const cols = r.int(14, 30);
    const rows = Math.max(8, Math.round((cols * H) / W));
    const cw = W / cols;
    const ch = H / rows;

    const terms = [];
    const tn = r.int(2, 4);
    for (let i = 0; i < tn; i++) {
      terms.push({
        fx: (r.float(0.6, 2.6) * TAU) / W,
        fy: (r.float(0.6, 2.6) * TAU) / H,
        ph: r.float(0, TAU),
        w: r.float(0.5, 1),
      });
    }
    const field = (x, y) => {
      let v = 0, s = 0;
      for (const t of terms) { v += Math.sin(x * t.fx + y * t.fy + t.ph) * t.w; s += t.w; }
      return (v / s + 1) / 2;
    };

    const hueShift = r.bool(0.5);
    let body = '';
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        const cx = (x + 0.5) * cw;
        const cy = (y + 0.5) * ch;
        const v = field(cx, cy);
        const rad = Math.min(cw, ch) * 0.5 * (0.1 + 0.86 * v);
        if (rad < 0.45) continue;
        const base = pal.colors[(x + y) % pal.colors.length];
        const h = hueShift ? base.h + v * 40 - 20 : base.h;
        body += `<circle cx="${n1(cx)}" cy="${n1(cy)}" r="${n2(rad)}" fill="${hsl(h, base.s, clamp(base.l + v * 10 - 5, 16, 88))}"/>`;
      }
    }
    return { body };
  }

  // 4. 光斑散景
  function bokeh(r, { W, H, pal }) {
    const count = r.int(24, 48);
    const minSide = Math.min(W, H);
    const blur = n1(minSide * r.float(0.008, 0.028));
    const defs = `<filter id="bk-blur" x="-40%" y="-40%" width="180%" height="180%"><feGaussianBlur stdDeviation="${blur}"/></filter>`;

    let body = `<g filter="url(#bk-blur)" style="mix-blend-mode:${pal.isDark ? 'screen' : 'multiply'}">`;
    for (let i = 0; i < count; i++) {
      const col = pal.colors[i % pal.colors.length];
      const rad = minSide * r.float(0.03, 0.22);
      body += `<circle cx="${n1(r.float(-0.05, 1.05) * W)}" cy="${n1(r.float(-0.05, 1.05) * H)}" r="${n1(rad)}" fill="${hsl(col.h + r.float(-12, 12), clamp(col.s + r.float(-10, 8), 30, 100), clamp(col.l + r.float(-6, 14), 30, 86), r.float(0.28, 0.72))}"/>`;
    }
    body += '</g>';
    return { defs, body };
  }

  // 5. 同心干涉
  function rings(r, { W, H, pal }) {
    const centers = [];
    const cn = r.int(3, 5);
    for (let i = 0; i < cn; i++) {
      centers.push({
        x: r.float(0.1, 0.9) * W,
        y: r.float(0.1, 0.9) * H,
        step: Math.min(W, H) * r.float(0.005, 0.015),
        ph: r.float(0, 1),
        w: r.float(0.6, 2.2),
      });
    }
    const maxR = Math.hypot(W, H) * 1.05;
    let body = '';
    centers.forEach((c, ci) => {
      const col = pal.colors[ci % pal.colors.length];
      const alpha = r.float(0.35, 0.75);
      body += '<g>';
      for (let rad = c.step * (0.4 + c.ph); rad < maxR; rad += c.step) {
        body += `<circle cx="${n1(c.x)}" cy="${n1(c.y)}" r="${n1(rad)}" fill="none" stroke="${hsl(col.h, col.s, clamp(col.l + r.float(-8, 8), 20, 88), alpha)}" stroke-width="${n2(c.w)}"/>`;
      }
      body += '</g>';
    });
    return { body };
  }

  // 6. 孟菲斯涂鸦
  function memphis(r, { W, H, pal }) {
    const minSide = Math.min(W, H);
    const count = r.int(34, 64);
    let body = '';
    for (let i = 0; i < count; i++) {
      const cx = r.float(0.02, 0.98) * W;
      const cy = r.float(0.02, 0.98) * H;
      const size = minSide * r.float(0.022, 0.095);
      const col = pal.colors[i % pal.colors.length];
      const fill = hsl(col.h + r.float(-4, 4), col.s, col.l, r.bool(0.75) ? 1 : 0.85);
      const t = `transform="translate(${n1(cx)},${n1(cy)}) rotate(${Math.round(r.float(0, 360))})"`;
      switch (r.int(0, 5)) {
        case 0:
          body += `<circle ${t} r="${n1(size / 2)}" fill="${fill}"/>`;
          break;
        case 1:
          body += `<rect ${t} x="${n1(-size / 2)}" y="${n1(-size / 2)}" width="${n1(size)}" height="${n1(size)}" rx="${r.bool() ? n1(size / 2) : 0}" fill="${fill}"/>`;
          break;
        case 2:
          body += `<polygon ${t} points="0,${n1(-size * 0.62)} ${n1(size * 0.56)},${n1(size * 0.46)} ${n1(-size * 0.56)},${n1(size * 0.46)}" fill="${fill}"/>`;
          break;
        case 3:
          body += `<path ${t} d="M ${n1(-size / 2)} 0 Q 0 ${n1(-size * 0.72)} ${n1(size / 2)} 0" fill="none" stroke="${fill}" stroke-width="${n1(size * 0.15)}" stroke-linecap="round"/>`;
          break;
        case 4:
          body += `<path ${t} d="M ${n1(-size / 2)} 0 H ${n1(size / 2)} M 0 ${n1(-size / 2)} V ${n1(size / 2)}" stroke="${fill}" stroke-width="${n1(size * 0.26)}" stroke-linecap="round"/>`;
          break;
        default:
          body += `<path ${t} d="M ${n1(-size / 2)} 0 A ${n1(size / 2)} ${n1(size / 2)} 0 0 1 ${n1(size / 2)} 0 Z" fill="${fill}"/>`;
      }
    }
    return { body };
  }

  // 7. 色块拼贴
  function blocks(r, { W, H, pal, cg }) {
    const gap = Math.min(W, H) * r.float(0.004, 0.012);
    const minSize = Math.min(W, H) * 0.07;
    const rects = [];
    const maxDepth = r.int(6, 9);

    (function split(x, y, w, h, depth) {
      const canSplit = depth > 0 && (w > minSize * 1.7 || h > minSize * 1.7) && r.bool(0.84);
      if (!canSplit) { rects.push([x, y, w, h]); return; }
      const horiz = Math.abs(w - h) > 1 ? w > h : r.bool();
      const t = r.float(0.3, 0.7);
      if (horiz) {
        const cut = x + w * t;
        split(x, y, w * t - gap / 2, h, depth - 1);
        split(cut + gap / 2, y, w * (1 - t) - gap / 2, h, depth - 1);
      } else {
        const cut = y + h * t;
        split(x, y, w, h * t - gap / 2, depth - 1);
        split(x, cut + gap / 2, w, h * (1 - t) - gap / 2, depth - 1);
      }
    })(0, 0, W, H, maxDepth);

    let body = '';
    for (const [x, y, w, h] of rects) {
      if (w <= 0.5 || h <= 0.5) continue;
      if (r.bool(0.1)) continue;
      const col = pal.colors[r.int(0, pal.colors.length - 1)];
      const bf = cg && cg.length ? cg[r.int(0, cg.length - 1)] : hsl(col.h + r.float(-4, 4), col.s, clamp(col.l + r.float(-12, 8), 14, 90));
      body += `<rect x="${n1(x)}" y="${n1(y)}" width="${n1(w)}" height="${n1(h)}" fill="${bf}"/>`;
      if (r.bool(0.16)) {
        const rr = Math.min(w, h) * r.float(0.12, 0.34);
        const c2 = pal.colors[r.int(0, pal.colors.length - 1)];
        body += `<circle cx="${n1(x + w / 2)}" cy="${n1(y + h / 2)}" r="${n1(rr)}" fill="${hsl(c2.h, c2.s, c2.l, 0.9)}"/>`;
      }
    }
    return { body };
  }

  // 8. 极光渐变
  function aurora(r, { W, H, pal }) {
    const count = r.int(8, 16);
    const blur = Math.min(W, H) * r.float(0.03, 0.075);
    let defs = `<filter id="au-blur" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="${n1(blur)}"/></filter>`;
    let body = '<g filter="url(#au-blur)">';

    for (let i = 0; i < count; i++) {
      const col = pal.colors[i % pal.colors.length];
      const h = col.h + r.float(-14, 14);
      const s = clamp(col.s + r.float(-8, 8), 40, 100);
      const l = clamp(col.l + r.float(-4, 12), 38, 82);
      const id = `au-g${i}`;
      defs += `<radialGradient id="${id}"><stop offset="0%" stop-color="${hsl(h, s, l, 0.95)}"/><stop offset="55%" stop-color="${hsl(h, s, l, 0.4)}"/><stop offset="100%" stop-color="${hsl(h, s, l, 0)}"/></radialGradient>`;
      const cx = r.float(0.05, 0.95) * W;
      const cy = r.float(0.05, 0.95) * H;
      const rx = W * r.float(0.18, 0.44);
      const ry = H * r.float(0.18, 0.44);
      body += `<ellipse cx="${n1(cx)}" cy="${n1(cy)}" rx="${n1(rx)}" ry="${n1(ry)}" fill="url(#${id})" transform="rotate(${n1(r.float(-45, 45))} ${n1(cx)} ${n1(cy)})"/>`;
    }
    body += '</g>';
    return { defs, body };
  }

  // 9. 花瓣线
  function petals(r, { W, H, pal }) {
    const cx = W / 2;
    const cy = H / 2;
    const lobe = r.int(6, 22);
    const ringsN = r.int(9, 20);
    const maxR = Math.min(W, H) * 0.47;
    const wob = r.float(0.06, 0.24);
    const phase = r.float(0, TAU);
    const asFill = r.bool(0.3);

    let body = '';
    for (let ring = ringsN; ring >= 1; ring--) {
      const baseR = (maxR * ring) / ringsN;
      const col = pal.colors[ring % pal.colors.length];
      const steps = Math.max(lobe * 8, 96);
      const pts = [];
      for (let i = 0; i <= steps; i++) {
        const th = (i / steps) * TAU;
        const rr = baseR * (1 - wob / 2 + (wob / 2) * Math.cos(lobe * th + phase));
        pts.push(`${n1(cx + Math.cos(th - Math.PI / 2) * rr)},${n1(cy + Math.sin(th - Math.PI / 2) * rr)}`);
      }
      const p = pts.join(' ');
      if (asFill) {
        body += `<polygon points="${p}" fill="${hsl(col.h, col.s, col.l, 0.16)}" stroke="${hsl(col.h, col.s, clamp(col.l + 8, 20, 92), 0.85)}" stroke-width="${n2(Math.min(W, H) * 0.0035)}"/>`;
      } else {
        body += `<polygon points="${p}" fill="none" stroke="${hsl(col.h, col.s, clamp(col.l + r.float(-4, 6), 20, 92), 0.9)}" stroke-width="${n2(Math.min(W, H) * r.float(0.002, 0.006))}"/>`;
      }
    }
    // 中心点缀
    const core = pal.colors[0];
    body += `<circle cx="${n1(cx)}" cy="${n1(cy)}" r="${n1(maxR * 0.045)}" fill="${hsl(core.h, core.s, clamp(core.l + 16, 30, 95))}"/>`;
    return { body };
  }

  // 10. 斜条纹
  function stripes(r, { W, H, pal, cg }) {
    const angle = r.pick([0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, -30, -45]);
    const count = r.int(16, 46);
    const span = Math.hypot(W, H) * 1.25;
    const weights = [];
    let sum = 0;
    for (let i = 0; i < count; i++) {
      const w = r.float(0.35, 1.8);
      weights.push(w);
      sum += w;
    }
    const scale = span / sum;

    let body = `<g transform="translate(${n1(W / 2)},${n1(H / 2)}) rotate(${angle})">`;
    let x = -span / 2;
    for (let i = 0; i < count; i++) {
      const w = weights[i] * scale;
      const col = pal.colors[i % pal.colors.length];
      body += `<rect x="${n1(x)}" y="${n1(-span / 2)}" width="${n2(w)}" height="${n1(span)}" fill="${cg && cg.length ? cg[i % cg.length] : hsl(col.h + r.float(-5, 5), col.s, clamp(col.l + r.float(-10, 8), 12, 90))}"/>`;
      x += w;
    }
    body += '</g>';
    return { body };
  }

  /* ---------------- 调色板 → 渐变色（大面积填充不再用纯色） ---------------- */
  function makeColorGradients(r, pal) {
    const defs = [];
    const fills = [];
    pal.colors.forEach((c, i) => {
      const d = r.pick([[1, 0.15], [0.15, 1], [1, 1], [-1, 1], [0.7, -1], [1, -0.6]]);
      const h1 = norm360(c.h + r.float(-32, 32));
      const h2 = norm360(c.h - r.float(8, 34));
      const l1 = clamp(c.l + r.float(12, 26), 12, 97);
      const l2 = clamp(c.l - r.float(6, 22), 8, 92);
      defs.push(`<linearGradient id="cg${i}" x1="0" y1="0" x2="${n2(d[0])}" y2="${n2(d[1])}">` +
        `<stop offset="0%" stop-color="${hsl(h1, clamp(c.s - 6, 20, 100), l1)}"/>` +
        `<stop offset="48%" stop-color="${hsl(c.h, c.s, c.l)}"/>` +
        `<stop offset="100%" stop-color="${hsl(h2, clamp(c.s + 8, 20, 100), l2)}"/></linearGradient>`);
      fills.push(`url(#cg${i})`);
    });
    return { defs: defs.join(''), fills };
  }

  /* ---------------- 质感叠加层：七层纹理（细斜线 / 交叉网 / 点阵 / 网格 / 微条纹 / 双频颗粒 / 碎点） ----------------
     目的：不放大也看得见纹理，任意局部都不存在「大面积连续纯色」。 */
  function richness(r, W, H, pal, strength) {
    const minSide = Math.min(W, H);
    const k = 0.3 + 0.7 * clamp(strength == null ? 1 : strength, 0, 1);
    const ink = pal.isDark ? '#ffffff' : '#0b0f18';
    const defs = [];
    const body = [];
    const layer = (fill, blend, extra) =>
      `<rect width="${n2(W)}" height="${n2(H)}" fill="${fill}" ${extra || ''} style="mix-blend-mode:${blend}"/>`;

    // 1) 细密斜线（最密的一层）
    const s1 = clamp(minSide * r.float(0.0032, 0.008), 2, 13);
    defs.push(`<pattern id="tx-hatch" width="${n2(s1)}" height="${n2(s1)}" patternUnits="userSpaceOnUse" patternTransform="rotate(${n1(r.float(0, 180))})">` +
      `<line x1="0" y1="0" x2="0" y2="${n2(s1)}" stroke="${ink}" stroke-opacity="${r.float(0.12, 0.26).toFixed(2)}" stroke-width="${n2(Math.max(0.35, s1 * 0.34))}"/></pattern>`);

    // 2) 第二向斜线（和第一层交叉成网）
    const s2 = clamp(minSide * r.float(0.007, 0.018), 4, 24);
    defs.push(`<pattern id="tx-hatch2" width="${n2(s2)}" height="${n2(s2)}" patternUnits="userSpaceOnUse" patternTransform="rotate(${n1(r.float(0, 180))})">` +
      `<line x1="0" y1="0" x2="0" y2="${n2(s2)}" stroke="${ink}" stroke-opacity="${r.float(0.06, 0.15).toFixed(2)}" stroke-width="${n2(Math.max(0.3, s2 * 0.2))}"/></pattern>`);

    // 3) 微点阵
    const s3 = clamp(minSide * r.float(0.007, 0.016), 4, 22);
    const r3 = Math.max(0.28, s3 * r.float(0.08, 0.2));
    const c0 = pal.colors[0];
    const dotInk = r.bool(0.55) ? ink : hsl(c0.h + r.float(-20, 20), clamp(c0.s, 35, 92), pal.isDark ? r.float(74, 92) : r.float(12, 30));
    defs.push(`<pattern id="tx-dots" width="${n2(s3)}" height="${n2(s3)}" patternUnits="userSpaceOnUse" patternTransform="rotate(${n1(r.float(0, 90))})">` +
      `<circle cx="${n2(s3 / 2)}" cy="${n2(s3 / 2)}" r="${n2(r3)}" fill="${dotInk}" fill-opacity="${r.float(0.2, 0.45).toFixed(2)}"/></pattern>`);

    // 4) 正交细网格
    const s4 = clamp(minSide * r.float(0.011, 0.028), 8, 48);
    defs.push(`<pattern id="tx-grid" width="${n2(s4)}" height="${n2(s4)}" patternUnits="userSpaceOnUse">` +
      `<path d="M 0 0 H ${n2(s4)} M 0 0 V ${n2(s4)}" fill="none" stroke="${ink}" stroke-opacity="${r.float(0.07, 0.15).toFixed(2)}" stroke-width="${n2(Math.max(0.4, s4 * 0.04))}"/></pattern>`);

    // 5) 微条纹（再一个方向，很淡，专治大面积平铺感）
    const s5 = clamp(minSide * r.float(0.004, 0.01), 3, 16);
    defs.push(`<pattern id="tx-lines" width="${n2(s5)}" height="${n2(s5)}" patternUnits="userSpaceOnUse" patternTransform="rotate(${n1(r.float(0, 180))})">` +
      `<rect x="0" y="0" width="${n2(s5)}" height="${n2(s5 * 0.45)}" fill="${ink}" fill-opacity="${r.float(0.05, 0.13).toFixed(2)}"/></pattern>`);

    // 6) 双频颗粒（粗 + 细，叠出胶片噪点感）
    defs.push(`<filter id="tx-grain" x="0" y="0" width="100%" height="100%">` +
      `<feTurbulence type="fractalNoise" baseFrequency="${r.float(0.5, 0.95).toFixed(2)}" numOctaves="3" stitchTiles="stitch"/>` +
      `<feColorMatrix type="saturate" values="0"/></filter>`);
    defs.push(`<filter id="tx-grain2" x="0" y="0" width="100%" height="100%">` +
      `<feTurbulence type="fractalNoise" baseFrequency="${r.float(1.4, 2.4).toFixed(2)}" numOctaves="2" stitchTiles="stitch"/>` +
      `<feColorMatrix type="saturate" values="0"/></filter>`);

    body.push(layer('url(#tx-hatch)', 'overlay'));
    body.push(layer('url(#tx-hatch2)', 'overlay'));
    body.push(layer('url(#tx-lines)', 'overlay'));
    body.push(layer('url(#tx-dots)', 'overlay'));
    body.push(layer('url(#tx-grid)', pal.isDark ? 'screen' : 'multiply'));
    body.push(layer('url(#tx-grain)', 'overlay', `opacity="${pal.isDark ? '0.26' : '0.18'}"`));
    body.push(layer('url(#tx-grain2)', 'overlay', `opacity="${pal.isDark ? '0.18' : '0.13'}"`));

    // 7) 碎点 / 短条 / 小弧 / 十字：把空白处也铺满元素
    const n = r.int(380, 700);
    const spk = [];
    for (let i = 0; i < n; i++) {
      const x = r.float(-0.01, 1.01) * W;
      const y = r.float(-0.01, 1.01) * H;
      const col = pal.colors[r.int(0, pal.colors.length - 1)];
      const s = minSide * r.float(0.0012, 0.0075);
      const c = hsl(col.h + r.float(-14, 14), col.s, clamp(col.l + r.float(-22, 22), 12, 94), r.float(0.2, 0.7));
      switch (r.int(0, 3)) {
        case 0:
          spk.push(`<circle cx="${n1(x)}" cy="${n1(y)}" r="${n2(s)}" fill="${c}"/>`);
          break;
        case 1:
          spk.push(`<rect x="${n1(x)}" y="${n1(y)}" width="${n2(s * 2.8)}" height="${n2(s * 0.85)}" fill="${c}" transform="rotate(${Math.round(r.float(0, 180))} ${n1(x)} ${n1(y)})"/>`);
          break;
        case 2:
          spk.push(`<path d="M ${n1(x)} ${n1(y)} q ${n1(s * 4)} ${n1(-s * 3)} ${n1(s * 8)} 0" fill="none" stroke="${c}" stroke-width="${n2(Math.max(0.35, s * 0.7))}" stroke-linecap="round"/>`);
          break;
        default:
          spk.push(`<path d="M ${n1(x - s * 2)} ${n1(y)} H ${n1(x + s * 2)} M ${n1(x)} ${n1(y - s * 2)} V ${n1(y + s * 2)}" fill="none" stroke="${c}" stroke-width="${n2(Math.max(0.3, s * 0.55))}" stroke-linecap="round"/>`);
      }
    }
    body.push(`<g>${spk.join('')}</g>`);

    return { defs: defs.join(''), body: '<g opacity="' + k.toFixed(2) + '">' + body.join('') + '</g>' };
  }

  const STYLES = {
    shards: { name: '几何碎片', draw: shards },
    waves: { name: '层叠波浪', draw: waves },
    dots: { name: '点阵波纹', draw: dots },
    bokeh: { name: '光斑散景', draw: bokeh },
    rings: { name: '同心干涉', draw: rings },
    memphis: { name: '孟菲斯涂鸦', draw: memphis },
    blocks: { name: '色块拼贴', draw: blocks },
    aurora: { name: '极光渐变', draw: aurora },
    petals: { name: '花瓣线', draw: petals },
    stripes: { name: '斜条纹', draw: stripes },
  };
  const STYLE_KEYS = Object.keys(STYLES);

  /* ---------------- 组装整张 SVG ---------------- */
  function buildArt({ seed, style = 'random', size = [900, 900], grain = true, texture = 1 }) {
    const r = makeRand(seed);
    const [W, H] = size;
    const pal = makePalette(r);
    const cg = makeColorGradients(r, pal);
    let styleId = style && style !== 'random' && STYLES[style] ? style : r.pick(STYLE_KEYS);
    const res = STYLES[styleId].draw(r, { W, H, pal, cg: cg.fills }) || {};
    const tx = grain ? richness(r, W, H, pal, texture) : { defs: '', body: '' };

    const parts = [];
    parts.push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="随机生成的 SVG 图像">`);
    parts.push('<defs>');
    // 背景本身也是多段渐变 + 暗角，绝不平涂
    parts.push(`<linearGradient id="bg-grad" x1="0" y1="0" x2="1" y2="1">` +
      `<stop offset="0%" stop-color="${pal.bg}"/>` +
      `<stop offset="55%" stop-color="${pal.bg2}"/>` +
      `<stop offset="100%" stop-color="${pal.bg}"/></linearGradient>`);
    parts.push(`<radialGradient id="bg-vig" cx="0.5" cy="0.45" r="0.75">` +
      `<stop offset="55%" stop-color="#000" stop-opacity="0"/>` +
      `<stop offset="100%" stop-color="#000" stop-opacity="${pal.isDark ? 0.42 : 0.18}"/></radialGradient>`);
    parts.push(cg.defs);
    parts.push(res.defs || '');
    parts.push(tx.defs || '');
    parts.push('</defs>');
    parts.push(`<rect width="${W}" height="${H}" fill="url(#bg-grad)"/>`);
    parts.push(res.body || '');
    parts.push(`<rect width="${W}" height="${H}" fill="url(#bg-vig)"/>`);
    parts.push(tx.body || '');
    parts.push('</svg>');

    return { svg: parts.join('\n'), styleId, seed, W, H };
  }

  return { buildArt: buildArt, STYLES: STYLES, STYLE_KEYS: STYLE_KEYS, richness: richness, makePalette: makePalette, makeColorGradients: makeColorGradients, n1: n1, n2: n2, clamp: clamp, hsl: hsl };
})();
if (typeof module !== 'undefined' && module.exports) module.exports = GenCore;
