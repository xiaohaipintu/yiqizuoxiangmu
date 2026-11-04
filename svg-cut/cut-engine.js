/* =========================================================================
   纯几何引擎：折线 → 栅格掩码 → 连通区域 → 轮廓环 → 简化多边形（局部坐标）
   纯 JS，无 DOM 依赖，可在 Node 里直接跑（便于回归测试）
   ========================================================================= */
var CutEngine = (function () {
  'use strict';

  function clamp(v, a, b) { return v < a ? a : v > b ? b : v; }

  /* 有向面积（screen 坐标系，y 向下） */
  function area(ring) {
    var s = 0, n = ring.length;
    for (var i = 0; i < n; i++) {
      var a = ring[i], b = ring[(i + 1) % n];
      s += a[0] * b[1] - b[0] * a[1];
    }
    return s / 2;
  }

  /* 闭合环的 Douglas–Peucker 简化 */
  function simplify(points, tol) {
    var n = points.length;
    if (n < 4) return points.slice();
    var pts = points.concat([points[0]]);
    var keep = new Uint8Array(pts.length);
    keep[0] = 1; keep[pts.length - 1] = 1;

    // 闭合环首尾重合，单锚点 DP 会退化成「只剩起点」→ 用离起点最远的点当第二锚点，分两段简化
    var far = 0, farD = -1;
    for (var fi = 1; fi < n; fi++) {
      var fx = points[fi][0] - points[0][0], fy = points[fi][1] - points[0][1];
      var fd = fx * fx + fy * fy;
      if (fd > farD) { farD = fd; far = fi; }
    }
    if (farD <= 0) return points.slice();
    keep[far] = 1;

    function dpRun(a, b) {
      var stack = [[a, b]];
      while (stack.length) {
        var seg = stack.pop(), i0 = seg[0], i1 = seg[1];
        if (i1 - i0 < 2) continue;
        var q1 = pts[i0], q2 = pts[i1];
        var dx = q2[0] - q1[0], dy = q2[1] - q1[1];
        var len = Math.sqrt(dx * dx + dy * dy);
        if (len < 1e-9) continue;
        var best = -1, bestD = -1;
        for (var i = i0 + 1; i < i1; i++) {
          var p = pts[i];
          var d = Math.abs(dy * p[0] - dx * p[1] + q2[0] * q1[1] - q2[1] * q1[0]) / len;
          if (d > bestD) { bestD = d; best = i; }
        }
        if (bestD > tol && best > 0) {
          keep[best] = 1;
          stack.push([i0, best], [best, i1]);
        }
      }
    }
    dpRun(0, far);
    dpRun(far, pts.length - 1);
    var out = [];
    for (var k = 0; k < pts.length - 1; k++) if (keep[k]) out.push(pts[k]);
    return out;
  }

  /* 把局部坐标折线画到 cols×rows 的掩码上（3×3 方笔刷，避免斜向渗漏） */
  function rasterize(polys, width, height, cols, rows) {
    var mask = new Uint8Array(cols * rows);
    var sx = cols / width, sy = rows / height;

    function put(x, y) {
      for (var dy = -1; dy <= 1; dy++) {
        var yy = y + dy;
        if (yy < 0 || yy >= rows) continue;
        var base = yy * cols;
        for (var dx = -1; dx <= 1; dx++) {
          var xx = x + dx;
          if (xx < 0 || xx >= cols) continue;
          mask[base + xx] = 1;
        }
      }
    }

    for (var pi = 0; pi < polys.length; pi++) {
      var poly = polys[pi];
      if (!poly || !poly.length) continue;
      if (poly.length === 1) {
        put(clamp(Math.round(poly[0][0] * sx), 0, cols - 1), clamp(Math.round(poly[0][1] * sy), 0, rows - 1));
        continue;
      }
      for (var i = 1; i < poly.length; i++) {
        var x0 = clamp(Math.round(poly[i - 1][0] * sx), 0, cols - 1);
        var y0 = clamp(Math.round(poly[i - 1][1] * sy), 0, rows - 1);
        var x1 = clamp(Math.round(poly[i][0] * sx), 0, cols - 1);
        var y1 = clamp(Math.round(poly[i][1] * sy), 0, rows - 1);
        var dxp = Math.abs(x1 - x0), dyp = Math.abs(y1 - y0);
        var stepX = x0 < x1 ? 1 : -1, stepY = y0 < y1 ? 1 : -1;
        var err = dxp - dyp;
        var x = x0, y = y0;
        var guard = (dxp + dyp) * 2 + 8;
        while (guard-- > 0) {
          put(x, y);
          if (x === x1 && y === y1) break;
          var e2 = 2 * err;
          if (e2 > -dyp) { err -= dyp; x += stepX; }
          if (e2 < dxp) { err += dxp; y += stepY; }
        }
      }
    }
    return mask;
  }

  /* 4 连通泛洪标记，返回 {lab, regions} */
  function labelRegions(mask, cols, rows) {
    var n = cols * rows;
    var lab = new Int32Array(n);
    lab.fill(-1);
    var stack = new Int32Array(n);
    var regions = [];
    var next = 0;
    for (var i = 0; i < n; i++) {
      if (mask[i] || lab[i] !== -1) continue;
      var id = next++;
      var sp = 0;
      stack[sp++] = i;
      lab[i] = id;
      var count = 0;
      var minX = cols, maxX = 0, minY = rows, maxY = 0;
      while (sp > 0) {
        var p = stack[--sp];
        var y = (p / cols) | 0, x = p - y * cols;
        count++;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        var q;
        if (x > 0) { q = p - 1; if (!mask[q] && lab[q] === -1) { lab[q] = id; stack[sp++] = q; } }
        if (x < cols - 1) { q = p + 1; if (!mask[q] && lab[q] === -1) { lab[q] = id; stack[sp++] = q; } }
        if (y > 0) { q = p - cols; if (!mask[q] && lab[q] === -1) { lab[q] = id; stack[sp++] = q; } }
        if (y < rows - 1) { q = p + cols; if (!mask[q] && lab[q] === -1) { lab[q] = id; stack[sp++] = q; } }
      }
      regions.push({ id: id, count: count, minX: minX, maxX: maxX, minY: minY, maxY: maxY });
    }
    return { lab: lab, regions: regions };
  }

  /* 某个 label 的所有边界环（栅格角点坐标 0..cols / 0..rows） */
  function traceRings(lab, cols, rows, id) {
    var W = cols + 1;
    var DIRX = [1, 0, -1, 0], DIRY = [0, 1, 0, -1];
    var edgeMap = new Map();

    function inside(x, y) {
      if (x < 0 || x >= cols || y < 0 || y >= rows) return false;
      return lab[y * cols + x] === id;
    }
    function add(x, y, dir) {
      var k = y * W + x;
      var arr = edgeMap.get(k);
      if (!arr) { arr = []; edgeMap.set(k, arr); }
      arr.push(dir);
    }

    for (var y = 0; y < rows; y++) {
      for (var x = 0; x < cols; x++) {
        if (!inside(x, y)) continue;
        if (!inside(x, y - 1)) add(x, y, 0);          // 上边 → 向右
        if (!inside(x + 1, y)) add(x + 1, y, 1);      // 右边 → 向下
        if (!inside(x, y + 1)) add(x + 1, y + 1, 2);  // 下边 → 向左
        if (!inside(x - 1, y)) add(x, y + 1, 3);      // 左边 → 向上
      }
    }

    var rings = [];
    var keys = Array.from(edgeMap.keys());
    for (var ki = 0; ki < keys.length; ki++) {
      var k0 = keys[ki];
      var alive = edgeMap.get(k0);
      while (alive && alive.length) {
        var sx = k0 % W, sy = (k0 - sx) / W;
        var cx = sx, cy = sy;
        var dir = alive.pop();
        var pts = [];
        var guard = cols * rows * 4 + 1000;
        while (guard-- > 0) {
          pts.push([cx, cy]);
          cx += DIRX[dir]; cy += DIRY[dir];
          if (cx === sx && cy === sy) break;
          var cand = edgeMap.get(cy * W + cx);
          if (!cand || !cand.length) break;
          // 优先右转（保持区域在同一侧），再直行、左转、掉头
          var prefer = [(dir + 1) & 3, dir, (dir + 3) & 3, (dir + 2) & 3];
          var pickDir = -1, pickAt = -1;
          for (var pi = 0; pi < prefer.length; pi++) {
            var at = cand.indexOf(prefer[pi]);
            if (at >= 0) { pickDir = prefer[pi]; pickAt = at; break; }
          }
          if (pickDir < 0) { pickDir = cand[cand.length - 1]; pickAt = cand.length - 1; }
          cand.splice(pickAt, 1);
          dir = pickDir;
        }
        if (pts.length >= 3) rings.push(pts);
        alive = edgeMap.get(k0);
      }
    }
    return rings;
  }

  /* 把切缝像素并入相邻区域（1 层 4 连通扩张）：相邻碎片的边界在中线对接，只剩 ~1 栅格像素的缝 */
  function growKerf(lab, mask, cols, rows) {
    var n = cols * rows;
    var pending = [];
    for (var p = 0; p < n; p++) {
      if (!mask[p]) continue;
      var y = (p / cols) | 0, x = p - y * cols;
      var owner = -1, conflict = false, q, L;
      if (x > 0) { q = p - 1; if (!mask[q] && (L = lab[q]) >= 0) { if (owner < 0) owner = L; else if (owner !== L) conflict = true; } }
      if (x < cols - 1) { q = p + 1; if (!mask[q] && (L = lab[q]) >= 0) { if (owner < 0) owner = L; else if (owner !== L) conflict = true; } }
      if (y > 0) { q = p - cols; if (!mask[q] && (L = lab[q]) >= 0) { if (owner < 0) owner = L; else if (owner !== L) conflict = true; } }
      if (y < rows - 1) { q = p + cols; if (!mask[q] && (L = lab[q]) >= 0) { if (owner < 0) owner = L; else if (owner !== L) conflict = true; } }
      if (!conflict && owner >= 0) pending.push(p, owner);
    }
    for (var i = 0; i < pending.length; i += 2) lab[pending[i]] = pending[i + 1];
    return lab;
  }

  /* 主入口：局部坐标 width×height，cuts = [[ [x,y], … ], …] */
  function computePieces(opts) {
    var width = opts.width, height = opts.height;
    var cuts = opts.cuts || [];
    var res = opts.res || 1400;
    var tol = opts.tol == null ? 0.7 : opts.tol;
    var minPixels = opts.minPixels == null ? 14 : opts.minPixels;

    var k = res / Math.max(width, height || 1);
    var cols = Math.max(8, Math.min(3600, Math.round(width * k)));
    var rows = Math.max(8, Math.min(3600, Math.round(height * k)));

    var mask = rasterize(cuts, width, height, cols, rows);
    var res2 = labelRegions(mask, cols, rows);
    var lab = res2.lab, regions = res2.regions;
    growKerf(lab, mask, cols, rows); // 切缝并入两侧区域，碎片之间只剩极细的缝
    var sx = width / cols, sy = height / rows;

    var pieces = [];
    for (var i = 0; i < regions.length; i++) {
      var rg = regions[i];
      if (rg.count < minPixels) continue;
      var rings = traceRings(lab, cols, rows, rg.id);
      if (!rings.length) continue;

      var conv = [];
      for (var r = 0; r < rings.length; r++) {
        var simple = simplify(rings[r], tol);
        simple = simplify(simple, tol * 2); // 第二遍抹掉切缝扩张产生的 1px 台阶/缺口
        if (simple.length < 3) continue;
        var ring = new Array(simple.length);
        for (var j = 0; j < simple.length; j++) ring[j] = [simple[j][0] * sx, simple[j][1] * sy];
        if (Math.abs(area(ring)) < 0.5) continue;
        conv.push(ring);
      }
      if (!conv.length) continue;

      // 最大环是外轮廓，其余是洞（环内挖空）
      conv.sort(function (a, b) { return Math.abs(area(b)) - Math.abs(area(a)); });
      var outer = conv[0];
      var minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      for (var m = 0; m < outer.length; m++) {
        var p = outer[m];
        if (p[0] < minX) minX = p[0];
        if (p[0] > maxX) maxX = p[0];
        if (p[1] < minY) minY = p[1];
        if (p[1] > maxY) maxY = p[1];
      }
      pieces.push({
        rings: conv,
        bbox: { minX: minX, minY: minY, maxX: maxX, maxY: maxY },
        area: Math.abs(area(outer)),
        pixels: rg.count,
        holes: conv.length - 1,
      });
    }

    // 阅读顺序：先按行分带，再按列
    pieces.sort(function (a, b) {
      var ah = a.bbox.maxY - a.bbox.minY, bh = b.bbox.maxY - b.bbox.minY;
      var bandTol = Math.min(ah, bh) * 0.5;
      if (Math.abs(a.bbox.minY - b.bbox.minY) > bandTol) return a.bbox.minY - b.bbox.minY;
      return a.bbox.minX - b.bbox.minX;
    });

    return pieces;
  }

  return {
    computePieces: computePieces,
    rasterize: rasterize,
    labelRegions: labelRegions,
    growKerf: growKerf,
    traceRings: traceRings,
    simplify: simplify,
    area: area,
  };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = CutEngine;
