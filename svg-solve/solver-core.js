/* =========================================================================
   拼图求解引擎：碎片栅格化 → 接缝代价 → 强边建图 → 抗错边定位 → 细精度精修
   纯几何 + canvas 位图，不读任何「碎片位置」信息（不抄答案）。
   ========================================================================= */
var SolverCore = (function () {
  'use strict';
  /* ============ 求解引擎：栅格化 → 接缝代价 → 贪心并查集 → 松弛精修 ============ */
  function clamp(v, a, b) { return v < a ? a : v > b ? b : v; }
  function n2(v) { return Math.round(v * 100) / 100; }
  function medianOf(arr) {
    var a = arr.slice().sort(function (x, y) { return x - y; });
    var m = a.length >> 1;
    return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
  }

  /* 3×3 均值模糊（只在内点内取样，alpha 保持原值）：压掉高频纹理/网纹，让颜色连续性更可信 */
  function blurRGBA(px, mask, w, h) {
    var out = new Uint8ClampedArray(px.length);
    out.set(px);
    for (var y = 1; y < h - 1; y++) {
      for (var x = 1; x < w - 1; x++) {
        var i = y * w + x;
        if (!mask[i]) continue;
        var r = 0, g = 0, b = 0, c = 0;
        for (var dy = -1; dy <= 1; dy++) {
          var row = i + dy * w;
          for (var dx = -1; dx <= 1; dx++) {
            var j = row + dx;
            if (!mask[j]) continue;
            r += px[j * 4]; g += px[j * 4 + 1]; b += px[j * 4 + 2]; c++;
          }
        }
        if (c) { out[i * 4] = r / c; out[i * 4 + 1] = g / c; out[i * 4 + 2] = b / c; }
      }
    }
    return out;
  }

  /* 一块碎片渲染成栅格：RGBA + mask + 边界像素表 */
  function rasterizePiece(piece, k) {
    return new Promise(function (resolve) {
      var vb = piece.vb;
      var w = Math.max(2, Math.round(vb.w * k));
      var h = Math.max(2, Math.round(vb.h * k));
      var img = new Image();
      img.onload = function () {
        var cv = document.createElement('canvas');
        cv.width = w; cv.height = h;
        var g = cv.getContext('2d', { willReadFrequently: true });
        g.clearRect(0, 0, w, h);
        g.drawImage(img, 0, 0, w, h);
        var px = g.getImageData(0, 0, w, h).data;
        var n = w * h, mask = new Uint8Array(n), inside = 0, i, x, y;
        for (i = 0; i < n; i++) { if (px[i * 4 + 3] > 24) { mask[i] = 1; inside++; } }
        var edge = [];
        for (y = 0; y < h; y++) {
          for (x = 0; x < w; x++) {
            if (!mask[y * w + x]) continue;
            if (x === 0 || !mask[y * w + x - 1] || x === w - 1 || !mask[y * w + x + 1] ||
                y === 0 || !mask[(y - 1) * w + x] || y === h - 1 || !mask[(y + 1) * w + x]) {
              edge.push(x, y);
            }
          }
        }
        resolve({ id: piece.id, w: w, h: h, px: blurRGBA(px, mask, w, h), mask: mask, edge: edge, inside: inside, piece: piece });
      };
      img.onerror = function () { resolve(null); };
      img.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(pieceSvgText(piece));
    });
  }

  function cdiff(pa, ia, pb, ib) {
    var a = ia * 4, b = ib * 4;
    return (Math.abs(pa[a] - pb[b]) + Math.abs(pa[a + 1] - pb[b + 1]) + Math.abs(pa[a + 2] - pb[b + 2])) / 3;
  }
  /* B 的原点落在 A 原点 + (dx,dy) 时的接缝代价（越小越像原本相邻）。
     关键细节：颜色比较用两侧各「往内 1px」的实心像素，避开抗锯齿边缘；
     重叠像素重罚，接缝太短重罚（否则只贴半条边也能拿低分）。 */
  function seamCost(A, B, dx, dy, stride) {
    var e = B.edge, aw = A.w, ah = A.h, am = A.mask, ap = A.px;
    var bw = B.w, bh = B.h, bm = B.mask, bp = B.px;
    var n = 0, diff = 0, overlap = 0, gapSum = 0, ndiff = 0, st = (stride || 1) * 2;
    var UX = [1, -1, 0, 0], UY = [0, 0, 1, -1];
    for (var i = 0; i < e.length; i += st) {
      var bx = e[i], by = e[i + 1];
      var ax = bx + dx, ay = by + dy;
      if (ax >= 0 && ax < aw && ay >= 0 && ay < ah && am[ay * aw + ax]) overlap++;
      for (var k = 0; k < 4; k++) {
        var ux = UX[k], uy = UY[k], hit = 0;
        // 允许 1~2px 的缝（切缝本身就有 ~1px 宽），缝越宽罚一点
        for (var d = 1; d <= 2; d++) {
          var ax1 = ax + ux * d, ay1 = ay + uy * d;            // A 侧最近的实心像素
          if (ax1 < 0 || ax1 >= aw || ay1 < 0 || ay1 >= ah) continue;
          if (!am[ay1 * aw + ax1]) continue;
          var ax2 = ax1 + ux, ay2 = ay1 + uy;                  // 再往 A 里 1px
          if (ax2 < 0 || ax2 >= aw || ay2 < 0 || ay2 >= ah) break;
          var qa = ay2 * aw + ax2;
          if (!am[qa] || ap[qa * 4 + 3] < 200) break;
          var bx2 = bx - ux, by2 = by - uy;                    // B 侧往内 1px
          if (bx2 < 0 || bx2 >= bw || by2 < 0 || by2 >= bh) break;
          var qb = by2 * bw + bx2;
          if (!bm[qb] || bp[qb * 4 + 3] < 200) break;
          diff += cdiff(ap, qa, bp, qb);
          gapSum += (d - 1);
          n++;
          hit = 1;
          break;
        }
        if (hit) break;                                        // 同一像素只计一次
      }
    }
    if (n < 6) return null;
    // 明显重叠就不可能是相邻关系 → 直接否决（几何做硬约束，颜色只负责排序）
    if (overlap > Math.max(2, n * 0.15)) return null;
    return { score: diff / n + 3 * (gapSum / n) + 240 / n, n: n, overlap: overlap };
  }

  /* 形状投票候选：把 B 的边界点投到 A 的边界点上，票数高的偏移 = 形状互补的位置
     （斜线 / 曲线切割时 bbox 互相覆盖、边不贴合，只有这种候选才管用） */
  function voteCandidates(A, B) {
    var CELL = 4, grid = {}, i, k, gx, gy, key;
    for (i = 0; i < A.edge.length; i += 2) {
      key = ((A.edge[i] / CELL) | 0) + ',' + ((A.edge[i + 1] / CELL) | 0);
      if (!grid[key]) grid[key] = [];
      grid[key].push(A.edge[i], A.edge[i + 1]);
    }
    var votes = {};
    for (i = 0; i < B.edge.length; i += 24) {        // 抽样 B 的边界点
      var bx = B.edge[i], by = B.edge[i + 1];
      var cx = (bx / CELL) | 0, cy = (by / CELL) | 0;
      for (gx = -3; gx <= 3; gx++) {
        for (gy = -3; gy <= 3; gy++) {
          var cells = grid[(cx + gx) + ',' + (cy + gy)];
          if (!cells) continue;
          for (k = 0; k < cells.length; k += 2) {
            var vk = Math.round((cells[k] - bx) / CELL) + ',' + Math.round((cells[k + 1] - by) / CELL);
            votes[vk] = (votes[vk] || 0) + 1;
          }
        }
      }
    }
    var arr = [];
    for (var vkey in votes) if (Object.prototype.hasOwnProperty.call(votes, vkey)) arr.push([vkey, votes[vkey]]);
    arr.sort(function (p, q) { return q[1] - p[1]; });
    var out = [];
    for (i = 0; i < Math.min(8, arr.length); i++) {
      var pp = arr[i][0].split(',');
      out.push([Math.round(parseFloat(pp[0]) * CELL), Math.round(parseFloat(pp[1]) * CELL)]);
    }
    return out;
  }

  /* 候选相邻位置：四条边贴合位置（以 0 对齐起步，保证 dy=0/dx=0 一定在候选里）+ bbox 对齐族 */
  function candidateOffsets(A, B, step, tol) {
    var out = [], s, t, v;
    var tolList = [-tol, 0, tol];
    for (s = 0; s < tolList.length; s++) {
      t = tolList[s];
      for (v = 0; v > -B.h; v -= step) { out.push([A.w + t, v]); out.push([-B.w + t, v]); }
      for (v = step; v < A.h; v += step) { out.push([A.w + t, v]); out.push([-B.w + t, v]); }
      for (v = 0; v > -B.w; v -= step) { out.push([v, A.h + t]); out.push([v, -B.h + t]); }
      for (v = step; v < A.w; v += step) { out.push([v, A.h + t]); out.push([v, -B.h + t]); }
    }
    // 不规则切割（bbox 互相覆盖、边不贴合）时用角/中心对齐族补充
    var ex = A.w - B.w, ey = A.h - B.h;
    var ax = [0, ex, ex / 2, -B.w / 2, A.w / 2];
    var ay = [0, ey, ey / 2, -B.h / 2, A.h / 2];
    for (s = 0; s < ax.length; s++) {
      for (t = 0; t < ay.length; t++) {
        for (v = -step; v <= step; v += step) out.push([Math.round(ax[s]) + v, Math.round(ay[t])]);
      }
    }
    // 形状投票候选：斜线/曲线切割时唯一起作用的来源
    var vc = voteCandidates(A, B);
    for (s = 0; s < vc.length; s++) out.push(vc[s]);
    return out;
  }

  /* 得分 = 颜色差 + 重叠惩罚 + 「只贴了半条边」惩罚（按本次候选里最长接缝归一化） */
  function pairScore(c, nMaxFull, penalty) {
    var cov = nMaxFull > 0 ? Math.min(1, c.n / nMaxFull) : 1;
    return c.score + penalty * (1 - cov);
  }

  /* 两块之间的最优相对偏移（粗扫 → 按覆盖度归一化排序 → 局部细化） */
  function bestPairOffset(A, B, cfg) {
    var cands = candidateOffsets(A, B, cfg.step, cfg.tol);
    var list = [], i, c, nMax = 0, best = null, sc, ddx, ddy;
    for (i = 0; i < cands.length; i++) {
      c = seamCost(A, B, cands[i][0], cands[i][1], cfg.stride);
      if (!c) continue;
      list.push({ dx: cands[i][0], dy: cands[i][1], n: c.n, raw: c.score });
      if (c.n > nMax) nMax = c.n;
    }
    if (!list.length) return null;
    var nMaxFull = nMax * (cfg.stride || 1);
    for (i = 0; i < list.length; i++) {
      // 粗扫阶段用「本次候选里的最大覆盖度」归一化；精修阶段（stride=1）折算回满分辨率
      list[i].score = pairScore({ n: list[i].n, score: list[i].raw }, nMax, cfg.lenPenalty);
      if (!best || list[i].score < best.score) best = list[i];
    }
    var fine = best;
    for (ddx = -cfg.step; ddx <= cfg.step; ddx++) {
      for (ddy = -cfg.step; ddy <= cfg.step; ddy++) {
        c = seamCost(A, B, best.dx + ddx, best.dy + ddy, 1);
        if (!c) continue;
        sc = pairScore(c, nMaxFull, cfg.lenPenalty);
        if (sc < fine.score) fine = { dx: best.dx + ddx, dy: best.dy + ddy, score: sc, n: c.n };
      }
    }
    return fine;
  }

  /* 在主分量内定位：低分边 BFS 传播 + 「中位数 + 内点平均」迭代（抗单条错边） */
  function positionFrom(nbr, comp, main, n) {
    var pos = [], fixed = [], i, z, t2;
    for (i = 0; i < n; i++) { pos.push({ x: 0, y: 0 }); fixed.push(false); }
    var rootPiece = -1;
    for (i = 0; i < n; i++) if (comp[i] === main) { rootPiece = i; break; }
    if (rootPiece < 0) return pos;
    fixed[rootPiece] = true;
    var bfs = [rootPiece];
    while (bfs.length) {
      var c2 = bfs.shift();
      for (z = 0; z < nbr[c2].length; z++) {
        var e2 = nbr[c2][z];
        if (fixed[e2.to] || comp[e2.to] !== main) continue;
        fixed[e2.to] = true;
        pos[e2.to] = { x: pos[c2].x + e2.dx, y: pos[c2].y + e2.dy };
        bfs.push(e2.to);
      }
    }
    for (var pass = 0; pass < 8; pass++) {
      var next = [], moved = 0;
      for (i = 0; i < n; i++) next.push({ x: pos[i].x, y: pos[i].y });
      for (i = 0; i < n; i++) {
        if (comp[i] !== main || i === rootPiece) continue;
        var xs = [], ys = [], wts = [];
        for (t2 = 0; t2 < nbr[i].length; t2++) {
          var e3 = nbr[i][t2];
          if (comp[e3.to] !== main) continue;
          xs.push(pos[e3.to].x - e3.dx);   // e3.dx 是「对方相对我」的偏移 → 反解要减
          ys.push(pos[e3.to].y - e3.dy);
          wts.push(e3.w);
        }
        if (!xs.length) continue;
        var mx = medianOf(xs), my = medianOf(ys), sx = 0, sy = 0, sw = 0;
        for (t2 = 0; t2 < xs.length; t2++) {
          if (Math.abs(xs[t2] - mx) > 6 || Math.abs(ys[t2] - my) > 6) continue;   // 离群边丢掉
          sx += xs[t2] * wts[t2]; sy += ys[t2] * wts[t2]; sw += wts[t2];
        }
        var nx = sw > 0 ? sx / sw : mx, ny = sw > 0 ? sy / sw : my;
        if (Math.abs(nx - pos[i].x) > 0.01 || Math.abs(ny - pos[i].y) > 0.01) moved++;
        next[i] = { x: 0.6 * pos[i].x + 0.4 * nx, y: 0.6 * pos[i].y + 0.4 * ny };
      }
      pos = next;
      if (!moved) break;
    }
    return pos;
  }

  /* 纹理求解主流程：两两最优偏移 → 按分数贪心合并（并查集 + 一致性） */
  async function solveTexture(rasters, cfg, onProgress) {
    var n = rasters.length;
    if (n <= 1) return { pos: [{ x: 0, y: 0 }], used: [], merged: n, inMain: [true] };
    var pairs = [], i, j, r, done = 0, totalPairs = n * (n - 1) / 2;

    for (i = 0; i < n; i++) {
      for (j = i + 1; j < n; j++) {
        r = bestPairOffset(rasters[i], rasters[j], cfg);
        if (r) pairs.push({ i: i, j: j, dx: r.dx, dy: r.dy, score: r.score, n: r.n });
        done++;
      }
      if (onProgress) await onProgress(0.05 + 0.7 * (done / totalPairs));
    }
    pairs.sort(function (a, b) { return a.score - b.score; });
    // 自适应阈值：只用「明显比其他候选好」的相邻对（绝对阈值 + 相对最优分的倍数）
    var bestScore = pairs.length ? pairs[0].score : 0;
    var cutScore = Math.min(cfg.strongScore, Math.max(4, bestScore * 2.0));
    if (onProgress) await onProgress(0.8);

    // ——— 建图 → 定位 → 用残差踢掉「与其他边矛盾」的边 → 重建（最多 4 轮）———
    var strongList = pairs.filter(function (q) { return q.score <= cutScore; });
    var nbr = [], comp = [], comps = [], main = 0, pos = [], z, rnd;

    function buildGraph() {
      nbr = [];
      for (var q = 0; q < n; q++) nbr.push([]);
      for (var e = 0; e < strongList.length; e++) {
        var sp = strongList[e], key = sp.i + ':' + sp.j, w = Math.max(1, sp.n);
        nbr[sp.i].push({ to: sp.j, dx: sp.dx, dy: sp.dy, w: w, key: key });
        nbr[sp.j].push({ to: sp.i, dx: -sp.dx, dy: -sp.dy, w: w, key: key });
      }
      comp = []; comps = [];
      for (var a = 0; a < n; a++) comp.push(-1);
      for (var b = 0; b < n; b++) {
        if (comp[b] >= 0) continue;
        var cid = comps.length, queue = [b], list = [];
        comp[b] = cid;
        while (queue.length) {
          var cur = queue.shift();
          list.push(cur);
          for (var c = 0; c < nbr[cur].length; c++) {
            if (comp[nbr[cur][c].to] < 0) { comp[nbr[cur][c].to] = cid; queue.push(nbr[cur][c].to); }
          }
        }
        comps.push(list);
      }
      main = 0;
      for (var d = 1; d < comps.length; d++) if (comps[d].length > comps[main].length) main = d;
    }

    for (rnd = 0; rnd < 4; rnd++) {
      // 每块只保留「跟自己最靠谱那条边」接近的边（错边通常只能排到该块第三、四条）
      var perBest = [], pi;
      for (pi = 0; pi < n; pi++) perBest.push(Infinity);
      for (pi = 0; pi < strongList.length; pi++) {
        var qq = strongList[pi];
        if (qq.score < perBest[qq.i]) perBest[qq.i] = qq.score;
        if (qq.score < perBest[qq.j]) perBest[qq.j] = qq.score;
      }
      strongList = strongList.filter(function (q) {
        return q.score <= Math.max(4, Math.min(perBest[q.i], perBest[q.j]) * 1.8);
      });
      buildGraph();
      pos = positionFrom(nbr, comp, main, n);
      var badKeys = [];
      for (i = 0; i < n; i++) {
        for (z = 0; z < nbr[i].length; z++) {
          var ei = nbr[i][z];
          if (ei.to < i) continue;
          var rx = Math.abs((pos[ei.to].x - pos[i].x) - ei.dx);
          var ry = Math.abs((pos[ei.to].y - pos[i].y) - ei.dy);
          if (Math.max(rx, ry) > 6) badKeys.push(ei.key);   // 与其他边矛盾
        }
      }
      if (!badKeys.length) break;
      var before = strongList.length;
      strongList = strongList.filter(function (q) { return badKeys.indexOf(q.i + ':' + q.j) < 0; });
      if (strongList.length === before || !strongList.length) break;
    }

    var inMain = [];
    for (i = 0; i < n; i++) inMain.push(comp[i] === main);
    var used = strongList.slice(0, Math.max(0, comps[main].length - 1));
    return { pos: pos, used: used, strong: strongList, merged: comps[main].length, inMain: inMain, rasters: rasters, top: pairs.slice(0, 8) };
  }

  /* 细精度精修：从根块起按 BFS 顺序，每块只跟「已定好位」的邻居对齐（锚定式，不会整体漂） */
  function refinePlacements(fine, res, ratio) {
    var n = res.pos.length, i, t;
    var pos = [];
    for (i = 0; i < n; i++) pos.push({ x: res.pos[i].x * ratio, y: res.pos[i].y * ratio });
    // 只用「够强」的相邻对（低分）建图，避免一条错边把整片拖歪
    var links = (res.strong && res.strong.length) ? res.strong : res.used;
    var adj = [];
    for (i = 0; i < n; i++) adj.push([]);
    for (i = 0; i < links.length; i++) {
      adj[links[i].i].push(links[i].j);
      adj[links[i].j].push(links[i].i);
    }
    var root = 0;
    for (i = 1; i < n; i++) if (adj[i].length > adj[root].length) root = i;

    var seen = [], order = [], queue = [root];
    for (i = 0; i < n; i++) seen.push(false);
    seen[root] = true;
    while (queue.length) {
      var cur = queue.shift();
      order.push(cur);
      for (t = 0; t < adj[cur].length; t++) {
        if (!seen[adj[cur][t]]) { seen[adj[cur][t]] = true; queue.push(adj[cur][t]); }
      }
    }

    var W = Math.max(4, Math.round(ratio * 2) + 2);
    for (i = 1; i < order.length; i++) {
      var p = order[i], placed = [];
      for (t = 0; t < adj[p].length; t++) if (order.indexOf(adj[p][t]) < i) placed.push(adj[p][t]);
      if (!placed.length || !fine[p]) continue;
      var best = null, ddx, ddy;
      for (ddx = -W; ddx <= W; ddx++) {
        for (ddy = -W; ddy <= W; ddy++) {
          var px = pos[p].x + ddx, py = pos[p].y + ddy, sum = 0, cnt = 0, q, c;
          for (t = 0; t < placed.length; t++) {
            q = placed[t];
            if (!fine[q]) continue;
            c = seamCost(fine[q], fine[p], Math.round(px - pos[q].x), Math.round(py - pos[q].y), 1);
            if (c) { sum += c.score; cnt++; }
          }
          if (cnt && (!best || sum / cnt < best.cost)) best = { x: px, y: py, cost: sum / cnt };
        }
      }
      if (best) pos[p] = { x: best.x, y: best.y };
    }
    return { pos: pos };
  }

  /* 单块独立 SVG（给 <img> 栅格化用；只用这块自己的几何，不涉及位置信息） */
  function pieceSvgText(p) {
    var vb = p.vb;
    return '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"' +
      ' viewBox="' + n2(vb.minX) + ' ' + n2(vb.minY) + ' ' + n2(vb.w) + ' ' + n2(vb.h) + '"' +
      ' width="' + n2(vb.w) + '" height="' + n2(vb.h) + '"><g' + p.attrs + '>' + p.content + '</g></svg>';
  }
  return {
    rasterizePiece: rasterizePiece, seamCost: seamCost, candidateOffsets: candidateOffsets,
    pairScore: pairScore, bestPairOffset: bestPairOffset, positionFrom: positionFrom,
    solveTexture: solveTexture, refinePlacements: refinePlacements, medianOf: medianOf,
    pieceSvgText: pieceSvgText, n2: n2, clamp: clamp,
  };
})();
if (typeof module !== 'undefined' && module.exports) module.exports = SolverCore;
