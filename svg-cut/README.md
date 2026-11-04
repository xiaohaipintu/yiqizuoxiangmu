# SVG 线切割 · Line Cutter

`index.html` 单文件网页，双击即开（`file://` 可用，无依赖、不联网）。
把**任意 SVG** 按横线 / 竖线 / 自由折线切成多块，每块导出为独立、可单独打开的 SVG。

```bash
start svg-cut\index.html      # Windows
```

## 用法

1. **载入源图**：拖 SVG 文件到画布 / 点「选择 SVG 文件」/ 粘贴 `<svg>…</svg>` 代码。首屏自带一张示例图。
2. **画切割线**（快捷键 `V` / `H` / `F`）：
   - 竖线、横线：在画布上点一下加一条，拖线中间的圆点移位置。
   - 折线：连续点击落点，双击或回车结束，`Esc` 取消，`Backspace` 撤销上一刀。
   - 或填「列 × 行」自动生成网格线（可加抖动 %，切出不规则格）。
3. **开始切割** → 下方预览每块，悬停出现 `⤓` 单块下载。
4. **导出**：`全部下载 ZIP`（每块一个 `piece-NN.svg` + `knife-lines.svg` 刀模线）、`刀模线 SVG`（只含切割轨迹的细线，可直接给激光 / CNC / 绘图仪）。

精度档：快 900px / 标准 1400px / 精细 2000px —— 是内部栅格分辨率，决定了切边贴合精度（越大越准、越慢）。

## 实现

单条流水线，横竖线、折线、曲线（折线逼近）都走同一条路径：

```
折线 → 栅格掩码(3×3 笔刷, 防斜向渗漏) → 4 连通泛洪分区
     → 切缝并入两侧区域(kerf growth) → 边界有向边链成闭环
     → Douglas–Peucker 简化(两遍) → 用户坐标多边形
```

每块导出为 `<clipPath>`（支持多环 = 带洞）+ `<g clip-path>` 包住原图内容，viewBox 正好是这块外接框：

```svg
<svg viewBox="…” width="…" height="…">
  <defs><clipPath id="clip-3" clipPathUnits="userSpaceOnUse"><path d="M … Z" clip-rule="evenodd"/></clipPath></defs>
  <g clip-path="url(#clip-3)" fill="#…">…源图全部内容…</g>
</svg>
```

- 每块自包含（源图 `<defs>`/滤镜/渐变随块带走），不依赖外部文件。
- 相邻碎片的边界落在切缝中线，缝隙 ≈ 1 个栅格像素，拼回去基本无缝。

## 自测

```bash
# 1) 几何引擎回归（Node，19 项：不切割/单刀/网格/45°斜切/带洞/多刀/性能）
node svg-cut/test-engine.js

# 2) 整页端到端（无头 Edge，14 项：载入/块数/覆盖率/碎片 SVG 良构/ZIP/刀模线）
msedge --headless=new --disable-gpu --dump-dom \
  "file:///I:/duhisjdkc/Reasonix/拼图项目/svg-cut/index.html?selftest=1"
# 看输出里的 <div id="report"> 和 <title>SELFTEST OK</title>
```

当前：引擎 19/19 通过；页面自检 14/14 通过（本机无头 Edge 实测）。

## 已知边界

- 只切割，不生成拼图凸耳（燕尾榫/圆头那种）——那种用 `拼图还原项目/svg_tool/puzzle_svg.html`。
- 切缝有物理损耗：每刀约 1~2px（随精度档变化），与真实刀具有 kerf 同理。
- 未闭合的切割线不会把画面分开（只有贯穿的部分才会分成两块）。
- 源图里如果有 `<style>` + `#id` 选择器，切块后 id 会重复——单块文件独立仍正常，同页内联多块时建议按需改名。
