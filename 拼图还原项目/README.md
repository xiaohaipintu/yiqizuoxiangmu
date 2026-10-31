# 拍照拼图还原 —— 项目代码说明

把一张"纯色背景上散落彩色纸片"的照片，自动分割出每一块碎片、匹配它们的边与角、并拼回完整图形。
本压缩包为 **Python 流水线代码 + 样例数据 + 各样例处理成果**（HTML 网页版另行交付）。

---

## 1. 目录结构

```
new-chat/
├── src/                    全部 Python 源码（21 个模块）
├── input/                  5 张样例照片（pieces.jpg ~ pieces5.jpg）
├── out/  ~  out5/          各样例的处理结果（png 成品图 + json + log）
├── requirements.txt        Python 依赖
└── .gitignore
```

## 2. 环境与依赖

- Python 3.13（3.10+ 均可），依赖仅 3 个：
  `pip install -r requirements.txt`
  （numpy、opencv-python、shapely）

## 3. 流水线使用（推荐顺序）

```bash
# 1) 分割碎片：绿/纯色背景上抠出每块纸片 → out5/pieces.json
python src/segment.py --image input/pieces5.jpg --out out5 --concave

# 2) 尺度补偿：吸收圆角/毛边导致的面积损失 → pieces_scaled.json
python src/rescale_pieces.py --pieces out5/pieces.json --out out5/pieces_scaled.json

# 3) 束搜索拼合：边几何拼接（允许 T 型接缝、shapely 拒重叠）→ beam_solutions.json
python src/beam_search.py --pieces out5/pieces_scaled.json --out out5

# 4) 真实并集评估：按"内部空洞 / 重叠"排序 → best_union.json + 41_union_rank.png
python src/eval_union.py --solutions out5/beam_solutions.json --out out5

# 5) 等长边配对（整条边共线贴合 + 闭环枚举）→ 31_closed.png 等
python src/equal_edges.py --pieces out5/pieces_scaled.json --out out5 --tol 0.06 --overview

# 6) 角度匹配（凹形夹角 vs 三角形化内角，容差 ±6°）→ 40_angle_match.png
python src/angle_match.py --pieces out5/pieces_scaled.json --out out5 --tol-deg 6
```

## 4. 模块说明

| 模块 | 作用 |
| --- | --- |
| `segment.py` | 抠碎片：Lab 色差取四边背景色（默认 ΔE=40），直线最小二乘拟合求交精修角点，`--concave` 沿真实轮廓拟合凹槽，面积偏差 ≤6% 校验 |
| `geometry.py` | 几何原语：面积 / 边长 / 内角 / 多边形交叠 / 凸包等 |
| `assemble.py` | 拼接内核：Piece / world / place_edge 等拼接原语 |
| `solve.py` | 最优优先（Best-first）拼装 |
| `enumerate.py` | 全拼法枚举 + 刚体签名 / 容差同构去重 |
| `beam_search.py` | 分层束搜索：逐层保留"凸包空隙最小"状态，支持 T 型部分贴合、shapely 精确拒穿透 |
| `closure_search.py` | 等长边闭环搜索 |
| `equal_edges.py` | 等长边整条共线贴合 + 三块/多块闭环枚举 |
| `angle_match.py` | 角度匹配：碎片"三角形化"（最大面积内接三角形）→ 凹形夹角 vs 三角形内角容差匹配 → 相似变换嵌入（无镜像）→ 穿透率过滤 → 渲染大图 |
| `ratio_match.py` | 边长比例指纹匹配（仅适用于纯三角形小碎片） |
| `pairwise.py` | 两两碎片边配对 |
| `match_census.py` | 边长度容差普查 |
| `rescale_pieces.py` | 尺度补偿：sqrt(像素面积/多边形面积) 相似缩放 |
| `eval_union.py` | 用 shapely unary_union 评估真实内部空洞与重叠（凹形拼合的正确度量） |
| `verify_closure.py` | 栅格化重叠 / 空洞验证 |
| `analyze_edges.py` / `inspect_vertices.py` / `diag_poly.py` | 边 / 顶点 / 多边形拟合诊断 |
| `render.py` / `render_all.py` / `render_attempt.py` | 拼合结果可视化 |

## 5. 各样例结果

| 样例 | 说明 | 结论 |
| --- | --- | --- |
| `pieces2.jpg`（out2） | 3 块（三角 + 六边形 + 三角） | **严丝合缝**：无重叠、无空洞（栅格化重叠 0.057%、空洞 0.005%），成品 `10_assembled.png`、候选 `11_candidates.png`、闭环 `31_closed.png`（空穴 0.35%） |
| `pieces3.jpg`（out3） | 7 块紫红碎片 | 历史遗留难点：等长边闭环只找到空穴 12.46% 的拼法（`40_attempts.png`），**尚未做到严丝合缝** |
| `pieces4.jpg`（out4） | 4 块（1 凹形 + 3 三角） | beam_search + eval_union：**空洞 0% / 重叠 0%**（`best_union.json`，顺序 [1,3,2,4]），`41_union_rank.png` |
| `pieces5.jpg`（out5） | 4 块（几何与 pieces4 相同） | 同上：**0 空洞 / 0 重叠**；另产出 `40_angle_match.png` 角度匹配大图（3 个角位命中） |

## 6. 关键方法与经验

- **角点精修**：碎片多边形化后做直线最小二乘拟合求交，把圆角/毛边"分裂"出的伪顶点还原成真实角点。
- **尺度补偿**：薄小碎片拟合面积系统性偏小 5~20%，用面积比开方做相似缩放吸收损失。
- **束搜索**：不镜像、允许 T 型部分贴合（共线 + 端点锚定 + 公共长度≥容差×短边），shapely 精确算重叠拒绝穿透，以凸包空隙最小保留 beam。
- **三角形化**：小碎片拟合出的 4~5 顶点多是弧边分裂的伪角，取"最大面积内接三角形"还原真实 3 内角，再做角度匹配。
- **角度匹配**（angle_match）：凹形 11 个夹角 vs 各三角形内角，|Δ|≤6° 视为吻合；命中即 3 锚点相似拟合（det R=+1 无镜像）嵌入凹形，穿透率 ≤35% 才保留。

## 7. 已知限制

- `ratio_match.py` 仅对纯三角形小碎片有效（对 4~5 边形碎片 0 命中）。
- `equal_edges.py` 的闭环枚举在 4 块及以上样例上可能耗时过长（曾后台运行被终止）。
- out3 七块碎片尚无严丝合缝解（最优等长边闭环空穴 12.46%）。
- 依赖拍摄条件：纸片与背景色差明显、边缘无阴影遮挡。

---
*HTML 网页版（拍照即拼）为独立交付物，不在本压缩包内。*
