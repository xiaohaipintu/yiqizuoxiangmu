# 网页版（局域网 · 手机扫码即用）

拍一张碎片散落的照片 → 上传 → 服务器跑完整条流水线 → 返回拼好的图。

## 启动

双击 `start.bat`，或在 `拼图还原项目/` 目录下执行：

```bash
python web/server.py --port 8000
```

启动后终端会打印：

```
本机访问 : http://127.0.0.1:8000/
手机访问 : http://192.168.102.62:8000/     ← 手机连同一个 Wi-Fi，扫码或手输
```

终端里还会直接打出二维码（`█▀▀` 那种方块图），手机相机扫一扫就能打开。

## 页面怎么用

1. 点「选择照片 / 拍照」，选一张碎片照片（手机可直接拍）
2. 选「搜索时长」：碎片越多、越复杂就往上调（15s 快 / 90s 搜得全）
3. 有弯边、缺角的凹形碎片，把「有凹形碎片」保持为「是」
4. 等进度条走完 → 左边是你拍的原图，右边是拼好的图，可长按/点按钮保存

底部「查看候选排名图」里还有最佳四名的对比图和分割掩膜，方便判断哪一步出了问题。

## 碎片与边的命名（方便两人沟通）

页面上的图默认带名称，右下「图上显示名称」可切换显示 / 隐藏。

| 对象 | 名字 | 规则 |
| --- | --- | --- |
| 碎片 | `A`、`B`、`C`… | 按 `pieces.json` 里的顺序（26 块之后 AA、AB…） |
| 边 | `A1`、`A2`… | 从**最上面的顶点**开始（同高取最左），按顶点顺序环状编号；第 k 条边 = 第 k 个顶点 → 第 k+1 个顶点 |

**原图和拼好的图上标的是同一套名字**，因为拼合过程只对碎片做刚体变换，顶点顺序不变。所以可以直接说「B 的 B2 边接 C 的 C1 边」。

页面下方的「碎片 / 边的名称对照表」会列出每块碎片的顶点数、面积，以及每条边的名字和像素长度——边长相近的两条边才可能是对接边，对照表就是用来干这个的。


## 拍照要求（成功率的关键）

- 碎片放在**纯色背景**上（绿/白最佳），不要用花纹桌布
- 碎片之间**不要互相压住**，留一点缝
- 光线均匀、不反光，整张照片别太暗
- 碎片占画面尽量大，边缘留白即可

不满足这些条件时，第一步分割就会少块或多块，后面怎么搜索都拼不对。

## 参数（可调）

页面上的两个参数会作为请求头传给后端：

| 页面控件 | 请求头 | 说明 |
| --- | --- | --- |
| 搜索时长 | `X-Beam-Time: 15/30/45/90` | 传给 `beam_search.py --time`，秒 |
| 有凹形碎片 | `X-Concave: 1/0` | 是否给 `segment.py` 加 `--concave` |

## HTTP 接口（前端就是调这三个）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/api/solve` | body = 图片原始字节（不是 multipart），返回 `{"id": "..."}` |
| `GET` | `/api/job?id=...` | 任务状态：`status`(queued/running/done/error)、`step`、`progress`、`log`、`result` |
| `GET` | `/api/file?id=...&n=文件名` | 取结果文件，白名单见 `server.py` 的 `ALLOWED_FILES` |

`result` 结构：

```json
{
  "pieces": 4,
  "solutions": 300,
  "metrics": {"holes_pct": 0.0, "overlap_pct": 0.0},
  "steps": {"segment": 2.2, "rescale_pieces": 1.0, "beam_search": 31.2,
            "eval_union": 1.2, "total": 36.2},
  "labels": [
    {"name": "A", "id": 1, "vertices": 11, "area": 395997.5,
     "edges": [{"name": "A1", "length": 375.5}, {"name": "A2", "length": 150.1}]}
  ],
  "has_labels": true,
  "files": ["assembled.jpg", "assembled_labels.jpg", "upload_labels.jpg", "..."]
}
```

带名称的图：`upload_labels.jpg`（原图标注版）、`assembled_labels.jpg`（成品图标注版）。

## 日志

**每一次任务的每一步都有完整日志**，包括命令行、stdout、stderr、返回码、耗时。

| 位置 | 内容 | 保留 |
| --- | --- | --- |
| `%TEMP%\jigsaw-jobs\<job_id>\logs\run.log` | 该任务全部步骤流水（含开始/结束时间戳、最终 result JSON） | 随任务目录，24 小时后清理 |
| `%TEMP%\jigsaw-jobs\<job_id>\logs\<脚本名>.log` | 单步日志：`segment.log`、`beam_search.log`… | 同上 |
| `%TEMP%\jigsaw-jobs\<job_id>\logs\error.log` | 只有失败时才有，含完整 traceback | 同上 |
| `%TEMP%\jigsaw-jobs\logs\job-<job_id>\` | **上面 logs 的永久归档副本**，任务目录被清理后仍然能查 | 永久 |
| `%TEMP%\jigsaw-jobs\logs\server-YYYYMMDD.log` | 服务器运行日志：收到任务 / 完成 / 失败（含碎片数、耗时、空洞率） | 永久 |

- 页面上：结果卡片 → 展开「查看候选排名图 / 原始分割结果 / 运行日志」→ **打开完整运行日志**按钮
- 接口：`GET /api/file?id=<job>&n=run.log`

日志长这样：

```
######################################################################
# 任务开始 2026-10-31 17:41:33
# 输入 ...\jigsaw-jobs\484e518c94c9\upload.jpg
# 参数 beam_time=15.0s concave=True
# Python 3.13.15  cwd=...\拼图还原项目
######################################################################

===== 2026-10-31 17:41:36  segment.py  (rc=0, 2.26s) =====
$ python ...\src\segment.py --image ... --out ... --concave
--- stdout ---
图像尺寸: 1802x2400
共 4 块碎片
...
--- stderr ---
(空)
```

## 实现要点（踩过的坑）

1. **成品图不是 `41_union_rank.png`** —— 那张是四个候选并排的排名图。真正的拼好图由 `pipeline.render_assembled()` 现场渲染：读 `best_union.json` 的 `polys` 和 `pieces_scaled.json` 的多边形，用顶点对拟合相似变换，把每个碎片的**真实像素** warp 回位。
2. **不要用 `placed` 里的 `rot/tx/ty` 自己算仿射** —— 旋转符号约定和顶点顺序对不上会贴出一张全白图。改用 `cv2.estimateAffinePartial2D(源多边形, 世界多边形)`，并用顶点残差（<2% 碎片尺度）做质量校验。
3. **Windows 下 cv2 读不了中文路径** —— 所有任务目录都放在系统临时目录（纯 ASCII）：`%TEMP%\jigsaw-jobs\<job_id>\`。本仓库路径含中文，所以 `input/` 也不能直接喂给 `cv2.imread`。
4. **子进程必须强制 UTF-8** —— `src/*.py` 会打印 `px²`，Windows 控制台默认 GBK，管道模式下直接 `UnicodeEncodeError` 崩掉。`pipeline.child_env()` 里设置了 `PYTHONIOENCODING=utf-8`。
5. **流水线串行执行** —— `beam_search.py` 是 CPU 密集型，并发跑会互相拖慢；`server.py` 用队列 + 单 worker。

## 耗时

4 块碎片、`--time 30` 约 **36 秒**（分割 2s + 尺度 1s + 束搜索 31s + 评估 1s + 渲染 <1s）。
碎片越多、搜索时长越久，成倍增长。

## 依赖

```bash
pip install -r requirements.txt      # numpy / opencv-python-headless / shapely / scipy
pip install qrcode                   # 可选：终端二维码 + /api/qr.png
```

`web/` 本身只用标准库（`http.server`），不需要 Flask。
