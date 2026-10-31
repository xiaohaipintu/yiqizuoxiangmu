#!/usr/bin/env bash
# 把本地代码部署到「野生青年」服务器（腾讯云上海 / Tailscale 节点 wildyouth-api）
#
# 服务器上 GitHub 直连被墙（GnuTLS recv error / Empty reply），
# 所以走「本地打包 → scp → systemd 重启」这条路。
#
# 用法（在自己电脑、仓库根目录执行）：
#   bash 拼图还原项目/web/deploy_to_server.sh                    # 部署当前工作区
#   REF=luna bash 拼图还原项目/web/deploy_to_server.sh            # 部署某个分支/提交（如队友的 luna）
#   REF=origin/luna bash 拼图还原项目/web/deploy_to_server.sh     # 同上，写全 ref 也行
#
#   PEM=/path/to/key.pem HOST=100.67.40.70 bash 拼图还原项目/web/deploy_to_server.sh
#
# 服务器布局：
#   /home/ubuntu/jigsaw/app      代码（web/ + src/ + requirements.txt + README.md）
#   /home/ubuntu/jigsaw/venv     Python 3.8 虚拟环境（numpy/cv2/shapely/scipy/qrcode）
#   /home/ubuntu/jigsaw/web.log  服务日志
#   systemd: jigsaw-web.service，端口 8099，仅 tailscale0 放行
#   入口：https://wildyouth.cn/jigsaw/ （nginx 反代）
set -euo pipefail

PEM="${PEM:-C:/Users/Administrator/Downloads/chongzuang.pem}"
HOST="${HOST:-100.67.40.70}"
USER_="${USER_:-ubuntu}"
PORT="${PORT:-8099}"
REF="${REF:-}"          # 空 = 用当前工作区；否则用 git archive <REF>

ROOT="$(cd "$(dirname "$0")/.." && pwd)"          # 拼图还原项目/
REPO="$(cd "$ROOT/.." && pwd)"                    # 仓库根
TMP="${TMPDIR:-/tmp}"
STAGE="$TMP/jigsaw-stage"
TARBALL="$TMP/jigsaw-app.tgz"

rm -rf "$STAGE"; mkdir -p "$STAGE"

if [ -n "$REF" ]; then
  echo "▸ 从 git ref 打包：$REF"
  git -C "$REPO" archive --format=tar -o "$TMP/jigsaw-ref.tar" "$REF" "拼图还原项目"
  tar -xf "$TMP/jigsaw-ref.tar" -C "$STAGE"
else
  echo "▸ 打包当前工作区 $ROOT"
  tar -cf - -C "$ROOT" web src requirements.txt README.md 2>/dev/null | tar -xf - -C "$STAGE"
fi

echo "▸ 生成上传包"
tar -czf "$TARBALL" -C "$STAGE" web src requirements.txt README.md 2>/dev/null || \
  tar -czf "$TARBALL" -C "$STAGE" "$(ls "$STAGE" | head -1)"

echo "▸ 上传到 $USER_@$HOST"
scp -i "$PEM" -o BatchMode=yes "$TARBALL" "$USER_@$HOST:/home/ubuntu/jigsaw/jigsaw-app.tgz"

echo "▸ 解包 + 修子路径 + 重启服务"
ssh -i "$PEM" -o BatchMode=yes "$USER_@$HOST" "
  set -e
  cd /home/ubuntu/jigsaw
  # 备份上一版
  rm -rf app.bak-last
  [ -d app ] && cp -a app app.bak-last || true
  rm -rf app/web app/src
  mkdir -p app
  tar -xzf jigsaw-app.tgz -C app
  rm -rf app/web/__pycache__ app/web/start.bat app/web/_*.py
  # 前端若用绝对路径 /api/…，挂到 /jigsaw/ 子路径下会打到主站后端 → 改成相对路径
  python3 - <<'PY'
from pathlib import Path
p = Path('app/web/index.html')
if p.exists():
    t = p.read_text(encoding='utf-8')
    n = t.count(\"'/api/\")
    if n:
        p.write_text(t.replace(\"'/api/\", \"'api/\"), encoding='utf-8')
        print(f'  index.html: {n} 处 /api/ → api/（适配 /jigsaw/ 子路径）')
    else:
        print('  index.html: 已是相对路径，无需改')
PY
  sudo systemctl restart jigsaw-web
  sleep 4
  systemctl is-active jigsaw-web
  ss -ltn | grep -q ':$PORT ' && echo '  端口 $PORT 已监听'
  curl -s -o /dev/null -w '  本机 HTTP %{http_code}\n' http://127.0.0.1:$PORT/
  tail -n 2 /home/ubuntu/jigsaw/web.log
"

echo "✓ 完成：https://wildyouth.cn/jigsaw/  ·  http://$HOST:$PORT/"
