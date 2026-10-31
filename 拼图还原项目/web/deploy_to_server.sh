#!/usr/bin/env bash
# 把本地代码部署到「野生青年」服务器（腾讯云上海 / Tailscale 节点 wildyouth-api）
#
# 服务器上 GitHub 直连被墙，所以走「本地打包 → scp → systemd 重启」这条路。
#
# 用法（在自己电脑上、仓库根目录执行）：
#   bash 拼图还原项目/web/deploy_to_server.sh
#   PEM=/path/to/key.pem HOST=100.67.40.70 bash 拼图还原项目/web/deploy_to_server.sh
#
# 服务器布局：
#   /home/ubuntu/jigsaw/app     代码（由本脚本覆盖，web/ + src/）
#   /home/ubuntu/jigsaw/venv    Python 3.8 虚拟环境（numpy/cv2/shapely/scipy/qrcode）
#   /home/ubuntu/jigsaw/web.log 服务日志
#   systemd: jigsaw-web.service，端口 8099，仅 tailscale0 放行
set -euo pipefail

PEM="${PEM:-C:/Users/Administrator/Downloads/chongzuang.pem}"
HOST="${HOST:-100.67.40.70}"
USER_="${USER_:-ubuntu}"
PORT="${PORT:-8099}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"          # 拼图还原项目/
TMP="${TMPDIR:-/tmp}"
TARBALL="$TMP/jigsaw-app.tgz"

echo "▸ 打包 $ROOT"
tar -czf "$TARBALL" -C "$ROOT" web src requirements.txt README.md

echo "▸ 上传到 $USER_@$HOST"
scp -i "$PEM" -o BatchMode=yes "$TARBALL" "$USER_@$HOST:/home/ubuntu/jigsaw/"

echo "▸ 解包 + 重启服务"
ssh -i "$PEM" -o BatchMode=yes "$USER_@$HOST" "
  set -e
  cd /home/ubuntu/jigsaw
  rm -rf app/web app/src
  tar -xzf jigsaw-app.tgz -C app
  rm -rf app/web/__pycache__ app/web/start.bat app/web/_*.py
  sudo systemctl restart jigsaw-web
  sleep 3
  systemctl is-active jigsaw-web
  ss -ltn | grep -q ':$PORT ' && echo '端口 $PORT 已监听'
  tail -n 3 /home/ubuntu/jigsaw/web.log
"

echo "✓ 完成：http://$HOST:$PORT/"
