#!/usr/bin/env bash
# One-time server setup on Ubuntu 22.04 (Azure / Oracle / any VM). Run from the app folder: bash deploy/setup_server.sh
set -euo pipefail
IP=$(curl -s https://api.ipify.org)
HOST="${IP//./-}.sslip.io"
# optional nice name (Azure DNS label or DuckDNS) pointing at this IP: PUBLIC_HOST=contextawareadbreaks.duckdns.org
SITES="$HOST${PUBLIC_HOST:+, $PUBLIC_HOST}"
echo ">> public address: https://${PUBLIC_HOST:-$HOST}"
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER"
fi
# open the web ports in the VM's own firewall (Oracle images block them by default; harmless on Azure)
sudo iptables -I INPUT -p tcp --dport 80 -j ACCEPT || true
sudo iptables -I INPUT -p tcp --dport 443 -j ACCEPT || true
printf '%s {\n  request_body {\n    max_size 600MB\n  }\n  reverse_proxy app:7860 {\n    transport http {\n      read_timeout 30m\n      write_timeout 30m\n    }\n  }\n}\n' "$SITES" > deploy/Caddyfile
export CPUS=$(nproc)
sudo CPUS=$CPUS docker compose -f deploy/docker-compose.yml up -d --build
echo ">> done: https://${PUBLIC_HOST:-$HOST}"
