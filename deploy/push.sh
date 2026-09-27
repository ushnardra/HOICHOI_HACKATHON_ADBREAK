#!/usr/bin/env bash
# From the laptop: copy the app to the server and (re)build it.
# Usage: bash deploy/push.sh <server-ip> [user] [public-host, e.g. contextawareadbreaks.duckdns.org]
set -euo pipefail
IP="$1"; USER_NAME="${2:-azureuser}"; PUBLIC_HOST="${3:-}"
KEY="$HOME/.ssh/oracle_adbreak"
cd "$(dirname "$0")/.."
grep '^GROQ_API_KEY=' .env > .env.server          # only the Groq key goes to the server
tar czf /tmp/adbreak_app.tgz adbreak/*.py config/*.json player Dockerfile requirements.txt README.md .dockerignore \
    deploy/docker-compose.yml deploy/setup_server.sh .env.server
scp -i "$KEY" -o StrictHostKeyChecking=accept-new /tmp/adbreak_app.tgz "$USER_NAME@$IP:~/adbreak_app.tgz"
ssh -i "$KEY" "$USER_NAME@$IP" "mkdir -p ~/app && tar xzf ~/adbreak_app.tgz -C ~/app && cd ~/app && PUBLIC_HOST=$PUBLIC_HOST bash deploy/setup_server.sh"
rm -f .env.server /tmp/adbreak_app.tgz
