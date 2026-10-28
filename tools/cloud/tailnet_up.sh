#!/usr/bin/env bash
# Joins the tailnet from a Claude Code cloud session (SessionStart hook in .claude/settings.json); no-op anywhere else.
# The cloud VM sends everything through an HTTP proxy, so tailscaled runs in userspace mode and reaches peers over DERP:
# `ssh <100.x | *.ts.net>` goes through `tailscale nc` (openssh-client is installed if missing), other tools through SOCKS5 localhost:1055.
# Needs TS_AUTHKEY (reusable + ephemeral auth key) in the environment variables and *.tailscale.com in its allowed domains.
set -euo pipefail
[ "${CLAUDE_CODE_REMOTE:-}" = true ] || exit 0
[ -n "${TS_AUTHKEY:-}" ] || { echo "tailnet_up: TS_AUTHKEY not set, not joining the tailnet" >&2; exit 0; }
V=1.104.4
if ! command -v tailscaled > /dev/null; then
  curl -fsSL "https://pkgs.tailscale.com/stable/tailscale_${V}_amd64.tgz" | tar xz -C /tmp
  install -m 755 /tmp/tailscale_${V}_amd64/tailscale /tmp/tailscale_${V}_amd64/tailscaled /usr/local/bin/
fi
if ! tailscale status > /dev/null 2>&1; then
  pgrep -x tailscaled > /dev/null || { nohup tailscaled --tun=userspace-networking --state=mem: --socks5-server=localhost:1055 > /tmp/tailscaled.log 2>&1 & }
  for _ in $(seq 40); do [ -S /var/run/tailscale/tailscaled.sock ] && break; sleep 0.25; done
  tailscale up --auth-key="$TS_AUTHKEY" --hostname=claude-cloud --timeout=60s
fi
command -v ssh > /dev/null || { apt-get update -qq && apt-get install -y -qq --no-install-recommends openssh-client; } > /dev/null 2>&1 \
  || echo "tailnet_up: could not install openssh-client, ssh unavailable" >&2
grep -q 'tailscale nc' ~/.ssh/config 2> /dev/null || { mkdir -p ~/.ssh; printf 'Host 100.* *.ts.net\n  ProxyCommand tailscale nc %%h %%p\n' >> ~/.ssh/config; }
echo "tailnet_up: joined as $(tailscale ip -4)"
