#!/bin/bash
# usage: devput.sh LOCAL REMOTE   -- copy a file to the dev machine through the tmux session (gzip+base64 heredoc)
set -e
L=$1; R=$2; S=$(dirname "$0")
md5l=$(md5 -q "$L")
tmp=$(mktemp)
{ echo "mkdir -p \"\$(dirname $R)\"; base64 -d <<'__B64__' | gunzip > $R"; gzip -c "$L" | base64 -b 76; echo "__B64__"; } > "$tmp"
tmux load-buffer -b devput "$tmp"; tmux paste-buffer -d -b devput -t dev; tmux send-keys -t dev Enter
rm -f "$tmp"
out=$("$S/devrun_tmux.sh" 60 "md5sum $R")
echo "$out" | grep -q "$md5l" && echo "OK $R ($md5l)" || { echo "MD5 MISMATCH: local $md5l remote: $out"; exit 1; }
