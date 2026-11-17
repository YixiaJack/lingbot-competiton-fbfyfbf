#!/usr/bin/env bash
# Rebuild the board's field page (https://lingbot-board.vercel.app/field) from local LeRobot datasets and publish it.
#
#   bash field_sync.sh [--title "现场数据 11/14 15:00"] DATASET[:source[:collector]] [DATASET...]
#
#   DATASET    a LeRobot v3 dataset directory (meta/ data/ videos/), e.g. a snapshot copied from the dev machine
#   source     demo (default) or dagger
#   collector  who collected it (free text), shown in the "谁采的" bars
#
# Runs field_build.py (detector on each episode's first head-camera frame) into a temp dir, then replaces the content
# of branch `field` of the private board-data repo ($FIELD_DATA_REMOTE) with it (one amended commit, force-pushed, so
# the branch does not grow). The page reloads its data every 2 minutes. Never touches branch main (the GPU board).
# Env: FIELD_WEIGHTS (detector .pt, default the pre-event yolo11s), FIELD_PY (python with ultralytics/av/pyarrow),
#      FIELD_REPO (local clone of branch field), FIELD_DATA_REMOTE (git URL of the board-data repo), FIELD_CONF (default 0.25), FIELD_CAMERA (default auto: high/top/head).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${FIELD_PY:-$HOME/.venvs/field/bin/python}
WEIGHTS=${FIELD_WEIGHTS:-$HOME/.cache/field_dev/weights/yolo11s_sim.pt}
REPO=${FIELD_REPO:-$HOME/.cache/field_dev/data}
TITLE="现场数据"
args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --title) TITLE="$2"; shift 2 ;;
    -h|--help) sed -n 2,17p "$0"; exit 0 ;;
    *) args+=(--dataset "$1"); shift ;;
  esac
done
[[ ${#args[@]} -gt 0 ]] || { echo "no dataset given (bash field_sync.sh --help)"; exit 2; }
[[ -f "$WEIGHTS" ]] || { echo "detector weights not found: $WEIGHTS"; exit 2; }
OUT=$(mktemp -d "${TMPDIR:-/tmp}/field_out.XXXXXX"); trap 'rm -rf "$OUT"' EXIT
extra=(); [[ -n "${FIELD_CAMERA:-}" ]] && extra+=(--camera "$FIELD_CAMERA")
"$PY" "$HERE/field_build.py" "${args[@]}" --weights "$WEIGHTS" --out "$OUT" --conf "${FIELD_CONF:-0.25}" \
  --title "$TITLE" --seen "${FIELD_SEEN:-$HOME/.cache/field_dev/seen.json}" ${extra[@]+"${extra[@]}"}   # (bash 3.2 on macOS: an empty array is "unbound" under set -u)
if [[ ! -d "$REPO/.git" ]]; then
  git clone -q --branch field --single-branch "${FIELD_DATA_REMOTE:?set FIELD_DATA_REMOTE to the board-data repo URL}" "$REPO"
fi
cd "$REPO"
git fetch -q origin field && git reset -q --hard origin/field
git rm -q -r --ignore-unmatch thumbs field_data.json 'bg_t*.jpg' >/dev/null
cp -R "$OUT"/. "$REPO"/
git add -A
git -c user.name="${FIELD_GIT_NAME:-field-bot}" -c user.email="${FIELD_GIT_EMAIL:-field-bot@users.noreply.github.com}" \
  commit -q --amend -m "field data: $TITLE" --date=now
git push -q --force origin HEAD:field
echo "$(date '+%F %T') published $(git rev-parse --short HEAD): $(ls thumbs | grep -vc _end) episodes -> https://lingbot-board.vercel.app/field"
