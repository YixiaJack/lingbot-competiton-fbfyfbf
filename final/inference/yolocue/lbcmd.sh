#!/bin/bash
# lbcmd.sh METHOD [PARAMS_JSON]  -- one JsonRpc call to the lingbot-infer command service (5080), prints result_json
set -e
source /home/guest/workspace/lingbot/inference/env.sh >/dev/null 2>&1
M=$1; P=${2:-'{}'}
P_ESC=$(printf '%s' "$P" | sed "s/'/''/g")
ros2 service call /lingbot/command arm_bridge_msgs/srv/JsonRpc "{api_version: '1', request_id: '$(cat /proc/sys/kernel/random/uuid)', method: $M, params_json: '$P_ESC'}" 2>&1 | tail -3
