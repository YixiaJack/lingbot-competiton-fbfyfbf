#!/bin/bash
# level_shift.sh on [PX] | off | status
#   on PX : move the orange box of the organizer middle and top levels up by PX pixels (default 20, about 2.5 cm;
#           adjacent levels are ~80 px apart in the top camera). Bottom level and task 1 are unchanged.
#   off   : no shift.          Read by yolocue at each session start (needs a yolocue_bridge.py with shift_dest).
cd ~/yolocue || exit 1
case "$1" in
  on)  PX=${2:-20}; cp config.json config.json.bak_shift_$(date +%H%M%S)
       python3 -c "import json,sys; c=json.load(open('config.json')); c['dest_shift_px']={'organizer_middle': float(sys.argv[1]), 'organizer_top': float(sys.argv[1])}; json.dump(c, open('config.json','w'))" "$PX" ;;
  off) cp config.json config.json.bak_shift_$(date +%H%M%S)
       python3 -c "import json; c=json.load(open('config.json')); c.pop('dest_shift_px', None); json.dump(c, open('config.json','w'))" ;;
  status) ;;
  *) echo "usage: $0 on [PX] | off | status"; exit 2 ;;
esac
cat config.json; echo
grep -q "def shift_dest" yolocue_bridge.py && echo "bridge: has shift_dest (active after the service has loaded this file)" || echo "bridge: NO shift_dest yet"
