#!/bin/bash
# cron wrapper for climb_lb_watch.py (12h cadence until the CLiMB deadline)
export HOME=/home/shunsuke
export PATH=/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin
cd "$(dirname "$0")" || exit 1
/usr/bin/python3 climb_lb_watch.py --quiet >> cron.out 2>&1
