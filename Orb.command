#!/bin/sh
# Double-click to start orb mode: the orb page opens in your browser, then click Start.
# Close this window (or press Ctrl+C) to shut everything down.
cd "$(dirname "$0")" || exit 1
exec python3 orb_body.py "$@"
