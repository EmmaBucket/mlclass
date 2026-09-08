#!/bin/bash
# Double-click to start reading. (macOS runs .command files in Terminal.)
# First time only: right-click -> Open, because it is not signed.
cd "$(dirname "$0")"
PY=/opt/anaconda3/envs/mlclass/bin/python
if [ ! -x "$PY" ]; then
  PY=$(command -v python3)
  echo "mlclass environment not found at /opt/anaconda3/envs/mlclass -- using $PY"
fi
exec "$PY" recorder/record.py
