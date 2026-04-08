#!/bin/sh
set -eu

RUN_MODE_VALUE="${RUN_MODE:-serve}"
PORT_VALUE="${PORT:-7860}"

if [ "$RUN_MODE_VALUE" = "eval" ]; then
  exec python3 inference.py
fi

exec python3 -m uvicorn server.app:app --host 0.0.0.0 --port "$PORT_VALUE"
