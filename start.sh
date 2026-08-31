#!/bin/sh
set -eu

# FastAPI remains private to this container. Streamlit is the single public port.
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!

cleanup() {
    kill "$API_PID" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

# Give FastAPI enough time to initialize its model/data state on low-CPU instances.
python - <<'PY'
import time
import urllib.request

for _ in range(120):
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=1) as r:
            if r.status == 200:
                break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("Blue Team API failed to become healthy")
PY

PORT="${PORT:-10000}"
exec python -m streamlit run ui/app.py \
    --server.address 0.0.0.0 \
    --server.port "$PORT" \
    --server.headless true \
    --browser.gatherUsageStats false
