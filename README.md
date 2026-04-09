# worker-llamacpp

Run `llama.cpp` behind a small FastAPI worker that exposes OpenAI-compatible endpoints for RunPod serverless load balancing.

## Architecture

- `app.py` exposes `GET /ping` and proxies `/v1/*` requests.
- `llama_proxy.py` starts `llama-server`, waits for `/health`, and forwards requests over HTTP.
- `config.py` loads and validates environment variables before startup.

The FastAPI app and `llama-server` run in the same container. The Python app listens on `PORT`, while `llama-server` listens on `LLAMA_HOST:LLAMA_PORT` and is only reached internally.

## Run locally

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
LLAMA_MODEL=/path/to/model.gguf python app.py
```

## Test

```bash
pytest tests/
```

## Configuration

See `docs/env.md` for the full environment variable reference.

Important behavior:

- `LLAMA_MODEL` is required.
- `LLAMA_EXTRA_ARGS` supports shell-style quoting.
- If `LLAMA_HOST` uses a wildcard bind such as `0.0.0.0`, the proxy connects through `127.0.0.1` unless `LLAMA_CONNECT_HOST` is set.

## Operational notes

- Startup failures include recent `llama-server` output when available.
- `/ping` only reports healthy when the backend `llama-server` health endpoint is healthy.
- Streaming requests are detected from either `Accept: text/event-stream` or a JSON body with `"stream": true`.
