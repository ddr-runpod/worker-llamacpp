# worker-llamacpp

Run `llama.cpp` behind a small FastAPI worker that exposes OpenAI-compatible endpoints for RunPod serverless load balancing.

## Architecture

- `app.py` exposes `GET /ping` and proxies `/v1/*` requests.
- `llama_proxy.py` starts `llama-server`, waits for `/health`, and forwards requests over HTTP.
- `config.py` loads and validates environment variables before startup.

The FastAPI app and `llama-server` run in the same container. The Python app listens on `PORT`, while `llama-server` listens on `LLAMA_HOST:LLAMA_PORT` and is only reached internally.

## Quick Start (Local)

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Run with a HuggingFace model (uses -hf flag)
LLAMA_MODEL=unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q6_K_XL uv run python -u app.py
```

## Run Tests

```bash
uv run pytest tests/
```

## Docker Build

```bash
docker build -t worker-llamacpp .
```

## RunPod Serverless Deployment

### 1. Create Network Volume

Create a network volume in the RunPod console (100GB recommended) to store the HuggingFace model cache.

### 2. Create Serverless Endpoint

1. Build and push your Docker image:

```bash
docker build -t ddr-runpod/worker-llamacpp .
docker push ddr-runpod/worker-llamacpp
```

2. Create a new Serverless endpoint:
   - Container Image: `ddr-runpod/worker-llamacpp`
   - GPU: RTX 5090 (32GB) recommended for 26B models
   - Container Disk: 50GB minimum
   - Network Volume: Attach your network volume

3. Set Environment Variables:

   | Variable | Required | Example |
   |----------|----------|---------|
   | `LLAMA_MODEL` | Yes | `unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q6_K_XL` |
   | `HF_HOME` | No | `/runpod-volume/huggingface-cache` |
   | `HF_TOKEN` | No | Your HuggingFace token (for gated models) |

   See [docs/env.md](docs/env.md) for all optional environment variables.

### 3. First Request

On the first request, llama.cpp will automatically:
- Download the model from HuggingFace to the network volume
- Load the model into GPU memory

Subsequent requests will use the cached model.

## Configuration

See `docs/env.md` for the full environment variable reference.

Important behavior:

- `LLAMA_MODEL` is required and is always passed to llama-server via `-hf` flag
- The `-hf` flag automatically handles HuggingFace model download and mmproj selection
- `LLAMA_EXTRA_ARGS` supports shell-style quoting
- If `LLAMA_HOST` uses a wildcard bind such as `0.0.0.0`, the proxy connects through `127.0.0.1` unless `LLAMA_CONNECT_HOST` is set

## Operational notes

- Startup failures include recent `llama-server` output when available
- `/ping` only reports healthy when the backend `llama-server` health endpoint is healthy
- Streaming requests are detected from either `Accept: text/event-stream` or a JSON body with `"stream": true`