## About

This project allows to run llama.cpp on a RunPod serverless load balancing worker.

## Architecture

The worker consists of two main components that communicate over HTTP on the same machine:

1. **FastAPI application (app.py)**: This is the entry point that RunPod communicates with. It listens on the PORT environment variable (typically 80). It exposes a health check endpoint at /ping and proxies all OpenAI-compatible API requests to the internal llama-server.

2. **llama-server**: This is the llama.cpp HTTP server that runs the actual model inference. The FastAPI app spawns it as a subprocess and it listens on `LLAMA_HOST:LLAMA_PORT`. By default this is `127.0.0.1:8080`, but wildcard binds such as `0.0.0.0` are supported for the server process. It provides endpoints like `/v1/chat/completions`, `/v1/completions`, and `/health`.

The request flow is:
- RunPod sends requests to the FastAPI app on PORT
- FastAPI forwards requests to llama-server on `LLAMA_CONNECT_HOST:LLAMA_PORT`
- llama-server processes the request using the configured model
- Response flows back through FastAPI to RunPod

### Key Design Decisions

1. **Spawn llama-server as subprocess**: The Python app manages the full lifecycle of llama-server including start, health check, and graceful shutdown via signal propagation.

2. **HTTP proxy via httpx**: Clean separation between FastAPI and llama-server. Uses async HTTP client to forward requests, enabling middleware support and easier testing.

3. **Streaming support**: Streaming responses (SSE) from llama-server are passed through to the client. Streaming is detected from either `Accept: text/event-stream` or a JSON request body with `"stream": true`. RunPod serverless supports streaming.

4. **Health check verifies llama-server**: The /ping endpoint checks if llama-server is responding at /health. Returns 503 if the model is not loaded or the server is unavailable. This prevents RunPod from routing traffic to unready workers.

5. **Dual port setup**: FastAPI listens on the RunPod-provided PORT environment variable. llama-server listens on `LLAMA_HOST:LLAMA_PORT`, while the proxy connects to `LLAMA_CONNECT_HOST:LLAMA_PORT`. This separation avoids connecting to an invalid wildcard address.

6. **Operational diagnostics**: The proxy captures `llama-server` output during startup, detects early process exit, and surfaces recent backend logs when startup fails.

## Configuration

All llama-server parameters are configurable via environment variables. See [docs/env.md](docs/env.md) for the full list.

Important runtime behavior:

- `LLAMA_MODEL` is required and validated at startup.
- `LLAMA_EXTRA_ARGS` is parsed with shell-style quoting, so values containing spaces should be quoted.
- If `LLAMA_HOST` is `0.0.0.0` or `::`, the proxy defaults `LLAMA_CONNECT_HOST` to `127.0.0.1` unless explicitly overridden.

### Common Parameters

| Env Var | Default | Description |
|---------|---------|-------------|
| `LLAMA_MODEL` | required | Path to model file |
| `LLAMA_CONTEXT_SIZE` | `4096` | Context window size |
| `LLAMA_N_GPU_LAYERS` | `99` | GPU layers to offload |
| `LLAMA_TEMPERATURE` | `0.8` | Sampling temperature |
| `LLAMA_TOP_P` | `0.95` | Top-p sampling |
| `LLAMA_TOP_K` | `40` | Top-k sampling |
| `LLAMA_MMPROJ` | - | Multimodal projector path |
| `LLAMA_HOST` | `127.0.0.1` | Host address llama-server binds to |
| `LLAMA_CONNECT_HOST` | derived from `LLAMA_HOST` | Host address the FastAPI app uses to reach llama-server |
| `LLAMA_PORT` | `8080` | Internal llama-server port |
| `LLAMA_N_PARALLEL` | `1` | Parallel request slots |
| `LLAMA_EXTRA_ARGS` | - | Additional llama-server args |

## Development

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run locally (requires model file)
LLAMA_MODEL=/path/to/model.gguf python app.py

# Run tests
.venv/bin/pytest tests/
```

## Testing Focus

The current tests cover:

- FastAPI health and proxy routing behavior
- Streaming detection and status propagation
- Config parsing and validation
- `llama-server` startup timeout and early-exit behavior

## Docs

* [llama.cpp Readme](https://raw.githubusercontent.com/ggml-org/llama.cpp/refs/heads/master/README.md)
* [llama-server Documentation](https://raw.githubusercontent.com/ggml-org/llama.cpp/refs/heads/master/tools/server/README.md)
* [Serverless Load Balancing Overview](https://docs.runpod.io/serverless/load-balancing/overview.md)
* [Serverless Load Balancing Build a Worker](https://docs.runpod.io/serverless/load-balancing/build-a-worker.md)
