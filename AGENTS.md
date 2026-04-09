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

1. **Always use -hf flag**: The `LLAMA_MODEL` is always passed to llama-server via the `-hf` flag, which automatically handles model download and mmproj selection from HuggingFace.

2. **No Python defaults**: Config parameters only use values explicitly set via environment variables. If an env var is not set, the parameter is not passed to llama-server, which uses its own defaults.

3. **HF_HOME for caching**: HuggingFace cache directory can be set via `HF_HOME` to persist models on a network volume across worker restarts.

4. **Spawn llama-server as subprocess**: The Python app manages the full lifecycle of llama-server including start, health check, and graceful shutdown via signal propagation.

5. **HTTP proxy via httpx**: Clean separation between FastAPI and llama-server. Uses async HTTP client to forward requests, enabling middleware support and easier testing.

6. **Streaming support**: Streaming responses (SSE) from llama-server are passed through to the client. Streaming is detected from either `Accept: text/event-stream` or a JSON request body with `"stream": true`. RunPod serverless supports streaming.

7. **Health check verifies llama-server**: The /ping endpoint checks if llama-server is responding at /health. Returns 503 if the model is not loaded or the server is unavailable. This prevents RunPod from routing traffic to unready workers.

8. **Dual port setup**: FastAPI listens on the RunPod-provided PORT environment variable. llama-server listens on `LLAMA_HOST:LLAMA_PORT`, while the proxy connects to `LLAMA_CONNECT_HOST:LLAMA_PORT`. This separation avoids connecting to an invalid wildcard address.

9. **Operational diagnostics**: The proxy captures `llama-server` output during startup, detects early process exit, and surfaces recent backend logs when startup fails.

## Configuration

All llama-server parameters are configurable via environment variables. Only `LLAMA_MODEL` is required.

### Required Variables

| Env Var | Description |
|---------|-------------|
| `LLAMA_MODEL` | HuggingFace model ID (e.g., `unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q6_K_XL`) |

### Optional Variables

| Env Var | Description |
|---------|-------------|
| `HF_HOME` | HuggingFace cache directory (default: not set, uses llama.cpp default) |
| `HF_TOKEN` | HuggingFace access token (for gated models) |
| `PORT` | FastAPI listening port (default: 80, set by RunPod) |
| `LLAMA_TEMPERATURE` | Sampling temperature |
| `LLAMA_TOP_P` | Top-p sampling |
| `LLAMA_TOP_K` | Top-k sampling |
| `LLAMA_CONTEXT_SIZE` | Context window size |
| `LLAMA_N_GPU_LAYERS` | GPU layers to offload |
| `LLAMA_HOST` | Host address llama-server binds to (default: 127.0.0.1) |
| `LLAMA_CONNECT_HOST` | Host address the FastAPI app uses to reach llama-server |
| `LLAMA_PORT` | Internal llama-server port (default: 8080) |
| `LLAMA_N_PARALLEL` | Parallel request slots |
| `LLAMA_CHAT_TEMPLATE_KWARGS` | e.g., `'{"enable_thinking":true}'` |
| `LLAMA_EXTRA_ARGS` | Additional llama-server args |

Important runtime behavior:

- `LLAMA_MODEL` is required and validated at startup.
- The model is always passed via `-hf` flag to enable automatic HuggingFace download.
- `LLAMA_EXTRA_ARGS` is parsed with shell-style quoting, so values containing spaces should be quoted.
- If `LLAMA_HOST` is `0.0.0.0` or `::`, the proxy defaults `LLAMA_CONNECT_HOST` to `127.0.0.1` unless explicitly overridden.

## RunPod Serverless Deployment

### 1. Create Network Volume

Create a network volume in the RunPod console (100GB recommended) to store the HuggingFace model cache. This allows the model to persist across worker restarts.

### 2. Build and Push Docker Image

```bash
docker build -t your-dockerhub/worker-llamacpp .
docker push your-dockerhub/worker-llamacpp
```

### 3. Create Serverless Endpoint

1. Create a new Serverless endpoint in the RunPod console:
   - Container Image: `your-dockerhub/worker-llamacpp`
   - GPU: Select appropriate GPU (A100 80GB recommended for 26B models)
   - Container Disk: 50GB minimum
   - Network Volume: Attach your network volume

2. Set Environment Variables:
   - `LLAMA_MODEL`: e.g., `unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q6_K_XL`
   - `HF_HOME`: `/runpod-volume/huggingface-cache` (point to network volume)
   - `HF_TOKEN`: Your HuggingFace token (required for gated models)

### 4. First Request

On the first request, llama.cpp will automatically:
- Download the model from HuggingFace to the network volume
- Load the model into GPU memory

Subsequent requests will use the cached model from the network volume.

## Development

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Run locally with HuggingFace model
LLAMA_MODEL=unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q6_K_XL uv run python -u app.py

# Run tests
uv run pytest tests/
```

## Testing Focus

The current tests cover:

- FastAPI health and proxy routing behavior
- Streaming detection and status propagation
- Config parsing and validation
- `llama-server` startup timeout and early-exit behavior

## Docs

* [llama.cpp Readme](https://raw.githubusercontent.com/ggml-org/llama.cpp/refs/heads/master/README.md)
* [Unsloth Gemma 4 Guide](https://unsloth.ai/docs/models/gemma-4.md)
* [llama.cpp Multimodal Documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md)
* [llama-server Documentation](https://raw.githubusercontent.com/ggml-org/llama.cpp/refs/heads/master/tools/server/README.md)
* [Serverless Load Balancing Overview](https://docs.runpod.io/serverless/load-balancing/overview.md)
* [Serverless Load Balancing Build a Worker](https://docs.runpod.io/serverless/load-balancing/build-a-worker.md)
* [RunPod Network Volumes](https://docs.runpod.io/serverless/storage/network-volumes)
* [RunPod Model Caching](https://docs.runpod.io/serverless/endpoints/model-caching)