# Environment Variables

## Required

| Variable | Description |
|----------|-------------|
| `LLAMA_MODEL` | HuggingFace model ID (e.g., `unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q6_K_XL`) |

## HuggingFace Configuration

| Variable | Description |
|----------|-------------|
| `HF_HOME` | HuggingFace cache directory. Recommended: `/runpod-volume/huggingface-cache` for network volume persistence |
| `HF_TOKEN` | HuggingFace access token (required for gated models) |

## Sampling (Optional - llama.cpp defaults used if not set)

| Variable | Description |
|----------|-------------|
| `LLAMA_TEMPERATURE` | Sampling temperature |
| `LLAMA_TOP_P` | Top-p (nucleus) sampling |
| `LLAMA_TOP_K` | Top-k sampling |

## Context & Performance (Optional)

| Variable | Description |
|----------|-------------|
| `LLAMA_CONTEXT_SIZE` | Context window size |
| `LLAMA_N_GPU_LAYERS` | Layers to offload to GPU |
| `LLAMA_N_PARALLEL` | Parallel request slots |
| `LLAMA_THREADS` | CPU threads |

## Chat Template (Optional)

| Variable | Description |
|----------|-------------|
| `LLAMA_CHAT_TEMPLATE_KWARGS` | JSON string for chat template, e.g., `'{"enable_thinking":true}'` |

## Advanced

| Variable | Description |
|----------|-------------|
| `LLAMA_EXTRA_ARGS` | Additional arguments passed directly to llama-server |

## Worker Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `80` | Port for the FastAPI app (set automatically by RunPod) |
| `LLAMA_HOST` | `127.0.0.1` | Host address that `llama-server` binds to |
| `LLAMA_CONNECT_HOST` | derived | Host address the FastAPI proxy uses to reach `llama-server` |
| `LLAMA_PORT` | `8080` | Internal port for llama-server |

## Notes

- `LLAMA_MODEL` is validated at always passed via `-hf` flag to llama-server, enabling automatic HuggingFace model download and mmproj selection.
- If `LLAMA_HOST` is `0.0.0.0` or `::`, the proxy automatically connects to `127.0.0.1` unless `LLAMA_CONNECT_HOST` is set.
- `LLAMA_EXTRA_ARGS` is parsed with shell-style quoting, so paths with spaces should be quoted.
- If an env var is not set, the parameter is not passed to llama-server, which uses its own defaults.

## Examples

### RunPod Serverless (Recommended)
```
LLAMA_MODEL=unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q6_K_XL
HF_HOME=/runpod-volume/huggingface-cache
HF_TOKEN=<your-huggingface-token>
LLAMA_CHAT_TEMPLATE_KWARGS={"enable_thinking":true}
```

### Local development
```
LLAMA_MODEL=unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q6_K_XL
HF_HOME=/tmp/huggingface-cache
```

### Custom sampling parameters
```
LLAMA_MODEL=unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q6_K_XL
LLAMA_TEMPERATURE=1.0
LLAMA_TOP_P=0.95
LLAMA_TOP_K=64
```

### Using extra args
```
LLAMA_EXTRA_ARGS=--flash-attn on --rope-scaling yarn --embedding
```

### Using extra args with spaces
```
LLAMA_EXTRA_ARGS=--log-file "/tmp/llama server.log"
```