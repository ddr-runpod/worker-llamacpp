# Environment Variables

## Required (choose one)

| Variable | Description |
|----------|-------------|
| `LLAMA_HF_MODEL` | HuggingFace model ID for auto-download (e.g., `philipsorst/gemma-4-26B-A4B-it-UD-Q6_K_XL`). Passed via `-hf` flag. |
| `LLAMA_MODEL` | Local path to a GGUF model file (e.g., `/runpod-volume/.../model.gguf`). Passed via `--model` flag. |
| `LLAMA_MODEL_RUNPOD_CACHE` | RunPod cached model path in `org/name/filename` format. Auto-resolves to the correct snapshot path. Passed via `--model` flag. |

Exactly one of `LLAMA_HF_MODEL`, `LLAMA_MODEL`, or `LLAMA_MODEL_RUNPOD_CACHE` must be set.

## HuggingFace Configuration

| Variable | Description |
|----------|-------------|
| `HF_HOME` | HuggingFace cache directory. Defaults to `/runpod-volume/huggingface-cache` for RunPod network volume persistence |
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

## Reasoning (Optional)

| Variable | Description |
|----------|-------------|
| `LLAMA_REASONING` | Enable/disable reasoning mode. Accepted values: `on`, `1`, `yes`, `off`, `0`, `no` (case-insensitive) |

## Chat Template (Optional)

| Variable | Description |
|----------|-------------|
| `LLAMA_CHAT_TEMPLATE_KWARGS` | JSON string for chat template, e.g., `'{"enable_thinking":true}'` |

## Multimodal (Optional)

| Variable | Description |
|----------|-------------|
| `LLAMA_MMPROJ` | Path to a multimodal projection (mmproj) GGUF file. Used for vision/language models. Passed via `--mmproj` flag. |
| `LLAMA_MMPROJ_RUNPOD_CACHE` | RunPod cached mmproj path in `org/name/filename` format. Auto-resolves to the correct snapshot path. Passed via `--mmproj` flag. |

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

## Logging

The worker outputs structured JSON logs to stderr, which RunPod captures automatically.

| Variable | Default | Description |
|----------|---------|-------------|
| `RUNPOD_SERVICE_NAME` | `worker-llamacpp` | Service identifier in logs |
| `RUNPOD_SERVICE_VERSION` | `unknown` | Version for logs (e.g., git tag) |
| `LOG_LEVEL` | `INFO` | Log level: `DEBUG`, `INFO`, `WARN`, `ERROR` |
| `ENV` | `unknown` | Environment name (e.g., `prod`, `dev`) |

## Notes

- Exactly one of `LLAMA_HF_MODEL`, `LLAMA_MODEL`, or `LLAMA_MODEL_RUNPOD_CACHE` is required. They are validated as XOR — attempting to set more than one or none will cause a startup error.
- `LLAMA_HF_MODEL` is passed via `-hf` flag, enabling automatic HuggingFace model download and mmproj selection.
- `LLAMA_MODEL` and `LLAMA_MODEL_RUNPOD_CACHE` are passed via `--model` flag for local GGUF file paths.
- `LLAMA_MMPROJ` and `LLAMA_MMPROJ_RUNPOD_CACHE` are passed via `--mmproj` flag for multimodal projection files.
- `HF_HOME` defaults to `/runpod-volume/huggingface-cache` when not set.
- If `LLAMA_HOST` is `0.0.0.0` or `::`, the proxy automatically connects to `127.0.0.1` unless `LLAMA_CONNECT_HOST` is set.
- `LLAMA_EXTRA_ARGS` is parsed with shell-style quoting, so paths with spaces should be quoted.
- If an env var is not set, the parameter is not passed to llama-server, which uses its own defaults.

## Examples

### RunPod Serverless with Model Caching (Recommended)
```
LLAMA_MODEL_RUNPOD_CACHE=philipsorst/gemma-4-26B-A4B-it-UD-Q6_K_XL/gemma-4-26B-A4B-it-UD-Q6_K_XL.gguf
LLAMA_MMPROJ_RUNPOD_CACHE=philipsorst/gemma-4-26B-A4B-it-UD-Q6_K_XL/mmproj-BF16.gguf
LLAMA_REASONING=on
```

### HuggingFace Auto-Download (Convenient for development)
```
LLAMA_HF_MODEL=philipsorst/gemma-4-26B-A4B-it-UD-Q6_K_XL
```
`HF_HOME` defaults to `/runpod-volume/huggingface-cache` so it can be omitted.

### Custom sampling parameters
```
LLAMA_MODEL=/models/test.gguf
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

### Debugging model downloads
To see detailed model download progress (useful for first-time setup or troubleshooting downloads), add the verbose flag:
```
LLAMA_EXTRA_ARGS=-v
```