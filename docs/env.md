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

## Batch Size & Attention (Optional)

| Variable | Description |
|----------|-------------|
| `LLAMA_FLASH_ATTN` | Enable/disable Flash Attention. Accepted values: `on`, `1`, `yes`, `off`, `0`, `no` (case-insensitive). Passed via `--flash-attn` flag. Upstream default: `auto`. |
| `LLAMA_BATCH_SIZE` | Logical maximum batch size for prompt processing. Passed via `-b` flag. Upstream default: `2048`. |
| `LLAMA_UBATCH_SIZE` | Physical (unbatched) maximum batch size for generation. Passed via `-ub` flag. Upstream default: `512`. |

## Multimodal (Optional)

| Variable | Description |
|----------|-------------|
| `LLAMA_MMPROJ` | Path to a multimodal projection (mmproj) GGUF file. Used for vision/language models. Passed via `--mmproj` flag. |
| `LLAMA_MMPROJ_RUNPOD_CACHE` | RunPod cached mmproj path in `org/name/filename` format. Auto-resolves to the correct snapshot path. Passed via `--mmproj` flag. |

## Speculative Decoding (Optional)

| Variable | Description |
|----------|-------------|
| `LLAMA_SPEC_DRAFT_MODEL` | Local path to a draft GGUF model for speculative decoding. Passed via `--spec-draft-model` flag. |
| `LLAMA_SPEC_DRAFT_MODEL_RUNPOD_CACHE` | RunPod cached draft model path in `org/name/filename` format. Auto-resolves to the correct snapshot path. Passed via `--spec-draft-model` flag. |
| `LLAMA_SPEC_TYPE` | Comma-separated list of speculative decoding types (e.g., `draft-mtp`, `draft-eagle3`, `ngram-simple`). Passed verbatim via `--spec-type` flag. |
| `LLAMA_SPEC_DRAFT_N_MAX` | Maximum number of tokens to draft for speculative decoding (default: 3). Passed via `--spec-draft-n-max` flag. |

Exactly one of `LLAMA_SPEC_DRAFT_MODEL` or `LLAMA_SPEC_DRAFT_MODEL_RUNPOD_CACHE` may be set.

## Model Identity (Optional)

| Variable | Description |
|----------|-------------|
| `LLAMA_ALIAS` | Comma-separated model name aliases returned by API endpoints (e.g., `gemma-4-26B,my-alias`). Passed verbatim via `--alias` flag. |

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
- `LLAMA_SPEC_DRAFT_MODEL` and `LLAMA_SPEC_DRAFT_MODEL_RUNPOD_CACHE` are passed via `--spec-draft-model` flag for speculative decoding draft models.
- `LLAMA_SPEC_TYPE` is passed verbatim via `--spec-type` flag. `LLAMA_SPEC_DRAFT_N_MAX` is passed via `--spec-draft-n-max` flag.
- `LLAMA_ALIAS` is passed verbatim via `--alias` flag; multiple aliases can be set as a comma-separated list.
- `LLAMA_FLASH_ATTN` is passed via `--flash-attn` flag (boolean). `LLAMA_BATCH_SIZE` is passed via `-b` flag; `LLAMA_UBATCH_SIZE` is passed via `-ub` flag.
- Continuous batching (`--cont-batching`) is enabled by default upstream and intentionally not exposed as an env var.
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

### Speculative decoding (Gemma 4 MTP)
```
LLAMA_MODEL_RUNPOD_CACHE=philipsorst/gemma-4-26B-A4B-it-UD-Q6_K_XL/gemma-4-26B-A4B-it-UD-Q6_K_XL.gguf
LLAMA_MMPROJ_RUNPOD_CACHE=philipsorst/gemma-4-26B-A4B-it-UD-Q6_K_XL/mmproj-BF16.gguf
LLAMA_SPEC_DRAFT_MODEL_RUNPOD_CACHE=philipsorst/gemma-4-26B-A4B-it-UD-Q6_K_XL/mtp-gemma-4-26B-A4B-it.gguf
LLAMA_SPEC_TYPE=draft-mtp
LLAMA_SPEC_DRAFT_N_MAX=2
```

### Debugging model downloads
To see detailed model download progress (useful for first-time setup or troubleshooting downloads), add the verbose flag:
```
LLAMA_EXTRA_ARGS=-v
```