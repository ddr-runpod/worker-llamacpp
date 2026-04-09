# Environment Variables

## Core Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_MODEL` | (required) | Path to model file (GGUF format) |
| `LLAMA_CONTEXT_SIZE` | `4096` | Context window size |
| `LLAMA_N_GPU_LAYERS` | `99` | Layers to offload to GPU |
| `LLAMA_MMPROJ` | - | Path to multimodal projector |
| `LLAMA_PORT` | `8080` | Internal port for llama-server |
| `LLAMA_N_PARALLEL` | `1` | Parallel request slots |

## Sampling

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_TEMPERATURE` | `0.8` | Sampling temperature |
| `LLAMA_TOP_P` | `0.95` | Top-p (nucleus) sampling |
| `LLAMA_TOP_K` | `40` | Top-k sampling |

## Threading

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_THREADS` | auto | CPU threads |

## Advanced

| Variable | Description |
|----------|-------------|
| `LLAMA_EXTRA_ARGS` | Additional arguments passed directly to llama-server |

## Worker Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `80` | Port for the FastAPI app |
| `LLAMA_HOST` | `127.0.0.1` | Host address that `llama-server` binds to |
| `LLAMA_CONNECT_HOST` | `LLAMA_HOST` or `127.0.0.1` for wildcard binds | Host address the FastAPI proxy uses to reach `llama-server` |

## Notes

- `LLAMA_MODEL` is validated at startup and must be set.
- If `LLAMA_HOST` is `0.0.0.0` or `::`, the proxy automatically connects to `127.0.0.1` unless `LLAMA_CONNECT_HOST` is set.
- `LLAMA_EXTRA_ARGS` is parsed with shell-style quoting, so paths with spaces should be quoted.

## Examples

### Basic usage
```
LLAMA_MODEL=/models/llama-3.1-8b.Q4_K_M.gguf
LLAMA_CONTEXT_SIZE=8192
LLAMA_TEMPERATURE=0.7
```

### Multimodal model
```
LLAMA_MODEL=/models/llava-1.6-mistral-7b.Q4_K_M.gguf
LLAMA_MMPROJ=/models/mmproj-model-f16.gguf
```

### Using extra args
```
LLAMA_EXTRA_ARGS=--flash-attn on --rope-scaling yarn --embedding
```

### Using extra args with spaces
```
LLAMA_EXTRA_ARGS=--log-file "/tmp/llama server.log"
```
