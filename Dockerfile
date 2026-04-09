# Stage 1: Get llama-server binary from llama.cpp official image
FROM ghcr.io/ggml-org/llama.cpp:server-cuda AS llama-build

# Stage 2: RunPod base with CUDA support
FROM runpod/base:1.0.3-cuda1281-ubuntu2404

# Copy llama-server from build stage
COPY --from=llama-build /app/llama-server /usr/local/bin/llama-server

# Set Python version (runpod/base includes multiple Python versions)
RUN ln -sf $(which python3.11) /usr/local/bin/python

# Install Python dependencies via uv (faster, better caching)
COPY requirements.txt /requirements.txt
RUN uv pip install --system -r /requirements.txt

# Copy application code
COPY app.py config.py llama_proxy.py ./

# Run the FastAPI app (unbuffered output for proper log streaming)
CMD ["python", "-u", "app.py"]