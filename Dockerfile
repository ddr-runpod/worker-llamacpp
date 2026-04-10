# Use the official llama.cpp CUDA server image as the base
# llama-server and all shared libraries (libmtmd.so.0, etc.) are already present
FROM ghcr.io/ggml-org/llama.cpp:server-cuda

# Install Python 3, uv, and openssh-server
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    curl \
    openssh-server \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ssh-keygen -A \
    && mkdir -p /run/sshd

ENV PATH="/root/.local/bin:${PATH}" \
    VIRTUAL_ENV=/opt/venv

ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Permit root login via key (no password)
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config \
    && sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config

# Install Python dependencies
COPY requirements.txt /requirements.txt
RUN uv venv $VIRTUAL_ENV && uv pip install --python $VIRTUAL_ENV -r /requirements.txt

# Copy application code and startup script
COPY app.py config.py llama_proxy.py rplog.py ./
COPY start.sh /start.sh
RUN chmod +x /start.sh

ENTRYPOINT ["/start.sh"]