#!/bin/bash

# Inject RunPod's SSH public key (set automatically by RunPod from your account keys)
if [ -n "$PUBLIC_KEY" ]; then
    mkdir -p /root/.ssh
    echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    chmod 700 /root/.ssh
    chmod 600 /root/.ssh/authorized_keys
fi

# Start SSH daemon in the background
service ssh start

# Start the FastAPI app (foreground, so the container stays alive)
exec python3.11 -u app.py