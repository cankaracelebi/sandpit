# UNTESTED scaffold for full-scale runs on a Linux NVIDIA GPU box.
# Godot must render real frames, so it runs under Xvfb (no --headless) with the Vulkan ICD from
# the NVIDIA driver (container needs NVIDIA_DRIVER_CAPABILITIES incl. "graphics"); Mesa lavapipe
# (software Vulkan) is installed as a slow fallback.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_DRIVER_CAPABILITIES=all \
    GODOT_BIN=/usr/local/bin/godot \
    DISPLAY=:99
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl unzip xvfb xauth libvulkan1 mesa-vulkan-drivers vulkan-tools \
        libgl1 libxcursor1 libxinerama1 libxrandr2 libxi6 libxkbcommon0 libasound2 libpulse0 libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*
# Godot 4.7.2 (same version as the macOS dev machine)
RUN curl -L -o /tmp/godot.zip https://github.com/godotengine/godot/releases/download/4.7.2-stable/Godot_v4.7.2-stable_linux.x86_64.zip \
    && unzip /tmp/godot.zip -d /tmp/godot && mv /tmp/godot/Godot_v4.7.2-stable_linux.x86_64 /usr/local/bin/godot \
    && chmod +x /usr/local/bin/godot && rm -rf /tmp/godot*
# Python 3.12 via uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
WORKDIR /app
COPY pyproject.toml ./
COPY python ./python
RUN uv sync --python 3.12
COPY . .
RUN godot --headless --path godot --import || true
ENTRYPOINT ["bash", "cloud/run_full.sh"]
CMD ["configs/shooter_large.yaml"]
