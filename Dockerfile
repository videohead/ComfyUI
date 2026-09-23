# NVIDIA-accelerated ComfyUI + comfy-cli + Comfy MCP (local connection)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    COMFY_WORKSPACE=/workspace/ComfyUI \
    COMFY_BIN=/usr/local/bin/comfy

# Ubuntu 22.04's universe python3.11 package is a stale 3.11.0~rc1 build missing
# stdlib APIs (e.g. sys.get_int_max_str_digits) that current torch expects, so
# pull a real released Python 3.11 from deadsnakes instead.
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
        git ffmpeg libgl1 libglib2.0-0 curl build-essential \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# comfy-cli drives install/launch; comfy-mcp is the local MCP server on top of it
RUN pip install --upgrade pip \
    && pip install "comfy-cli>=1.14.0" comfy-mcp ffmpeg-python

# Installs ComfyUI itself with CUDA-matched torch into $COMFY_WORKSPACE.
# --cuda-version 13.0 is required for Blackwell (sm_120, RTX 50-series) support;
# older auto-selected builds (cu126) lack kernels for those GPUs.
RUN comfy --skip-prompt --workspace ${COMFY_WORKSPACE} install --nvidia --cuda-version 13.0 --fast-deps \
    && comfy --skip-prompt set-default ${COMFY_WORKSPACE}

# Model and output directories are bind-mounted at runtime for persistence, see docker-compose.yml
VOLUME ["${COMFY_WORKSPACE}/models", "${COMFY_WORKSPACE}/output", "${COMFY_WORKSPACE}/input", "${COMFY_WORKSPACE}/user", "${COMFY_WORKSPACE}/custom_nodes"]

EXPOSE 8188

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
