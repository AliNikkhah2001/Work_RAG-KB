# syntax=docker/dockerfile:1
# KB Manager — production image (CPU or CUDA via KB_*_DEVICE env at runtime)
#
# TORCH_CHANNEL=cpu  → resolve torch from PyTorch CPU index (laptop / OpenRouter CI)
# TORCH_CHANNEL=cuda → default PyPI torch resolver (GPU server builds)
ARG TORCH_CHANNEL=cpu

FROM python:3.11-slim

ARG TORCH_CHANNEL=cpu
ARG PIP_INDEX_URL=
ARG PIP_EXTRA_INDEX_URL=
ARG PIP_TRUSTED_HOST=
WORKDIR /app
ENV PIP_DEFAULT_TIMEOUT=120 \
    TORCH_CHANNEL=${TORCH_CHANNEL}

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY kb_manager ./kb_manager
COPY packs ./packs
COPY configs ./configs
COPY run_server.py ./

# BuildKit cache mount: wheels survive rebuilds on this host.
# Laptop CI MUST use TORCH_CHANNEL=cpu. A two-step "CPU then pip -e ." still lets
# the second resolve overwrite torch with +cu* from PyPI (Jenkins #43 saw 2.15.0+cu132).
# Primary index = CPU wheelhouse so torch stays +cpu and nvidia_* never enter the graph.
# PIP_EXTRA_INDEX_URL may list Iranian mirrors + pypi.org (space-separated).
RUN --mount=type=cache,target=/root/.cache/pip \
    set -e; \
    if [ -n "$PIP_INDEX_URL" ]; then export PIP_INDEX_URL; fi; \
    if [ -n "$PIP_EXTRA_INDEX_URL" ]; then export PIP_EXTRA_INDEX_URL; fi; \
    if [ -n "$PIP_TRUSTED_HOST" ]; then export PIP_TRUSTED_HOST; fi; \
    EXTRA_PYPI="${PIP_EXTRA_INDEX_URL:-https://pypi.org/simple}"; \
    if [ "${TORCH_CHANNEL}" = "cpu" ]; then \
      echo "KB image: pip -e . with CPU torch index (TORCH_CHANNEL=cpu)"; \
      pip_cmd="pip install --default-timeout=120 --index-url https://download.pytorch.org/whl/cpu"; \
      for u in $EXTRA_PYPI; do pip_cmd="$pip_cmd --extra-index-url $u"; done; \
      eval "$pip_cmd -e ."; \
      python -c "import torch,sys; v=torch.__version__; print('torch=%s cuda_available=%s'%(v,torch.cuda.is_available()));\
 sys.exit('ERROR: expected CPU torch, got '+v) if ('+cu' in v) else None"; \
    else \
      echo "KB image: TORCH_CHANNEL=${TORCH_CHANNEL} (PIP_INDEX_URL=${PIP_INDEX_URL:-default})"; \
      pip install --default-timeout=120 -e .; \
    fi

EXPOSE 8000
CMD ["python", "run_server.py"]
