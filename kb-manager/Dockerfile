# syntax=docker/dockerfile:1
# KB Manager — production image (CPU or CUDA via KB_*_DEVICE env at runtime)
#
# TORCH_CHANNEL=cpu  → install CPU torch first (laptop / OpenRouter CI; avoids NVIDIA cudnn wheels)
# TORCH_CHANNEL=cuda → let pip resolve default torch (GPU server builds)
ARG TORCH_CHANNEL=cpu

FROM python:3.11-slim

ARG TORCH_CHANNEL=cpu
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
# Laptop CI MUST use TORCH_CHANNEL=cpu — default manylinux torch pulls nvidia_* CUDA wheels
# and stalls for 10–60+ minutes on flaky files.pythonhosted.org (see Jenkins develop #41/#42).
RUN --mount=type=cache,target=/root/.cache/pip \
    set -e; \
    if [ "${TORCH_CHANNEL}" = "cpu" ]; then \
      echo "KB image: installing CPU torch (TORCH_CHANNEL=cpu)"; \
      pip install --default-timeout=120 \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        'torch>=2.0.0'; \
    else \
      echo "KB image: TORCH_CHANNEL=${TORCH_CHANNEL} (default torch resolver)"; \
    fi; \
    pip install --default-timeout=120 -e .

EXPOSE 8000
CMD ["python", "run_server.py"]
