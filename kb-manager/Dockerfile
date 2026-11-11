# syntax=docker/dockerfile:1
# KB Manager — production image (CPU or CUDA via KB_*_DEVICE env)
FROM python:3.11-slim

WORKDIR /app
ENV PIP_DEFAULT_TIMEOUT=120

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

# BuildKit cache mount: torch/sentence-transformers wheels survive rebuilds on this host.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=120 -e .

EXPOSE 8000
CMD ["python", "run_server.py"]
