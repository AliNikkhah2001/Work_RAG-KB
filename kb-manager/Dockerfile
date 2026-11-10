# KB Manager — production image (CPU or CUDA via KB_*_DEVICE env)
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY kb_manager ./kb_manager
COPY run_server.py ./

RUN pip install --no-cache-dir -e .

EXPOSE 8000
CMD ["python", "run_server.py"]
