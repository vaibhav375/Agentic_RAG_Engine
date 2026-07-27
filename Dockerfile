FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# System deps kept minimal; add build-essential only if you compile faiss/torch.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

# Install core + store extras. Add ".[local]" or ".[api]" if you want the heavy
# ML backends baked into the image.
RUN pip install --upgrade pip && pip install -e ".[store,api,viz]"

COPY config ./config
COPY eval ./eval
COPY data ./data

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --retries=5 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["python", "-m", "arag.cli", "serve", "--config", "config/config.yaml"]
