# ---------- builder ----------
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build-time system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only PyTorch first, then remaining deps
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# ---------- runtime ----------
FROM python:3.12-slim AS runtime

WORKDIR /app

# OpenCV needs libgl
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY main.py .
COPY src/ src/
COPY static/ static/
COPY scripts/ scripts/

# Download model from Hugging Face if configured
ARG MODEL_SOURCE=local
ENV APP_MODEL_SOURCE=${MODEL_SOURCE}

RUN if [ "$MODEL_SOURCE" = "huggingface" ]; then \
        python scripts/download_model.py; \
    fi

# Model is mounted as a volume — not baked into the image
VOLUME ["/app/models"]

ENV APP_MODEL_PATH=/app/models/best_model.pth
ENV APP_HOST=0.0.0.0
ENV APP_PORT=8000

# Render.com uses PORT env var - this will be overridden at runtime
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request, os; urllib.request.urlopen(f'http://localhost:{os.getenv(\"PORT\", \"8000\")}/health')" || exit 1

# Use shell form to allow environment variable expansion
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
