FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy requirements (IMPORTANT: your file is inside server/)
COPY server/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# Set ownership
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 7860

# Optional: remove healthcheck for now (can break build if route not present)
# HEALTHCHECK ...

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]