# CrisisRoom – SRE Incident Response RL Environment
# Deployable to HuggingFace Spaces as a Docker Space
#
# HuggingFace Spaces Docker requirements:
#   - Must expose port 7860
#   - Must use CMD to start the server
#   - USER 1000 (non-root)

FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (HuggingFace Spaces requirement)
RUN useradd -m -u 1000 appuser

# Set workdir
WORKDIR /app

# Copy and install Python dependencies (without GPU-specific packages for Spaces)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY environment/ ./environment/
COPY server/ ./server/
COPY training/ ./training/

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port expected by HuggingFace Spaces
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]