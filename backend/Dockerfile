FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Upgrade pip and set longer timeout
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --default-timeout=300 -r requirements.txt

# Copy application code
COPY . .

# Create a non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app"

# Switch to non-root user
USER appuser

# Command to run
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
