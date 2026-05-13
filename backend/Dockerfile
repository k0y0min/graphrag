# Use python 3.13 as base to match pyproject.toml
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies and uv
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Copy dependency files first
COPY pyproject.toml uv.lock ./

# Sync dependencies using uv
RUN uv sync --frozen

# Copy the application code
COPY . .

# Expose the API port
EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DB_PATH=db/demo_db

# Command to run the API
CMD sh -c "uv run uvicorn api:app --host 0.0.0.0 --port ${PORT:-8080}"
