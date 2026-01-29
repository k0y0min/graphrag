# Use a slim version of Python as a base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the API port
EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DB_PATH=db/demo_db

# Command to run the API (downloads model first using runtime env vars)
CMD sh -c "python download_model.py && uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"
