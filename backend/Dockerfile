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

# Pre-download Qwen model to bake it into the image
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; model_id='Qwen/Qwen3-0.6B'; AutoTokenizer.from_pretrained(model_id, cache_dir='./models'); AutoModelForCausalLM.from_pretrained(model_id, cache_dir='./models')"

# Copy the application code
COPY . .

# Expose the API port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DB_PATH=db/demo_db

# Command to run the API
CMD sh -c "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"
