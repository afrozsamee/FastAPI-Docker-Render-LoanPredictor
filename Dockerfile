# Use Python 3.12 slim image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install system dependencies (required for numpy, pandas, sklearn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY ./requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY ./app /app/app
COPY ./static /app/static
# Expose port
#EXPOSE 80

# Start Gunicorn with Uvicorn workers (same behavior as tiangolo image)
#CMD ["gunicorn", "app.main:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:${PORT}"]

# Shell form (allows $PORT expansion)
CMD gunicorn app.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
