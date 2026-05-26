# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
# PYTHONUNBUFFERED: Prevents Python from buffering stdout/stderr
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files in the container
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set the working directory
WORKDIR /app

# --- Dependency Layer ---
# Copy only requirements first to leverage Docker cache
# This layer will ONLY be rebuilt if requirements.txt changes
COPY app/requirements.txt .

# Install dependencies
# We use --no-cache-dir to keep the image size small
RUN pip install --no-cache-dir -r requirements.txt

# --- Application Layer ---
# Copy the rest of the application
# This is where code changes happen; this layer will be fast to rebuild
COPY app/ /app/app/

# Set PYTHONPATH to ensure 'import app.xxx' works across the project
ENV PYTHONPATH=/app

# Expose the port the app runs on
EXPOSE 8000

# Define the command to run the application
# Using the absolute module path for reliability
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]
