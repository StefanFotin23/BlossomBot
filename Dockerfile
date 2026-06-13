# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables for build time
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set the working directory
WORKDIR /app

# --- 1. Dependency Layer (Heavy) ---
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- 2. Model Layer (Very Heavy, but Static) ---
# Create a directory for models and download them during build time
RUN mkdir -p /app/models
# Copy only the download script and minimal package structure
COPY app/build/download_models.py app/build/
RUN touch app/__init__.py app/build/__init__.py
# Download the models and save them in /app/models
RUN python -m app.build.download_models

# --- 3. Application Layer (Light, changes often) ---
# Copy rest of the application code (except models)
COPY app/ /app/app/
COPY scripts/stress_test.py /app/scripts/stress_test.py

# Set PYTHONPATH
ENV PYTHONPATH=/app

# --- 4. Runtime Settings ---
# Activate offline mode for transformers and Hugging Face Hub
ENV TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

# Expose the port the app runs on
EXPOSE 8000

# Define the command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]
