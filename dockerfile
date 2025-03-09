# Use an official Python runtime as the base image
FROM python:3.11.8-slim

# Set working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install system dependencies and OpenCV requirements
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Ensure models, snapshots, and history directories exist
RUN mkdir -p models snapshots history templates

# Expose port 5000
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]