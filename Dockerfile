# -------------------------
#  Dockerfile for Medical AI Assistant
# -------------------------

# Use a lightweight Python base image
FROM python:3.13

# Prevent Python from buffering output and creating .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies for OCR, FAISS, and Tesseract
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose Flask port
EXPOSE 5000

# Default command to run the Flask app
CMD ["python", "app.py"]
