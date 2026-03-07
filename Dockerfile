# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (needed for pdfplumber/chromium if scraper runs here, but mostly just standard libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install standard Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Expose port (Render defaults to 10000)
EXPOSE 10000

# Start the FastAPI application
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "10000"]
