# Dockerfile for deploying the FastAPI app to Cloud Run
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install first for better caching
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy remaining files
COPY . .

# Cloud Run uses PORT environment variable; default to 8000
ENV PORT 8000
EXPOSE 8000

# Start the app with uvicorn
CMD ["uvicorn", "app:API", "--host", "0.0.0.0", "--port", "8000"]
