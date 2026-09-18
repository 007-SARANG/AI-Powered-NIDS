FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpcap-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies (CPU only for the container to keep it light, unless GPU is strictly required)
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose API port
EXPOSE 8000
# Expose Dashboard port
EXPOSE 8501

# By default, start the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
