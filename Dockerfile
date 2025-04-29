# Use a slim Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Avoid interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# Install basic OS dependencies for nltk, spacy
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first and install
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Download NLTK data manually
RUN python -m nltk.downloader punkt averaged_perceptron_tagger words

# Copy app code
COPY . .

# Expose port for Flask/Gunicorn
EXPOSE 8080

# Start app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
