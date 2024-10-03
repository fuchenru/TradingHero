# Use a slim Python base image
FROM python:3.8-slim-bullseye

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app

# Install system-level dependencies required by Prophet and Stan
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    make \
    libssl-dev \
    libffi-dev \
    libpython3-dev \
    libatlas-base-dev \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR $APP_HOME

# Copy only requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port for Streamlit
EXPOSE 8080

# Run the Streamlit application on container startup
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
