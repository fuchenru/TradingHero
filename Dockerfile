FROM python:3.12.2-bullseye

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app

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