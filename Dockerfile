FROM python:3.12.2-bullseye

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Expose port
EXPOSE 8080

# Set working directory
ENV APP_HOME /app
WORKDIR $APP_HOME

# Copy Streamlit app
ADD streamlit_app.py ./

# Install dependencies
RUN python -m pip install --no-cache-dir \
        streamlit==1.33.0 \
        google-generativeai==0.5.0


# Start Streamlit app
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8080"]