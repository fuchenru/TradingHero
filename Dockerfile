# Use an official Python runtime as a parent image
FROM python:3.12.2-bullseye

# Set environment variables
ENV PYTHONUNBUFFERED True
ENV APP_HOME /app

# Set working directory
WORKDIR $APP_HOME

# Expose port for Streamlit
EXPOSE 8080

# Copy the local code to the container
COPY . $APP_HOME

# Install Python dependencies
# Ensure you have a requirements.txt in your application directory, or list packages here
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Run the Streamlit application on container startup
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py", "--server.port=8080"]
