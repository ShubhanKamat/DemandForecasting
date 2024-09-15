# Using official Python image as the base image
FROM python:3.9-slim

# Setting environment variables to avoid buffering issues
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

# Setting the working directory in the container
WORKDIR /app

# Copying the requirements file to the container
COPY requirements.txt /app/

# Installing any system dependencies and Python dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && pip install --upgrade pip \
    && pip install -r requirements.txt \
    && apt-get purge -y --auto-remove gcc \
    && rm -rf /var/lib/apt/lists/*

# Copying the rest of the application code to the working directory
COPY . /app/

# Exposing the port the app will run on
EXPOSE 5000

# setting the environment variable for Flask
ENV FLASK_APP=app/app.py
ENV FLASK_ENV=production

# Command to run the Flask app using Gunicorn (Production-ready WSGI server)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "wsgi:app"]
