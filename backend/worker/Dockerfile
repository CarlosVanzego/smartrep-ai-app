# Use the official lightweight Python image.
FROM python:3.11-slim

# Set the working directory in the container.
WORKDIR /app

# Copy the requirements file into the container.
COPY requirements.txt .

# Install the dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code.
COPY . .

# Run the web server.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app