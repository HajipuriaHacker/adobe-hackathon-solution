# Use the official python:3.10 base image for amd64 architecture
FROM --platform=linux/amd64 python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required by camelot-py
RUN apt-get update && apt-get install -y --no-install-recommends ghostscript tk-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your processing script into the container
COPY process_headings.py .

# The CMD instruction specifies the command to run when the container starts.
# It runs your script, which will automatically find PDFs in /app/input.
CMD ["python", "process_headings.py"]
