# Use the Python 3.11.9 base image to match your local environment
FROM python:3.11.9-slim

# STEP 1: Install system dependencies required by libraries like camelot-py
RUN apt-get update && apt-get install -y --no-install-recommends \
    ghostscript \
    build-essential \
    python3-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set the working directory inside the container
WORKDIR /app

# STEP 2: Set environment variables for offline data access
# Tell sentence-transformers where to find models inside the container
ENV SENTENCE_TRANSFORMERS_HOME=/app/models
# Tell NLTK where to find its data
ENV NLTK_DATA=/app/nltk_data

# STEP 3: Install Python libraries from your requirements.txt
# This leverages Docker caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# STEP 4: Download NLTK data during the build process
RUN python -c "import nltk; nltk.download('punkt', quiet=True, download_dir='/app/nltk_data'); nltk.download('punkt_tab', quiet=True, download_dir='/app/nltk_data')"

# STEP 5: Copy your entire application into the container
# This copies all scripts AND the './models' directory into the image,
# making the container self-contained and able to run offline.
COPY . .

# Define the command to run your application.
ENTRYPOINT ["python", "main.py"]