FROM python:3.11-slim

WORKDIR /app

# Install system libraries required by PyMuPDF for PDF text extraction
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies before copying the rest of the code.
# This layer is cached by Docker as long as requirements.txt does not change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code into the container
COPY backend/ .

# Copy the .env file from the project root into the container
COPY .env .env

# Create the uploads directory where processed PDFs will be stored
RUN mkdir -p uploads

EXPOSE 8000

CMD ["python", "main.py"]
