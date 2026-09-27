# Dockerfile
FROM python:3.12.14-slim-trixie

WORKDIR /home

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Install essential system tools first
# wget and gnupg are needed for the ROCm repository setup
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg \
    libatomic1 \ 
    libquadmath0 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get install amdrocm7.14-gfx1102

# Copy requirement files first (for better caching)
COPY . /home

RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other Python dependencies
RUN pip3 install --no-cache-dir \
    fastapi
    uvicorn[standard] \
    python-multipart  \     
    pillow \
    pydantic>=2

# Default command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]