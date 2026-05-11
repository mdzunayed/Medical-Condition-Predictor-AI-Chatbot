# Multi-stage build for Medical Diagnosis AI
# Stage 1: Build React frontend with Node.js
# Stage 2: Python backend serving both API and static files

# ============================================================
# Stage 1: Build React Frontend
# ============================================================
FROM node:18-slim AS frontend-builder

WORKDIR /build/frontend

# Copy frontend source code
COPY frontend/package*.json ./
RUN npm install --frozen-lockfile

# Copy frontend source
COPY frontend/ .

# Build the frontend (creates dist/ folder)
RUN npm run build

# ============================================================
# Stage 2: Python Backend with Static File Serving
# ============================================================
FROM python:3.10-slim

# Set metadata
LABEL maintainer="Medical Diagnosis AI Team"
LABEL description="Medical Diagnosis AI - Full Stack Application"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user with UID 1000 (Hugging Face Spaces requirement)
RUN groupadd -r user && useradd -r -u 1000 -g user user

# Set working directory
WORKDIR /home/user/app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Install curl for health checks
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy backend application code
COPY app/ ./app/
COPY server.py .

# Copy the ML model (CRITICAL - needed for disease predictions)
COPY models/ ./models/

# Create frontend dist directory and copy built frontend from Stage 1
RUN mkdir -p frontend
COPY --from=frontend-builder /build/frontend/dist ./frontend/dist

# Change ownership of all files to the non-root user
RUN chown -R user:user /home/user/app

# Switch to non-root user
USER user

# Expose port 7860 (Hugging Face Spaces standard)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run the server
CMD ["python", "server.py"]
