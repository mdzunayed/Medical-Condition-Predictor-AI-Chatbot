# Docker Local Testing Guide

Quick reference for testing the Docker build locally before deploying to Hugging Face Spaces.

---

## 🚀 Quick Start (5 minutes)

### 1. Build the Docker Image

```bash
# From project root directory
docker build -t medical-diagnosis-ai:latest .
```

**What to expect:**
- First time: 3-5 minutes (downloads dependencies)
- Subsequent: 1-2 minutes (uses cache)
- Final image size: 600-800MB

### 2. Run the Container

```bash
docker run -p 7860:7860 medical-diagnosis-ai:latest
```

**What you should see:**
```
Starting Medical Diagnosis AI Server (💻 Local Development)
📍 Server: http://0.0.0.0:7860
🌐 Frontend: http://0.0.0.0:7860
📊 API Base: http://0.0.0.0:7860/api
📖 API Docs: http://0.0.0.0:7860/docs
```

### 3. Test the Application

**Open in browser:**
- Frontend: http://localhost:7860
- API Docs: http://localhost:7860/docs
- Health Check: http://localhost:7860/health

---

## 🧪 Testing Scenarios

### Test 1: Frontend Loads

```bash
curl http://localhost:7860
# Should return HTML content (React app)
```

### Test 2: API Responds

```bash
curl http://localhost:7860/health
# Should return: {"status": "ok", "service": "Medical Diagnosis AI"}
```

### Test 3: Chat Endpoint

```bash
curl -X POST http://localhost:7860/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": null,
    "message": "I am 34 years old",
    "history": []
  }'
```

### Test 4: Full Assessment Flow

```python
import requests
import json

BASE_URL = "http://localhost:7860/api"

# Test messages that cover different features
messages = [
    "I am 35 years old",
    "My glucose is 120 mg/dL",
    "My HbA1c is 6.5%",
    "My BMI is 28",
    "My cholesterol is 200",
    "My triglycerides are 150",
    "My blood pressure is 130/85",
    "I exercise 3 times a week",
    "I sleep 7 hours per night",
    "My stress level is 6 out of 10",
    "My diet score is 75",
    "I don't smoke",
    "I drink socially",
    "No family history of disease",
    "I've been in the hospital 2 days",
    "My oxygen saturation is 98%"
]

session_id = None
for i, msg in enumerate(messages, 1):
    response = requests.post(f"{BASE_URL}/chat", json={
        "session_id": session_id,
        "message": msg,
        "history": []
    })
    
    data = response.json()
    session_id = data["session_id"]
    
    print(f"\n[{i}/{len(messages)}] {msg}")
    print(f"Status: {data['message'][:100]}...")
    print(f"Collected: {data['collected_count']}/16")
    
    if data["is_complete"]:
        print(f"\n✅ ASSESSMENT COMPLETE!")
        print(f"Diagnosis: {data['prediction']['prediction_name']}")
        print(f"Confidence: {data['prediction']['confidence']*100:.1f}%")
        break
```

---

## 🐳 Docker Compose (Advanced)

For easier local testing with auto-rebuild on code changes:

### Create `docker-compose.yml`

```yaml
version: '3.8'

services:
  medical-ai:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: medical-diagnosis-ai
    ports:
      - "7860:7860"
    environment:
      - SPACE_ID=local-test
      - PYTHONUNBUFFERED=1
    volumes:
      - ./app:/home/user/app/app
      - ./server.py:/home/user/app/server.py
    restart: unless-stopped
```

### Run with Docker Compose

```bash
# Start
docker-compose up --build

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## 🔍 Debugging

### View Container Logs

```bash
# While running
docker ps

# Get container ID and view logs
docker logs <container-id>

# Follow logs in real-time
docker logs -f <container-id>
```

### Interactive Shell

```bash
docker run -it -p 7860:7860 medical-diagnosis-ai:latest /bin/bash

# Inside container
python server.py
```

### Check Container Status

```bash
docker ps -a
docker inspect <container-id>
docker stats <container-id>
```

---

## 🛠️ Common Issues & Fixes

### Issue: "Port 7860 is already in use"

```bash
# Find process using port
lsof -i :7860

# Kill it
kill -9 <PID>

# Or use different port
docker run -p 7861:7860 medical-diagnosis-ai:latest
```

### Issue: "Cannot find module 'app.api'"

**Solution:** Make sure you're in the project root directory:

```bash
pwd
# Should show: /path/to/medical-predictor-chatbot

ls app/api.py
# Should exist
```

### Issue: "Frontend blank/404"

**Check:**
1. Frontend was built: `ls frontend/dist/index.html`
2. Build it if missing: `cd frontend && npm install && npm run build`
3. Rebuild Docker image: `docker build --no-cache -t medical-diagnosis-ai:latest .`

### Issue: "Health check failing"

The Dockerfile includes a health check. If it fails:

```bash
# Check if curl is available
docker run medical-diagnosis-ai:latest which curl

# If not, the health check will fail but app still works
# You can still test manually: curl http://localhost:7860/health
```

---

## 📊 Performance Testing

### Check Image Size

```bash
docker images medical-diagnosis-ai:latest
# Shows: REPOSITORY  TAG     IMAGE ID  CREATED   SIZE
```

### Check Memory Usage

```bash
docker stats medical-diagnosis-ai
# Watch CPU and memory in real-time
```

### Build Time Analysis

```bash
# Time the build
time docker build -t medical-diagnosis-ai:latest .
```

---

## 🧹 Cleanup

### Remove Container

```bash
docker rm <container-id>
```

### Remove Image

```bash
docker rmi medical-diagnosis-ai:latest
```

### Deep Cleanup (Remove Everything)

```bash
# Remove all stopped containers
docker container prune

# Remove all unused images
docker image prune

# Remove all unused volumes
docker volume prune

# Complete reset (⚠️ Deletes everything)
docker system prune -a
```

---

## ✅ Pre-Deployment Checklist

Before pushing to Hugging Face Spaces:

- [ ] Docker image builds successfully: `docker build -t medical-diagnosis-ai:latest .`
- [ ] Container starts without errors: `docker run -p 7860:7860 medical-diagnosis-ai:latest`
- [ ] Frontend loads at http://localhost:7860
- [ ] API responds at http://localhost:7860/api/chat
- [ ] Health check passes: http://localhost:7860/health
- [ ] Can complete full assessment flow
- [ ] No errors in Docker logs: `docker logs <container-id>`
- [ ] Image size is reasonable: `docker images`
- [ ] All environment variables work
- [ ] Non-root user is running app: `docker run medical-diagnosis-ai:latest whoami`

---

## 🎯 Expected Output

When everything works correctly:

```
Starting Medical Diagnosis AI Server (💻 Local Development)
📍 Server: http://0.0.0.0:7860
🌐 Frontend: http://0.0.0.0:7860
📊 API Base: http://0.0.0.0:7860/api
📖 API Docs: http://0.0.0.0:7860/docs
INFO: Uvicorn running on http://0.0.0.0:7860 (Press CTRL+C to quit)
INFO: Started server process [1]
```

---

## 📝 Docker Command Reference

```bash
# Build
docker build -t medical-diagnosis-ai:latest .

# Run
docker run -p 7860:7860 medical-diagnosis-ai:latest

# Run with env var
docker run -p 7860:7860 -e SPACE_ID=test medical-diagnosis-ai:latest

# Run interactive
docker run -it -p 7860:7860 medical-diagnosis-ai:latest /bin/bash

# View logs
docker logs <container-id>
docker logs -f <container-id>

# Stop container
docker stop <container-id>

# List containers
docker ps
docker ps -a

# List images
docker images

# Remove image
docker rmi medical-diagnosis-ai:latest

# View image size
docker images medical-diagnosis-ai:latest

# Inspect container
docker inspect <container-id>

# View stats
docker stats <container-id>

# Execute command in running container
docker exec -it <container-id> /bin/bash
```

---

**Status:** ✅ READY FOR LOCAL TESTING  
**Last Updated:** 2026-04-07
