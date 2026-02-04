# Quick Start Guide

This guide will help you get the Brain Tumor Classification API running locally with all security features.

## Prerequisites

- Python 3.10+
- Node.js 18+ (for frontend)
- Git

## Installation

### 1. Clone Repository

```bash
cd /Users/dylangeraci/Desktop/personal-projects/neuroimaging-tumor-detector
```

### 2. Create Environment File

```bash
cp .env.example .env
```

The default `.env` is configured for local development (no authentication required).

### 3. Install Python Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (including new security packages)
pip install -r requirements.txt
```

This will install 7 new security-related packages:
- `python-dotenv` - Environment variable management
- `python-jose` & `passlib` - Authentication
- `slowapi` - Rate limiting
- `python-json-logger` - Structured logging
- `psycopg2-binary` - PostgreSQL support
- `huggingface-hub` - Model hosting

### 4. Start Backend

```bash
python3 main.py
```

Expected output:
```
2024-01-15 10:00:00 - brain_tumor_api - INFO - Loading model...
2024-01-15 10:00:01 - brain_tumor_api - INFO - Using device: cpu
2024-01-15 10:00:05 - brain_tumor_api - INFO - Model loaded from models/best_model.pth
2024-01-15 10:00:05 - brain_tumor_api - INFO - Best validation accuracy: 0.971
2024-01-15 10:00:05 - brain_tumor_api - INFO - Grad-CAM initialized
2024-01-15 10:00:05 - brain_tumor_api - INFO - Ready to accept requests!
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 5. Install Frontend Dependencies

```bash
cd frontend
npm install
```

### 6. Start Frontend

```bash
npm run dev
```

Expected output:
```
VITE v5.0.0  ready in 300 ms

➜  Local:   http://localhost:5173/
➜  Network: use --host to expose
```

### 7. Open Application

Navigate to: http://localhost:5173

You should see:
- ⚠️ Yellow medical disclaimer banner (newly added)
- Upload area for MRI scans
- All features working without authentication (dev mode)

---

## Verify Installation

### Test 1: Health Check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "gradcam_loaded": true,
  "device": "cpu",
  "classes": ["glioma", "meningioma", "no_tumor", "pituitary"]
}
```

### Test 2: API Documentation

Visit: http://localhost:8000/docs

You should see:
- Swagger UI with all endpoints
- New security scheme: "APIKeyHeader" (X-API-Key)
- All endpoints documented

### Test 3: Single Prediction (No Auth in Dev)

```bash
# Create a test image (or use a real MRI scan)
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/test/image.jpg"
```

Expected response:
```json
{
  "success": true,
  "prediction": {
    "class": "glioma",
    "class_index": 0,
    "confidence": 0.95,
    "probabilities": {...}
  },
  "images": {
    "original": "data:image/png;base64,...",
    "heatmap": "data:image/png;base64,...",
    "overlay": "data:image/png;base64,..."
  }
}
```

### Test 4: Medical Disclaimer Visible

Open http://localhost:5173 and verify:
- Yellow banner appears below header
- Contains warning icon (⚠️)
- Text: "Educational Use Only: This system is for research and educational purposes..."

---

## Test Security Features

### Enable Production Mode

Edit `.env`:
```bash
APP_ENVIRONMENT=production
APP_API_KEYS=testkey123,testkey456
APP_RATE_LIMIT_ENABLED=true
```

Restart backend:
```bash
# Stop server (Ctrl+C)
python3 main.py
```

### Test Authentication

```bash
# Without API key (should fail)
curl -X POST http://localhost:8000/predict \
  -F "file=@test.jpg"

# Expected: 401 Unauthorized
# Response: {"detail": "Missing API key. Include X-API-Key header."}

# With valid API key (should work)
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: testkey123" \
  -F "file=@test.jpg"

# Expected: 200 OK with prediction
```

### Test Rate Limiting

```bash
# Make 11 rapid requests
for i in {1..11}; do
  echo "Request $i:"
  curl -X POST http://localhost:8000/predict \
    -H "X-API-Key: testkey123" \
    -F "file=@test.jpg" \
    -w "\nStatus: %{http_code}\n\n"
done
```

Expected:
- Requests 1-10: 200 OK
- Request 11: 429 Too Many Requests
- Response: `{"error": "Rate limit exceeded", "detail": "Too many requests..."}`

### Test File Validation

```bash
# Create 51MB file (exceeds limit)
dd if=/dev/zero of=large.jpg bs=1M count=51

# Upload large file
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: testkey123" \
  -F "file=@large.jpg"

# Expected: 400 Bad Request
# Response: {"detail": "File too large (51.0MB). Max: 50MB"}
```

---

## Development vs Production

### Development Mode (Default)

```bash
# .env
APP_ENVIRONMENT=development
APP_API_KEYS=  # Empty
APP_RATE_LIMIT_ENABLED=false
```

Features:
- ✅ No authentication required
- ✅ No rate limiting
- ✅ Detailed error messages
- ✅ Human-readable logs
- ✅ CORS allows all origins

Perfect for local development and testing.

### Production Mode

```bash
# .env
APP_ENVIRONMENT=production
APP_API_KEYS=key1,key2,key3
APP_RATE_LIMIT_ENABLED=true
APP_CORS_ORIGINS=https://yourdomain.com
```

Features:
- 🔒 API key required for predictions
- 🔒 Rate limiting enforced (10/min)
- 🔒 Generic error messages
- 🔒 JSON structured logs
- 🔒 CORS restricted to allowed domains

---

## Common Issues

### Issue: ModuleNotFoundError

**Symptom:**
```
ModuleNotFoundError: No module named 'slowapi'
```

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: Model Not Found

**Symptom:**
```
FileNotFoundError: Model not found at models/best_model.pth
```

**Solution:**
```bash
# Ensure model file exists
ls -lh models/best_model.pth

# If missing, train model or download from Hugging Face
# See docs/DEPLOYMENT.md for Hugging Face setup
```

### Issue: Port Already in Use

**Symptom:**
```
OSError: [Errno 48] Address already in use
```

**Solution:**
```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>

# Or change port in .env
APP_PORT=8001
```

### Issue: Frontend Can't Connect to Backend

**Symptom:**
```
Failed to fetch http://localhost:8000/health
```

**Solution:**
1. Ensure backend is running (`python3 main.py`)
2. Check backend logs for errors
3. Verify port 8000 is accessible: `curl http://localhost:8000/health`
4. Check CORS settings in `.env`

---

## Next Steps

### For Local Development
1. ✅ You're all set! Start building features.
2. Keep `APP_ENVIRONMENT=development` for easier testing
3. Use `/docs` endpoint to explore API

### For Production Deployment
1. Read [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for Railway.app guide
2. Generate secure keys: `python -c "import secrets; print(secrets.token_urlsafe(32))"`
3. Upload model to Hugging Face Hub
4. Follow security checklist in [docs/SECURITY.md](docs/SECURITY.md)

### For Contributing
1. Create a new branch: `git checkout -b feature/my-feature`
2. Make changes and test locally
3. Run tests: `pytest` (if available)
4. Submit pull request

---

## File Structure

```
neuroimaging-tumor-detector/
├── .env.example          # Environment template (NEW)
├── .env                  # Your local config (create from .env.example)
├── main.py              # FastAPI backend (MODIFIED - auth, logging)
├── requirements.txt     # Python deps (MODIFIED - 7 new packages)
├── src/
│   ├── auth.py          # API key authentication (NEW)
│   ├── rate_limit.py    # Rate limiting (NEW)
│   ├── logger.py        # Structured logging (NEW)
│   ├── validation.py    # Input validation (NEW)
│   ├── model_loader.py  # Model loading (NEW)
│   ├── config.py        # Settings (MODIFIED - 10 new fields)
│   ├── model.py
│   ├── predict.py
│   └── gradcam.py
├── frontend/
│   └── src/
│       ├── App.tsx              # Main app (MODIFIED - disclaimer)
│       └── styles/globals.css   # Styles (MODIFIED - disclaimer styles)
├── scripts/
│   └── download_model.py   # HuggingFace downloader (NEW)
├── docs/
│   ├── DEPLOYMENT.md      # Railway guide (NEW)
│   └── SECURITY.md        # Security guide (NEW)
└── models/
    └── best_model.pth     # Trained model (128MB)
```

---

## Support

- **Issues**: Create a GitHub issue
- **Questions**: Check docs/DEPLOYMENT.md and docs/SECURITY.md
- **Security**: See docs/SECURITY.md for security contacts

---

## Summary

You now have a production-ready Brain Tumor Classification API with:

✅ API key authentication (disabled in dev, enabled in prod)
✅ Rate limiting (10 requests/minute in production)
✅ Input validation (file size, type, batch limits)
✅ Structured logging (JSON in production)
✅ Secure error handling (no internal details leaked)
✅ CORS hardening (restricted origins in production)
✅ Model hosting support (Hugging Face Hub)
✅ Medical disclaimer (visible in UI)
✅ Comprehensive documentation (deployment, security)

**Development workflow unchanged** - just run `python3 main.py` as before!

**Production deployment** - Follow [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for Railway.app
