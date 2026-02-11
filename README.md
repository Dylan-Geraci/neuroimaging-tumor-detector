[![CI](https://github.com/Dylan-Geraci/neuroimaging-tumor-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/Dylan-Geraci/neuroimaging-tumor-detector/actions/workflows/ci.yml)

> **Live Demo Note:** The frontend demo showcases the UI/UX design only. The backend API is not currently deployed due to hosting costs, but the full system (97.10% accuracy ResNet18 model with Grad-CAM visualizations) works perfectly when [run locally](#getting-started). Backend deployment was successfully implemented and tested on Render.com.

# Brain Tumor Classification System

Deep learning system for classifying brain MRI scans into four categories (Glioma, Meningioma, Pituitary Tumor, No Tumor) using a fine-tuned ResNet18 model with 97.10% test accuracy. Includes Grad-CAM visual explanations and a web interface for real-time analysis.

## Features

- 97.10% test accuracy on 1,311 held-out images
- Batch processing: analyze multiple MRI scans simultaneously
- Visual explanations via Grad-CAM attention heatmaps
- Aggregated diagnosis with agreement scoring across batches
- REST API for integration with clinical workflows
- Production-ready security: API key authentication, rate limiting, input validation
- Deployable to Railway.app with one click

## Tech Stack

**Deep Learning**: PyTorch, ResNet18 (transfer learning), Grad-CAM
**Backend**: FastAPI, uvicorn
**Frontend**: JavaScript, HTML5, CSS3
**Evaluation**: scikit-learn, matplotlib, seaborn

## Getting Started

```bash
git clone <repository-url>
cd neuroimaging-tumor-detector
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python3 main.py
# Open http://localhost:8000/
```

## CLI Tools

```bash
python src/predict.py --image path/to/mri.jpg   # Single prediction
python src/evaluate.py                           # Model evaluation
python src/visualize.py                          # Grad-CAM visualizations
```

## Model

- Architecture: ResNet18 pre-trained on ImageNet, fine-tuned on brain MRI
- Input: 224x224 grayscale MRI images
- Training: 30 epochs with Adam optimizer and data augmentation
- Dataset: 7,000+ images from the Brain Tumor MRI Dataset (Kaggle)

## Deployment

### Quick Deploy to Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/YOUR-TEMPLATE-ID)

### Environment Variables

Create a `.env` file from `.env.example`:

```bash
cp .env.example .env
```

**Required for Production:**
- `APP_ENVIRONMENT=production`
- `APP_SECRET_KEY` - Generate with: `python -c "import secrets; print(secrets.token_urlsafe(32))"`
- `APP_API_KEYS` - Comma-separated API keys for authentication
- `APP_CORS_ORIGINS` - Allowed frontend domains (e.g., `https://yourdomain.com`)
- `APP_DATABASE_URL` - PostgreSQL connection string (Railway provides this)

**Security Features:**
- `APP_RATE_LIMIT_ENABLED=true` - Enable rate limiting (10 req/min default)
- `APP_MAX_FILE_SIZE_MB=50` - Maximum upload size
- `APP_MAX_BATCH_SIZE=20` - Maximum files per batch

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for detailed deployment guide.

### Security Checklist

Before deploying to production:

- [ ] Generate strong `APP_SECRET_KEY` and `APP_API_KEYS`
- [ ] Set `APP_ENVIRONMENT=production`
- [ ] Configure CORS origins (not `*`)
- [ ] Enable rate limiting
- [ ] Upload model to Hugging Face Hub
- [ ] Set `APP_MODEL_SOURCE=huggingface`
- [ ] Verify medical disclaimer is visible in UI
- [ ] Test authentication with API keys
- [ ] Review logs for security warnings

See [docs/SECURITY.md](docs/SECURITY.md) for complete security guide.

## Docker Deployment

```bash
# Build with local model
docker build -t brain-tumor-api .

# Or build with Hugging Face model download
docker build --build-arg MODEL_SOURCE=huggingface -t brain-tumor-api .

# Run with docker-compose
docker-compose up -d
```

Access at `http://localhost:8000`

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### Single Prediction (with authentication)

```bash
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: your-api-key-here" \
  -F "file=@mri_scan.jpg"
```

### Batch Prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "X-API-Key: your-api-key-here" \
  -F "files=@scan1.jpg" \
  -F "files=@scan2.jpg" \
  -F "files=@scan3.jpg"
```

API documentation available at `/docs` endpoint.

## Development

### Local Development (No Authentication)

By default, authentication is **disabled** in development mode:

```bash
# .env file
APP_ENVIRONMENT=development
APP_API_KEYS=  # Empty = no auth required
```

All endpoints work without API keys for local development.

### Production Mode (Authentication Required)

```bash
# .env file
APP_ENVIRONMENT=production
APP_API_KEYS=key1,key2,key3
```

All prediction endpoints require `X-API-Key` header.

## Project Structure

```
neuroimaging-tumor-detector/
├── src/
│   ├── auth.py              # API key authentication
│   ├── rate_limit.py        # Rate limiting configuration
│   ├── validation.py        # Input validation
│   ├── logger.py            # Structured logging
│   ├── model_loader.py      # Multi-source model loading
│   ├── config.py            # Settings management
│   ├── model.py             # ResNet18 architecture
│   ├── predict.py           # Inference pipeline
│   └── gradcam.py           # Attention visualization
├── frontend/                # React web interface
├── docs/
│   ├── DEPLOYMENT.md        # Railway deployment guide
│   └── SECURITY.md          # Security best practices
├── scripts/
│   └── download_model.py    # Hugging Face model downloader
├── main.py                  # FastAPI backend
├── requirements.txt         # Python dependencies
└── Dockerfile               # Container configuration
```

## Disclaimer

This system is intended for research and educational purposes only. Not for clinical diagnosis.
