# Deployment Guide

## Prerequisites

1. GitHub account (for deploying from repository)
2. Render.com account (free)
3. Vercel account (free)
4. Model file (`best_model.pth`) available

## Part 1: Deploy Backend to Render.com

### 1.1 Prepare Model File

Your model needs to be accessible at runtime. Options:

**Option A: Use Render Persistent Disk** (Recommended)
1. After creating the service, add a persistent disk
2. Upload your model to `/app/models/best_model.pth`

**Option B: Download from external storage**
1. Upload model to Google Drive, Dropbox, or S3
2. Add download step in Dockerfile or startup script

### 1.2 Deploy to Render

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: `neuroimaging-tumor-detector`
   - **Environment**: `Docker`
   - **Plan**: `Free`
   - **Branch**: `main`

5. Click "Advanced" and set environment variables:
   ```
   PORT=10000 (auto-set by Render)
   APP_MODEL_PATH=/app/models/best_model.pth
   APP_CORS_ORIGINS=https://your-frontend.vercel.app
   API_KEY=<generate-a-secure-random-key>
   LOG_LEVEL=INFO
   RATE_LIMIT_PER_MINUTE=60
   ```

6. Click "Create Web Service"

7. Wait for deployment (5-10 minutes for first build)

8. Note your backend URL: `https://neuroimaging-tumor-detector.onrender.com`

### 1.3 Upload Model

If using persistent disk:
1. In your service settings, go to "Disks"
2. Create a disk mounted at `/app/models`
3. Use Render Shell or SSH to upload your model:
   ```bash
   scp models/best_model.pth render:/app/models/
   ```

### 1.4 Test Backend

Visit: `https://your-backend.onrender.com/health`

Should return:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "gradcam_loaded": true,
  "device": "cpu",
  "classes": ["glioma", "meningioma", "notumor", "pituitary"]
}
```

## Part 2: Deploy Frontend to Vercel

### 2.1 Configure API URL

1. Create `.env` file in `frontend/` directory:
   ```env
   VITE_API_URL=https://your-backend.onrender.com
   VITE_API_KEY=<your-api-key-from-render>
   ```

2. Update frontend API calls to use environment variables

### 2.2 Deploy to Vercel

**Option A: Vercel CLI**
```bash
cd frontend
npm install -g vercel
vercel login
vercel --prod
```

**Option B: Vercel Dashboard**
1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Click "Add New" → "Project"
3. Import your GitHub repository
4. Configure:
   - **Root Directory**: `frontend`
   - **Framework Preset**: Vite
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`

5. Add environment variables:
   ```
   VITE_API_URL=https://your-backend.onrender.com
   VITE_API_KEY=<your-api-key>
   ```

6. Click "Deploy"

7. Note your frontend URL: `https://your-app.vercel.app`

### 2.3 Update CORS

Update your Render backend environment variable:
```
APP_CORS_ORIGINS=https://your-app.vercel.app
```

Redeploy backend for changes to take effect.

## Part 3: Verify Deployment

1. Visit your Vercel frontend URL
2. Upload a test image
3. Verify prediction works end-to-end

## Security Checklist

- ✅ API key authentication enabled
- ✅ Rate limiting configured (60 req/min)
- ✅ CORS restricted to frontend domain
- ✅ File upload validation (type, size)
- ✅ Request logging enabled
- ✅ HTTPS enforced (automatic on Render/Vercel)

## Cost Breakdown

| Service | Plan | Cost |
|---------|------|------|
| Render.com | Free | $0/month |
| Vercel | Hobby | $0/month |
| **Total** | | **$0/month** |

### Free Tier Limits

**Render.com Free Tier:**
- 750 hours/month
- Spins down after 15 min inactivity
- Cold start: ~30-60 seconds
- 512 MB RAM

**Vercel Hobby Tier:**
- 100 GB bandwidth/month
- Unlimited requests
- Edge network (fast globally)

## Troubleshooting

### Backend won't start
- Check logs in Render dashboard
- Verify model file exists at correct path
- Check environment variables

### Frontend can't connect to backend
- Verify CORS origins match
- Check API URL in frontend .env
- Test backend `/health` endpoint directly

### Model loading fails
- Ensure model file is accessible
- Check file path matches `APP_MODEL_PATH`
- Verify sufficient memory (may need paid tier for large models)

### Rate limiting issues
- Adjust `RATE_LIMIT_PER_MINUTE` environment variable
- Check logs for blocked requests

## Monitoring

### Render.com
- View logs: Dashboard → Service → Logs
- Metrics: Dashboard → Service → Metrics
- Health checks: Automatic via `/health` endpoint

### Vercel
- Analytics: Dashboard → Project → Analytics
- Logs: Dashboard → Project → Deployments → View Function Logs

## Scaling Up

If you need more performance:

**Backend ($7/month - Starter):**
- No cold starts
- 512 MB RAM
- Background workers

**Backend ($25/month - Standard):**
- 2 GB RAM
- Better for larger models

**Vercel (stays free):**
- Hobby tier usually sufficient for frontend

## Environment Variables Reference

### Backend (Render)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | Yes (auto) | 10000 | Server port (set by Render) |
| `APP_MODEL_PATH` | Yes | - | Path to model file |
| `APP_CORS_ORIGINS` | Yes | * | Allowed CORS origins (comma-separated) |
| `API_KEY` | Recommended | - | API authentication key |
| `LOG_LEVEL` | No | INFO | Logging level |
| `RATE_LIMIT_PER_MINUTE` | No | 60 | Rate limit per IP |

### Frontend (Vercel)

| Variable | Required | Description |
|----------|----------|-------------|
| `VITE_API_URL` | Yes | Backend API URL |
| `VITE_API_KEY` | Yes | API key for authentication |

## Next Steps

1. Set up monitoring/alerting
2. Configure custom domain (optional)
3. Set up CI/CD for automatic deployments
4. Implement database backups
5. Add user authentication (if needed)
