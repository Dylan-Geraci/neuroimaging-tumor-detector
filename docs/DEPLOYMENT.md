# Deployment Guide - Railway.app

> **Note:** Alternative deployment guide. Backend tested on Render.com (see main DEPLOYMENT.md).

---

This guide walks through deploying the Brain Tumor Classification API to Railway.app for production use.

## Prerequisites

- Railway.app account (free tier available)
- Hugging Face account (for model hosting)
- Git repository (GitHub/GitLab)
- Model file uploaded to Hugging Face Hub

---

## Step 1: Prepare Model for Hugging Face

### 1.1 Create Hugging Face Account & Repository

1. Sign up at [huggingface.co](https://huggingface.co)
2. Create a new model repository:
   - Click "New Model" → Name it `brain-tumor-classifier`
   - Set to Public (or Private with access token)

### 1.2 Upload Model

```bash
# Install Hugging Face CLI
pip install huggingface-hub

# Login
hf login

# Upload model (replace YOUR-USERNAME)
hf upload YOUR-USERNAME/brain-tumor-classifier models/best_model.pth
```

### 1.3 Update Model Loader

Edit these files to use your Hugging Face username:
- `src/model_loader.py` (line 16)
- `scripts/download_model.py` (line 10)

Replace `YOUR-USERNAME` with your actual Hugging Face username.

---

## Step 2: Configure Environment Variables

### 2.1 Create Production `.env`

Create a `.env.production` file (DO NOT commit this):

```bash
# Application
APP_ENVIRONMENT=production
APP_SECRET_KEY=<generate-with-command-below>
APP_API_KEYS=<comma-separated-keys>

# CORS (add your Railway domain after deployment)
APP_CORS_ORIGINS=https://your-app.railway.app,https://yourdomain.com

# Database (Railway will provide this)
APP_DATABASE_URL=postgresql+asyncpg://user:pass@host:port/dbname

# Security
APP_RATE_LIMIT_ENABLED=true
APP_RATE_LIMIT_PER_MINUTE=10
APP_MAX_FILE_SIZE_MB=50
APP_MAX_BATCH_SIZE=20

# Model
APP_MODEL_PATH=models/best_model.pth
APP_MODEL_SOURCE=huggingface

# Server
APP_HOST=0.0.0.0
APP_PORT=8000
APP_LOG_LEVEL=INFO

# PostgreSQL
POSTGRES_USER=postgres
POSTGRES_PASSWORD=<strong-password>
POSTGRES_DB=predictions
```

### 2.2 Generate Secure Keys

```bash
# Generate APP_SECRET_KEY
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate API keys (do this 3+ times for different keys)
python -c "import secrets; print(secrets.token_urlsafe(24))"
```

Save these keys securely (e.g., in a password manager).

---

## Step 3: Deploy to Railway

### 3.1 Initial Setup

1. Go to [railway.app](https://railway.app) and sign in
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Railway will auto-detect the Dockerfile

### 3.2 Add PostgreSQL Database

1. In your Railway project, click "+ New"
2. Select "Database" → "PostgreSQL"
3. Railway will create a database and provide connection details
4. Copy the `DATABASE_URL` (it will look like `postgresql://user:pass@host:port/db`)

### 3.3 Configure Environment Variables

In Railway project settings → Variables:

```
APP_ENVIRONMENT=production
APP_SECRET_KEY=<your-generated-secret>
APP_API_KEYS=<key1>,<key2>,<key3>
APP_CORS_ORIGINS=https://*.railway.app
APP_DATABASE_URL=${{Postgres.DATABASE_URL}}
APP_RATE_LIMIT_ENABLED=true
APP_RATE_LIMIT_PER_MINUTE=10
APP_MAX_FILE_SIZE_MB=50
APP_MAX_BATCH_SIZE=20
APP_MODEL_SOURCE=huggingface
APP_MODEL_PATH=models/best_model.pth
APP_HOST=0.0.0.0
APP_PORT=8000
APP_LOG_LEVEL=INFO
```

**Note:** Railway automatically converts `DATABASE_URL` to the correct format. Use `${{Postgres.DATABASE_URL}}` to reference the PostgreSQL service.

### 3.4 Deploy

1. Railway will automatically deploy on git push
2. Monitor build logs in Railway dashboard
3. First deployment downloads model from Hugging Face (~2-3 minutes)

---

## Step 4: Verify Deployment

### 4.1 Health Check

```bash
curl https://your-app.railway.app/health
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

### 4.2 Test Authentication

```bash
# Without API key (should fail)
curl -X POST https://your-app.railway.app/predict \
  -F "file=@test_image.jpg"

# Expected: 401 Unauthorized

# With valid API key (should succeed)
curl -X POST https://your-app.railway.app/predict \
  -H "X-API-Key: your-api-key-here" \
  -F "file=@test_image.jpg"

# Expected: 200 OK with prediction
```

### 4.3 Test Rate Limiting

```bash
# Make 11 rapid requests
for i in {1..11}; do
  curl -X POST https://your-app.railway.app/predict \
    -H "X-API-Key: your-api-key" \
    -F "file=@test.jpg"
done

# 11th request should return 429 Rate Limit Exceeded
```

---

## Step 5: Configure Custom Domain (Optional)

### 5.1 In Railway

1. Go to project Settings → Networking
2. Click "Generate Domain" (free Railway subdomain)
3. Or add custom domain:
   - Click "Custom Domain"
   - Enter your domain (e.g., `api.yourdomain.com`)
   - Add CNAME record in your DNS provider

### 5.2 Update CORS Origins

After domain is set up, update environment variable:

```
APP_CORS_ORIGINS=https://api.yourdomain.com,https://yourdomain.com
```

---

## Step 6: Frontend Deployment

### 6.1 Build Frontend

```bash
cd frontend
npm install
npm run build
```

This creates `frontend/dist/` with static files.

### 6.2 Deploy Frontend

**Option 1: Same Railway Project**
- Copy `frontend/dist/*` to `static/` in backend
- Backend serves frontend at root path

**Option 2: Separate Static Host (Vercel/Netlify)**
- Deploy `frontend/dist/` to Vercel or Netlify
- Set API URL: `VITE_API_URL=https://your-app.railway.app`
- Update CORS origins to include frontend domain

---

## Step 7: Monitoring & Logs

### 7.1 View Logs

In Railway dashboard:
- Click on your service
- Go to "Deployments" → Select deployment → "View Logs"
- Logs are in JSON format (production mode)

### 7.2 Set Up Alerts (Optional)

Railway doesn't have built-in alerting. Consider:
- **Uptime monitoring**: UptimeRobot, Pingdom
- **Error tracking**: Sentry (add `sentry-sdk` to requirements.txt)
- **Log aggregation**: Datadog, LogDNA

---

## Step 8: Cost Optimization

### 8.1 Railway Pricing (as of 2024)

- **Starter Plan**: $5/month
  - 512MB RAM, 1GB storage
  - 100GB bandwidth
  - Sufficient for portfolio projects

- **Developer Plan**: $20/month (if needed)
  - 8GB RAM, 100GB storage

### 8.2 Reduce Costs

1. **Use Hugging Face for model**: Avoid storing 128MB in git/container
2. **Optimize container size**: Multi-stage Dockerfile reduces size
3. **Rate limiting**: Prevents abuse and excess usage
4. **Hibernate unused services**: Railway can pause after inactivity

---

## Troubleshooting

### Model Download Fails

**Symptom:** Build fails with "Model not found"

**Solutions:**
1. Verify Hugging Face repo is public or access token is set
2. Check model filename in `model_loader.py` matches uploaded file
3. Add Hugging Face token to Railway env vars:
   ```
   HUGGING_FACE_HUB_TOKEN=<your-token>
   ```

### Database Connection Errors

**Symptom:** 500 errors, logs show database connection failed

**Solutions:**
1. Verify `APP_DATABASE_URL` uses `postgresql+asyncpg://` (not `postgresql://`)
2. Check PostgreSQL service is running in Railway
3. Restart database service if needed

### CORS Errors in Frontend

**Symptom:** Browser console shows CORS blocked

**Solutions:**
1. Add frontend domain to `APP_CORS_ORIGINS`
2. Ensure no trailing slashes in origins
3. Format: `https://app.railway.app` (not `https://app.railway.app/`)

### Rate Limit Too Strict

**Symptom:** Users hit 429 too quickly

**Solutions:**
1. Increase `APP_RATE_LIMIT_PER_MINUTE` (default: 10)
2. Or disable in dev: `APP_RATE_LIMIT_ENABLED=false`
3. Consider per-user rate limits (requires user auth)

### High Memory Usage

**Symptom:** Railway shows OOM errors

**Solutions:**
1. Model loaded once on startup (not per request) ✓
2. Upgrade Railway plan (512MB → 2GB)
3. Use smaller PyTorch model variant

---

## Security Checklist

Before going live, verify:

- [ ] `APP_ENVIRONMENT=production` set
- [ ] Strong `APP_SECRET_KEY` generated
- [ ] API keys configured and documented
- [ ] CORS origins restricted (not `*`)
- [ ] Rate limiting enabled
- [ ] PostgreSQL password is strong
- [ ] Database backups enabled (Railway auto-backups)
- [ ] HTTPS enabled (Railway provides free SSL)
- [ ] Medical disclaimer visible in UI
- [ ] Error messages don't leak internal details
- [ ] Logs don't contain sensitive data

---

## Maintenance

### Update Model

1. Upload new model to Hugging Face
2. Trigger rebuild in Railway (or push to git)
3. Model auto-downloads on next deployment

### Update Code

```bash
git add .
git commit -m "Update: description"
git push
```

Railway auto-deploys on push to main branch.

### Backup Database

Railway automatically backs up PostgreSQL. To manually export:

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Connect to database
railway connect postgres

# Export data
pg_dump > backup.sql
```

---

## Next Steps

- Set up monitoring (Sentry, UptimeRobot)
- Configure custom domain
- Add CI/CD tests (GitHub Actions)
- Implement user authentication (JWT)
- Add batch prediction queueing (Celery + Redis)
- Containerize frontend separately

---

## Support

- Railway Docs: https://docs.railway.app
- Hugging Face Docs: https://huggingface.co/docs
- FastAPI Docs: https://fastapi.tiangolo.com
- Project Issues: https://github.com/your-repo/issues
