# Quick Start Guide

Fast-track guide to get the Brain Tumor Classification app running locally and deployed.

## Local Development (5 minutes)

### 1. Setup Environment
```bash
# Clone repository (if not already)
git clone <your-repo-url>
cd neuroimaging-tumor-detector

# Create environment files
./scripts/setup-env.sh

# Optional: Set API key in .env
# Edit .env and set: API_KEY=your-secret-key
# (Leave empty for dev mode)
```

### 2. Start Backend
```bash
# Option A: Docker (recommended)
docker-compose up

# Option B: Local Python
pip install -r requirements.txt
uvicorn main:app --reload
```

Backend runs at: http://localhost:8000

### 3. Start Frontend
```bash
# In a new terminal
cd frontend
npm install
npm run dev
```

Frontend runs at: http://localhost:5173

### 4. Test
- Visit http://localhost:5173
- Upload a brain MRI scan
- View prediction results

## Production Deployment (30 minutes)

### Prerequisites
- GitHub account
- Render.com account (free)
- Vercel account (free)
- Model file ready

### Deploy Backend (15 min)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Add security and deployment config"
   git push
   ```

2. **Create Render Service**
   - Go to https://dashboard.render.com/
   - New + → Web Service
   - Connect your GitHub repo
   - Settings:
     - Environment: Docker
     - Plan: Free
   - Environment variables:
     ```
     APP_CORS_ORIGINS=https://your-app.vercel.app
     API_KEY=<generate-with-openssl-rand-base64-32>
     ```

3. **Upload Model**
   - Add persistent disk: `/app/models`
   - Upload `best_model.pth` via Render Shell

4. **Test**
   - Visit: `https://your-app.onrender.com/health`
   - Should see: `{"status": "healthy", ...}`

### Deploy Frontend (15 min)

1. **Deploy to Vercel**
   ```bash
   cd frontend
   npm install -g vercel
   vercel login
   vercel --prod
   ```

2. **Set Environment Variables** (in Vercel dashboard)
   ```
   VITE_API_URL=https://your-backend.onrender.com
   VITE_API_KEY=<same-api-key-as-render>
   ```

3. **Update Backend CORS**
   - In Render dashboard, update:
   - `APP_CORS_ORIGINS=https://your-app.vercel.app`
   - Redeploy

4. **Test**
   - Visit: `https://your-app.vercel.app`
   - Upload image and verify prediction

## Common Commands

### Backend
```bash
# Run locally
uvicorn main:app --reload

# Run with Docker
docker-compose up

# Run tests
pytest

# Check logs
tail -f *.log
```

### Frontend
```bash
# Development
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

### Docker
```bash
# Build image
docker build -t tumor-detector .

# Run container
docker run -p 8000:8000 tumor-detector

# View logs
docker logs <container-id>

# Shell into container
docker exec -it <container-id> bash
```

## Environment Variables

### Backend (.env)
```bash
API_KEY=your-secret-key
APP_CORS_ORIGINS=http://localhost:5173
LOG_LEVEL=INFO
RATE_LIMIT_PER_MINUTE=60
```

### Frontend (frontend/.env)
```bash
# Local dev (uses proxy)
VITE_API_URL=
VITE_API_KEY=

# Production
VITE_API_URL=https://your-backend.onrender.com
VITE_API_KEY=your-secret-key
```

## Troubleshooting

### "Model not loaded"
- Check `APP_MODEL_PATH` environment variable
- Verify model file exists at that path
- Check backend logs for errors

### "CORS error"
- Verify `APP_CORS_ORIGINS` includes frontend URL
- Check for typos (trailing slash matters!)
- Redeploy backend after changing CORS

### "API key required"
- Check `VITE_API_KEY` in frontend .env
- Verify it matches backend `API_KEY`
- Check browser network tab for header

### "Rate limit exceeded"
- Wait 60 seconds
- Increase `RATE_LIMIT_PER_MINUTE`
- Check logs for unusual patterns

### "Cold start" on Render
- Free tier spins down after 15 min
- First request takes 30-60 seconds
- Consider paid tier ($7/mo) to avoid

## Security Checklist

Before going to production:

- [ ] Set strong API key (32+ chars)
- [ ] Restrict CORS to your domain
- [ ] Set LOG_LEVEL=INFO (not DEBUG)
- [ ] Review rate limits
- [ ] Test all endpoints
- [ ] Monitor logs regularly

## Resources

- Full deployment guide: [DEPLOYMENT.md](DEPLOYMENT.md)
- Security features: [SECURITY.md](SECURITY.md)
- Implementation details: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)

## Support

Issues? Check:
1. Logs (Render dashboard / Vercel dashboard)
2. Environment variables
3. Model file path
4. CORS configuration

## Cost

- **Local development**: $0
- **Production (Render + Vercel)**: $0/month

Free tier limits:
- Render: 750 hours/month, sleeps after 15 min
- Vercel: 100 GB bandwidth/month

Need more? See [DEPLOYMENT.md](DEPLOYMENT.md) for paid tier info.
