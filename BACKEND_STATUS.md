# Backend Deployment Status

## Overview
The neuroimaging tumor detector backend is **fully functional** but not currently deployed in production.

## Why Backend Is Not Live

**Cost-Benefit Analysis:**
- **Render.com Free Tier:** Spins down after 15 min inactivity → 30-60s cold starts
- **Render.com Paid Tier:** $7/month minimum for always-on service
- **Model Size:** 128MB ResNet18 model + PyTorch dependencies = significant RAM usage
- **Portfolio Use Case:** Demonstrating ML engineering skills doesn't require 24/7 uptime

**Decision:** Deploy frontend to showcase UI/UX design; provide clear local setup for technical reviewers.

## What's Documented

All deployment infrastructure is **production-ready** and documented:
- ✅ Docker containerization
- ✅ Render.com configuration (`render.yaml`)
- ✅ Railway.app deployment guide
- ✅ Environment variable management
- ✅ API key authentication
- ✅ Rate limiting
- ✅ CORS configuration
- ✅ Health checks
- ✅ Structured logging

## Live Demo

**Frontend:** [Deployed on Vercel]
**Backend:** Run locally (see [Getting Started](README.md#getting-started))

## For Hiring Managers

This project demonstrates:
1. **Full-stack capability** (FastAPI + React)
2. **ML deployment knowledge** (model serving, inference optimization)
3. **Cost-conscious engineering** (strategic deployment decisions)
4. **Production best practices** (security, monitoring, documentation)

To see the full system in action, clone the repo and run locally—takes < 2 minutes to set up.
