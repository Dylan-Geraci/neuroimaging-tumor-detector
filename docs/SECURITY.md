# Security Guide

This document outlines the security features implemented in the Brain Tumor Classification API and best practices for secure deployment.

---

## Security Features Overview

### ✅ Implemented Security Controls

| Feature | Status | Description |
|---------|--------|-------------|
| **API Key Authentication** | ✅ Active | Protects prediction endpoints from unauthorized access |
| **Rate Limiting** | ✅ Active | Prevents DoS attacks and API abuse |
| **Input Validation** | ✅ Active | Validates file size, type, and batch limits |
| **CORS Configuration** | ✅ Active | Restricts cross-origin requests to allowed domains |
| **Secure Error Handling** | ✅ Active | Hides internal details in production |
| **Structured Logging** | ✅ Active | JSON logs for security monitoring |
| **Environment-based Config** | ✅ Active | Secrets in env vars, not code |
| **HTTPS/TLS** | ✅ Railway | Free SSL certificates via Railway |
| **Database Security** | ✅ Active | Parameterized queries (SQLAlchemy ORM) |
| **Medical Disclaimer** | ✅ Active | Clear UI warning about limitations |

---

## Pre-Deployment Security Checklist

Use this checklist before deploying to production:

### 🔐 Environment Configuration

- [ ] **`.env` file created** from `.env.example`
- [ ] **`APP_ENVIRONMENT=production`** set in Railway
- [ ] **`APP_SECRET_KEY`** generated using cryptographically secure method
- [ ] **`APP_API_KEYS`** contains strong, unique keys (24+ characters each)
- [ ] **`.env` file added to `.gitignore`** (never committed)
- [ ] **Hardcoded secrets removed** from all code files

### 🌐 Network Security

- [ ] **CORS origins restricted** (not `["*"]`)
- [ ] **HTTPS enabled** (Railway provides free SSL)
- [ ] **Rate limiting enabled** (`APP_RATE_LIMIT_ENABLED=true`)
- [ ] **File size limits configured** (default 50MB max)
- [ ] **Batch size limits set** (default 20 files max)

### 🗄️ Database Security

- [ ] **PostgreSQL password is strong** (16+ characters, mixed case, symbols)
- [ ] **Database not publicly accessible** (Railway private networking)
- [ ] **Connection uses SSL/TLS** (enforced by Railway)
- [ ] **Backups enabled** (Railway auto-backups)

### 📝 API Security

- [ ] **All sensitive endpoints protected** with `verify_api_key`
- [ ] **/health endpoint public** (needed for monitoring)
- [ ] **/docs endpoint reviewed** (consider protecting in production)
- [ ] **Error messages sanitized** (no stack traces in production)
- [ ] **Logging configured** (JSON format for production)

### 🏥 Medical/Legal Compliance

- [ ] **Medical disclaimer visible** in UI
- [ ] **"Educational Use Only" warning** prominent
- [ ] **Terms of Service created** (optional but recommended)
- [ ] **Privacy policy reviewed** (if collecting user data)
- [ ] **Data retention policy defined**

---

## How to Generate Secure Keys

### API Secret Key

```bash
# Generate a 32-byte URL-safe secret
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Example output: `xK8d_vN2pQ7jR4mS6tY9wZ1aB3cD5eF7gH0iJ2kL4nM6`

### API Keys (for users)

```bash
# Generate strong API key (24 bytes)
python -c "import secrets; print(secrets.token_urlsafe(24))"
```

Example output: `A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6`

**Best Practices:**
- Use 24+ character keys
- Generate separate key per user/application
- Rotate keys every 90 days
- Store securely (password manager)
- Never share via email/Slack

---

## Authentication Implementation

### How API Key Auth Works

1. **Configuration**: API keys set in `APP_API_KEYS` env var (comma-separated)
2. **Request**: Client sends `X-API-Key` header with request
3. **Validation**: Backend checks if key exists in configured list
4. **Bypass**: If no keys configured, auth is disabled (dev mode)

### Protected Endpoints

| Endpoint | Method | Auth Required | Rate Limit |
|----------|--------|---------------|------------|
| `/predict` | POST | ✅ Yes | 10/min |
| `/predict/batch` | POST | ✅ Yes | 5/min |
| `/predictions` | GET | ✅ Yes | - |
| `/predictions/{id}` | GET | ❌ No | - |
| `/predictions/{id}` | DELETE | ✅ Yes | - |
| `/health` | GET | ❌ No | 60/min |
| `/docs` | GET | ❌ No | - |

### Example Usage

```bash
# Without API key (development mode)
curl http://localhost:8000/predict -F "file=@scan.jpg"

# With API key (production)
curl https://api.example.com/predict \
  -H "X-API-Key: A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6" \
  -F "file=@scan.jpg"
```

---

## Rate Limiting

### Configuration

| Parameter | Default | Production Recommended |
|-----------|---------|------------------------|
| `APP_RATE_LIMIT_ENABLED` | `false` | `true` |
| `APP_RATE_LIMIT_PER_MINUTE` | `10` | `10-60` |

### Limits by Endpoint

- **Health check**: 60/min (monitoring systems)
- **Single prediction**: 10/min (prevents abuse)
- **Batch prediction**: 5/min (resource intensive)

### Response on Limit Exceeded

```json
{
  "error": "Rate limit exceeded",
  "detail": "Too many requests. Please try again later."
}
```

HTTP Status: `429 Too Many Requests`
Header: `Retry-After: 60`

---

## Input Validation

### File Upload Validation

**Allowed file types:**
- `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.webp`

**Limits:**
- **Max file size**: 50MB (configurable via `APP_MAX_FILE_SIZE_MB`)
- **Max batch size**: 20 files (configurable via `APP_MAX_BATCH_SIZE`)

**Validation flow:**
1. Check file size before processing
2. Validate MIME type and extension
3. Reject if too large or wrong type
4. Return 400 Bad Request with clear error message

---

## Error Handling

### Development Mode

- **Detailed errors**: Full exception messages and stack traces
- **Logging**: Human-readable format
- **Purpose**: Debugging and development

### Production Mode

- **Generic errors**: "Internal server error" without details
- **Logging**: JSON format with full context
- **Purpose**: Security (don't leak internal state)

Example:

```python
# Development error response
{
  "detail": "Prediction failed: ValueError: invalid image format"
}

# Production error response
{
  "detail": "Internal server error"
}

# Both cases: Full error logged server-side
```

---

## Logging & Monitoring

### Log Levels

| Level | Development | Production |
|-------|-------------|------------|
| DEBUG | ✅ Verbose | ❌ Disabled |
| INFO | ✅ Enabled | ✅ Enabled |
| WARNING | ✅ Enabled | ✅ Enabled |
| ERROR | ✅ Enabled | ✅ Enabled |

### What Gets Logged

**✅ Safe to log:**
- Request timestamps
- Endpoint accessed
- Response status codes
- File sizes and counts
- Rate limit hits
- Model loading events

**❌ Never log:**
- API keys or secrets
- User personal data (if collected)
- Full file contents
- Database passwords

### Log Format

**Development:**
```
2024-01-15 10:30:45 - brain_tumor_api - INFO - Model loaded from models/best_model.pth
```

**Production (JSON):**
```json
{
  "asctime": "2024-01-15T10:30:45Z",
  "name": "brain_tumor_api",
  "levelname": "INFO",
  "message": "Model loaded from models/best_model.pth"
}
```

---

## CORS Configuration

### Development

```python
APP_CORS_ORIGINS=http://localhost:5173,http://localhost:8000
```

- Allows local frontend development
- Credentials enabled

### Production

```python
APP_CORS_ORIGINS=https://api.example.com,https://example.com
```

- **Specific domains only** (no wildcards)
- HTTPS required
- Credentials enabled if origins restricted

### Security Warning

The system logs a warning if production uses wildcard CORS:

```
SECURITY WARNING: CORS wildcard in production!
```

---

## Database Security

### SQLAlchemy ORM Benefits

- **Parameterized queries**: No SQL injection risk
- **Type validation**: Pydantic models validate input
- **Async support**: Uses `asyncpg` for PostgreSQL

### Connection Security

- **SSL/TLS enforced**: Railway PostgreSQL uses encrypted connections
- **Private networking**: Database not exposed to internet
- **Strong passwords**: Generated by Railway or set manually

### Data Sanitization

All user inputs validated before database insertion:
- Filename length limits
- Class names from predefined list
- Confidence scores validated as floats (0-1)

---

## Incident Response

### If API Key Compromised

1. **Immediate**: Remove key from `APP_API_KEYS`
2. **Deploy**: Push updated env vars to Railway
3. **Notify**: Inform key owner (if applicable)
4. **Investigate**: Check logs for unauthorized usage
5. **Rotate**: Generate new key for legitimate users

### If Database Breached

1. **Immediate**: Rotate database password
2. **Assess**: Review what data was exposed
3. **Backup**: Restore from pre-incident backup if needed
4. **Update**: Apply security patches
5. **Audit**: Review access logs

### If DDoS Attack

1. **Enable**: Ensure rate limiting is active
2. **Monitor**: Check Railway metrics and logs
3. **Scale**: Railway auto-scales (may increase costs)
4. **Block**: Add IP blocking via Railway/Cloudflare
5. **Investigate**: Review attack source and patterns

---

## Compliance Considerations

### HIPAA (Health Insurance Portability and Accountability Act)

⚠️ **This application is NOT HIPAA compliant by default.**

To make HIPAA compliant, you would need:
- Signed Business Associate Agreement (BAA) with Railway
- Encrypted data at rest and in transit ✅ (Railway provides)
- Audit logging ✅ (implemented)
- Access controls ✅ (API keys)
- Data retention policies ❌ (not implemented)
- Patient consent mechanisms ❌ (not implemented)

**Recommendation:** Use for research/education only, not clinical patient data.

### GDPR (General Data Protection Regulation)

Current status:
- ✅ No personal data collected (only MRI images)
- ✅ User can delete predictions (`DELETE /predictions/{id}`)
- ❌ No data export feature (could be added)
- ❌ No privacy policy

If you add user accounts, you'll need:
- Cookie consent banner
- Privacy policy
- Data export/deletion endpoints
- User consent tracking

---

## Recommended Additional Security Measures

For production deployments, consider:

### 1. Web Application Firewall (WAF)
- **Cloudflare**: Free tier available
- **Benefits**: DDoS protection, bot filtering, SSL

### 2. Monitoring & Alerting
- **Sentry**: Error tracking and alerting
- **UptimeRobot**: Uptime monitoring
- **LogDNA/Datadog**: Log aggregation

### 3. Secrets Management
- **Railway Secrets**: Built-in env var encryption
- **Doppler/Vault**: For multi-environment management

### 4. Dependency Scanning
- **Dependabot**: Auto-update vulnerable dependencies
- **Snyk**: Scan for security issues

### 5. Penetration Testing
- Run OWASP ZAP or Burp Suite
- Test authentication bypass
- Test rate limit effectiveness
- Fuzz test file upload validation

---

## Security Testing Procedures

### Manual Tests

**Test 1: Authentication Bypass**
```bash
# Try without API key (should fail in production)
curl https://api.example.com/predict -F "file=@test.jpg"
# Expected: 401 Unauthorized
```

**Test 2: Rate Limiting**
```bash
# Make 11 rapid requests
for i in {1..11}; do
  curl https://api.example.com/predict \
    -H "X-API-Key: valid-key" \
    -F "file=@test.jpg"
done
# Expected: 11th request returns 429
```

**Test 3: File Size Limit**
```bash
# Create 51MB file
dd if=/dev/zero of=large.jpg bs=1M count=51

# Upload
curl https://api.example.com/predict \
  -H "X-API-Key: valid-key" \
  -F "file=@large.jpg"
# Expected: 400 File too large
```

**Test 4: CORS Restriction**
```bash
# From disallowed origin
curl https://api.example.com/predict \
  -H "Origin: https://evil.com" \
  -H "X-API-Key: valid-key" \
  -F "file=@test.jpg"
# Expected: CORS error or missing CORS headers
```

### Automated Tests

Add to `tests/test_security.py`:

```python
def test_requires_auth():
    response = client.post("/predict", files={"file": test_image})
    assert response.status_code == 401

def test_rate_limit():
    for i in range(11):
        response = client.post("/predict",
            headers={"X-API-Key": "test-key"},
            files={"file": test_image})
    assert response.status_code == 429

def test_file_too_large():
    large_file = b"x" * (51 * 1024 * 1024)  # 51MB
    response = client.post("/predict",
        headers={"X-API-Key": "test-key"},
        files={"file": ("large.jpg", large_file)})
    assert response.status_code == 400
```

---

## Security Contacts

For security issues or vulnerabilities:
- **Email**: security@yourdomain.com
- **GitHub**: Private security advisory
- **Response time**: 48 hours

---

## Changelog

| Date | Change | Version |
|------|--------|---------|
| 2024-01-15 | Initial security implementation | 1.0.0 |
| - | API key authentication added | - |
| - | Rate limiting implemented | - |
| - | Input validation added | - |
| - | CORS hardening | - |

---

## References

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [Railway Security](https://docs.railway.app/reference/security)
- [HIPAA Compliance Guide](https://www.hhs.gov/hipaa/index.html)
- [GDPR Overview](https://gdpr.eu/)
