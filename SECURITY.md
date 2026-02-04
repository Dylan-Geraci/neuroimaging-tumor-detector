# Security Features

This document describes the security features implemented in the Brain Tumor Classification API.

## Overview

The application implements multiple layers of security to protect against common vulnerabilities and ensure safe operation in production environments.

## Security Features

### 1. API Key Authentication

**Purpose**: Prevent unauthorized access to the API endpoints.

**Implementation**:
- Optional API key validation via `X-API-Key` header
- Configured via `API_KEY` environment variable
- When not set, runs in development mode (no authentication required)

**Usage**:
```bash
# Set API key
export API_KEY="your-secret-key-here"

# Make authenticated request
curl -H "X-API-Key: your-secret-key-here" https://api.example.com/predict
```

**Location**: `src/auth.py`

### 2. Rate Limiting

**Purpose**: Prevent API abuse and DoS attacks.

**Implementation**:
- Sliding window rate limiter
- Default: 60 requests per minute per IP address
- Tracks requests in-memory (consider Redis for production with multiple workers)
- Returns HTTP 429 when limit exceeded

**Configuration**:
```bash
export RATE_LIMIT_PER_MINUTE=60  # Adjust as needed
```

**Headers**:
- `Retry-After: 60` - Included in 429 responses

**Location**: `src/rate_limit.py`

### 3. Input Validation

**Purpose**: Prevent malicious file uploads and invalid requests.

**Validations**:
- **File Type**: Only allows image formats (JPEG, PNG, GIF, BMP, WebP, TIFF)
- **File Size**: Maximum 10 MB per file
- **Batch Size**: Maximum 50 files per batch request
- **File Content**: Validates both MIME type and file extension

**Error Responses**:
- `400 Bad Request`: Invalid file type/extension/size
- `413 Request Entity Too Large`: File exceeds size limit

**Location**: `src/validation.py`

### 4. Structured Logging

**Purpose**: Security monitoring and audit trail.

**Features**:
- Request logging with method, path, status code, and duration
- Error logging with full stack traces
- Configurable log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Timestamp and context for all log entries

**Configuration**:
```bash
export LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

**Location**: `src/logger.py`

### 5. CORS Protection

**Purpose**: Control which domains can access the API.

**Implementation**:
- Configurable allowed origins
- Defaults to `["*"]` for development
- Should be restricted in production

**Configuration**:
```bash
export APP_CORS_ORIGINS="https://your-frontend.com,https://www.your-frontend.com"
```

### 6. HTTPS Enforcement

**Purpose**: Encrypt data in transit.

**Implementation**:
- Automatic when deployed on Render.com/Vercel
- For local HTTPS, use reverse proxy (nginx, Caddy)

## Security Best Practices

### Production Deployment

1. **Set a Strong API Key**
   ```bash
   # Generate a secure random key
   openssl rand -base64 32
   ```

2. **Restrict CORS Origins**
   ```bash
   export APP_CORS_ORIGINS="https://your-actual-domain.com"
   ```

3. **Enable Request Logging**
   ```bash
   export LOG_LEVEL=INFO
   ```

4. **Configure Appropriate Rate Limits**
   ```bash
   export RATE_LIMIT_PER_MINUTE=60
   ```

5. **Use HTTPS Only**
   - Ensure all traffic goes through HTTPS
   - Never send API keys over HTTP

### Environment Variables

**Never commit these to git**:
- `.env` files
- API keys
- Secrets

**Use environment variables**:
- ✅ Set in hosting platform (Render.com environment variables)
- ✅ Use `.env.example` as template
- ❌ Don't hardcode secrets in code

### File Upload Security

The application implements defense-in-depth for file uploads:

1. **Content-Type Validation**: Checks MIME type
2. **Extension Validation**: Checks file extension
3. **Size Limits**: Prevents large file attacks
4. **Batch Limits**: Prevents resource exhaustion
5. **Memory Safety**: Files processed in memory with size limits

### Database Security

**SQLite Configuration**:
- Database file has restricted permissions
- No sensitive data stored (predictions only)
- Regular backups recommended

**SQL Injection Prevention**:
- SQLAlchemy ORM used (parameterized queries)
- No raw SQL execution

## Vulnerability Disclosures

### Known Limitations

1. **Rate Limiting**: In-memory rate limiter won't work across multiple workers
   - **Mitigation**: Use Redis-based rate limiter for production scale
   - **Impact**: Low for single-worker deployments (Free tier Render)

2. **Session Management**: No user sessions implemented
   - **Mitigation**: API key is sufficient for current use case
   - **Future**: Implement JWT tokens for user-specific features

3. **File Storage**: Uploaded files not persisted
   - **Impact**: Low (files only held in memory during processing)

### Reporting Security Issues

Please report security vulnerabilities via:
- GitHub Issues (for non-critical issues)
- Email: [your-email] (for critical vulnerabilities)

## Security Headers

The following security headers should be added via reverse proxy (nginx/Caddy):

```nginx
# nginx example
add_header X-Content-Type-Options "nosniff" always;
add_header X-Frame-Options "DENY" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
add_header Content-Security-Policy "default-src 'self'" always;
```

## Compliance

### GDPR Considerations

If processing medical data from EU users:
- Obtain proper consent
- Implement data retention policies
- Allow data deletion requests
- Maintain audit logs

### HIPAA Considerations

For healthcare use in the US:
- **This application is NOT HIPAA-compliant by default**
- Additional requirements:
  - Encrypt data at rest
  - Audit logging
  - Access controls
  - Business Associate Agreements (BAA)
  - PHI handling procedures

**Recommendation**: For medical use, consult with compliance experts.

## Security Checklist

Before deploying to production:

- [ ] Set strong API key
- [ ] Restrict CORS origins
- [ ] Enable request logging
- [ ] Configure rate limiting
- [ ] Use HTTPS only
- [ ] Review file upload limits
- [ ] Set up monitoring/alerts
- [ ] Regular dependency updates
- [ ] Backup database regularly
- [ ] Review access logs periodically

## Dependencies

### Security Updates

Keep dependencies updated:

```bash
# Check for security vulnerabilities
pip list --outdated

# Update requirements
pip install --upgrade -r requirements.txt
```

### Known Vulnerabilities

Check for CVEs:
- [Snyk Advisor](https://snyk.io/advisor/)
- [Safety](https://github.com/pyupio/safety)

```bash
pip install safety
safety check
```

## Monitoring

### Metrics to Monitor

1. **Failed Authentication Attempts**
   - High rate may indicate brute force attack

2. **Rate Limit Violations**
   - Unusual patterns may indicate scraping/abuse

3. **Error Rates**
   - Spike may indicate attack or system issue

4. **Request Patterns**
   - Unusual geographic distribution
   - Suspicious user agents

### Alerting

Set up alerts for:
- High error rates (>5% of requests)
- Authentication failures
- Rate limit violations
- Unusual traffic patterns

## Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [PyTorch Model Security](https://pytorch.org/blog/security/)
