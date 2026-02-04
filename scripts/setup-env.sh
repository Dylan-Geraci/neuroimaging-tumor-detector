#!/bin/bash
# Setup environment files for local development

set -e

echo "Setting up environment files..."

# Backend .env
if [ ! -f .env ]; then
    echo "Creating backend .env file..."
    cp .env.example .env
    echo "✓ Created .env from .env.example"
    echo "  Please edit .env and set your API_KEY"
else
    echo "✓ Backend .env already exists"
fi

# Frontend .env
if [ ! -f frontend/.env ]; then
    echo "Creating frontend .env file..."
    cp frontend/.env.example frontend/.env
    echo "✓ Created frontend/.env from frontend/.env.example"
    echo "  For local development, the defaults should work (uses proxy)"
else
    echo "✓ Frontend .env already exists"
fi

echo ""
echo "Environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env and set a secure API_KEY (or leave empty for dev mode)"
echo "2. Run 'docker-compose up' or 'uvicorn main:app' to start backend"
echo "3. Run 'cd frontend && npm install && npm run dev' to start frontend"
