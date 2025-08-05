#!/bin/bash

# StoxChai Podman Deployment Script

set -e

echo "🚀 Starting StoxChai with Podman..."

# Check if podman is installed
if ! command -v podman &> /dev/null; then
    echo "❌ Podman is not installed. Please install Podman first."
    exit 1
fi

# Build the image
echo "📦 Building StoxChai image..."
podman build -t stoxchai:latest .

# Stop and remove existing container if it exists
echo "🧹 Cleaning up existing container..."
podman stop stoxchai-app 2>/dev/null || true
podman rm stoxchai-app 2>/dev/null || true

# Create necessary directories
mkdir -p data logs cache

# Run the container
echo "🏃 Starting StoxChai container..."
podman run -d \
    --name stoxchai-app \
    --restart unless-stopped \
    -p 8501:8501 \
    -v ./data:/app/data:Z \
    -v ./logs:/app/logs:Z \
    -v ./cache:/app/cache:Z \
    -e ENVIRONMENT=production \
    -e DEBUG=false \
    stoxchai:latest

echo "✅ StoxChai is now running!"
echo "🌐 Access the application at: http://localhost:8501"
echo ""
echo "📊 Container logs:"
podman logs -f stoxchai-app 