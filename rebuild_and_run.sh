#!/bin/bash

# StoxChai Force Rebuild and Run Script

set -e

echo "🚀 Force rebuilding StoxChai with Podman..."

# Check if podman is installed
if ! command -v podman &> /dev/null; then
    echo "❌ Podman is not installed. Please install Podman first."
    exit 1
fi

# Stop and remove existing container
echo "🧹 Stopping and removing existing container..."
podman stop stoxchai-app 2>/dev/null || true
podman rm stoxchai-app 2>/dev/null || true

# Remove existing image to force rebuild
echo "🗑️ Removing existing image to force rebuild..."
podman rmi stoxchai:latest 2>/dev/null || true

# Clean up any dangling images
echo "🧹 Cleaning up dangling images..."
podman image prune -f 2>/dev/null || true

# Build the image with no cache
echo "📦 Building StoxChai image (no cache)..."
podman build --no-cache -t stoxchai:latest .

# Create necessary directories
echo "📁 Creating necessary directories..."
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

echo "✅ StoxChai is now running with fresh build!"
echo "🌐 Access the application at: http://localhost:8501"
echo ""
echo "📊 Container logs:"
podman logs -f stoxchai-app 