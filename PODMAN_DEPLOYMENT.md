# StoxChai Podman Deployment Guide

This guide explains how to deploy StoxChai using Podman instead of Docker.

## Prerequisites

1. **Podman Installation**
   ```bash
   # On macOS
   brew install podman
   
   # On Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install podman
   
   # On CentOS/RHEL/Fedora
   sudo dnf install podman
   ```

2. **Podman Compose (Optional)**
   ```bash
   # Install podman-compose for multi-container deployment
   pip install podman-compose
   ```

## Quick Start

### Option 1: Simple Single Container Deployment

```bash
# Make the deployment script executable
chmod +x run_with_podman.sh

# Run the application
./run_with_podman.sh
```

### Option 2: Full Stack Deployment with Podman Compose

```bash
# Start all services (simplified version without nginx)
podman-compose -f podman-compose-simple.yml up -d

# Or use the full version (requires nginx config)
podman-compose -f podman-compose.yml up -d

# View logs
podman-compose -f podman-compose-simple.yml logs -f

# Stop all services
podman-compose -f podman-compose-simple.yml down
```

### Option 3: Manual Podman Commands

```bash
# Build the image
podman build -t stoxchai:latest .

# Create necessary directories
mkdir -p data logs cache

# Run the container
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

# View logs
podman logs -f stoxchai-app

# Stop and remove container
podman stop stoxchai-app
podman rm stoxchai-app
```

## Testing the Deployment

Before deploying, you can test that all imports work correctly:

```bash
# Test imports
python test_imports.py

# Test the application locally
python run_app.py
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'src'**
   - **Solution**: The new `run_app.py` script fixes this by setting the Python path correctly
   - **Verify**: Run `python test_imports.py` to confirm imports work

2. **Permission Denied Errors**
   - **Solution**: Use `:Z` suffix in volume mounts for SELinux compatibility
   - **Example**: `-v ./data:/app/data:Z`

3. **Port Already in Use**
   - **Solution**: Stop existing containers or change the port
   ```bash
   podman stop stoxchai-app
   podman rm stoxchai-app
   ```

4. **Memory Issues with AI Models**
   - **Solution**: Increase container memory limits
   ```bash
   podman run --memory=4g --memory-swap=4g ...
   ```

5. **Rootless Port Binding Error**
   - **Error**: `rootlessport cannot expose privileged port 80`
   - **Solution 1**: Use non-privileged ports (>= 1024)
   - **Solution 2**: Enable privileged ports for rootless mode
   ```bash
   # Add to /etc/sysctl.conf
   echo "net.ipv4.ip_unprivileged_port_start=80" | sudo tee -a /etc/sysctl.conf
   sudo sysctl -p
   ```
   - **Solution 3**: Use the simplified compose file without nginx

### Debugging Commands

```bash
# Check container status
podman ps -a

# View container logs
podman logs stoxchai-app

# Execute commands in running container
podman exec -it stoxchai-app /bin/bash

# Check container resource usage
podman stats stoxchai-app

# Inspect container configuration
podman inspect stoxchai-app
```

## Environment Variables

You can customize the deployment with environment variables:

```bash
# Create .env file
cat > .env << EOF
DB_PASSWORD=your_secure_password
REDIS_PASSWORD=your_redis_password
GRAFANA_PASSWORD=your_grafana_password
ENVIRONMENT=production
DEBUG=false
EOF

# Use with podman-compose
podman-compose -f podman-compose-simple.yml --env-file .env up -d
```

## Production Considerations

### Security
- Run containers as non-root user (already configured in Dockerfile)
- Use secrets management for sensitive data
- Enable SELinux if available

### Performance
- Allocate sufficient memory for AI models (4GB+ recommended)
- Use volume mounts for persistent data
- Consider using podman-compose for multi-container setups

### Monitoring
- Use the provided Prometheus and Grafana setup (optional)
- Monitor container resource usage
- Set up log aggregation

## Access Points

After successful deployment:

- **StoxChai App**: http://localhost:8501
- **Nginx (if using full stack)**: http://localhost:8080 (HTTP) / https://localhost:8443 (HTTPS)
- **Grafana Dashboard**: http://localhost:3000 (admin/admin) - *optional*
- **Prometheus**: http://localhost:9090 - *optional*
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379
- **Ollama**: http://localhost:11434

## Differences from Docker

1. **Volume Mounts**: Use `:Z` suffix for SELinux compatibility
2. **Networking**: Podman uses different network drivers
3. **Compose**: Use `podman-compose` instead of `docker-compose`
4. **Rootless**: Podman can run rootless by default
5. **Ports**: Use non-privileged ports (>= 1024) for rootless mode

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Run `python test_imports.py` to verify imports
3. Check container logs: `podman logs stoxchai-app`
4. Ensure all required files are present in the project directory

## Migration from Docker

If migrating from Docker to Podman:

1. Stop Docker containers: `docker-compose down`
2. Remove Docker images: `docker rmi stoxchai:latest`
3. Follow the Podman deployment steps above
4. Update any CI/CD pipelines to use Podman commands

## Quick Troubleshooting for Port Issues

If you get port binding errors:

```bash
# Option 1: Use simplified compose (recommended for testing)
podman-compose -f podman-compose-simple.yml up -d

# Option 2: Enable privileged ports (requires sudo)
echo "net.ipv4.ip_unprivileged_port_start=80" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p

# Option 3: Run with sudo (not recommended for production)
sudo podman-compose -f podman-compose.yml up -d
``` 