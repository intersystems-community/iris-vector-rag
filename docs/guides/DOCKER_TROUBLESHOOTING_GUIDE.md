# Docker Troubleshooting Guide for RAG Templates

This guide provides comprehensive troubleshooting steps for Docker-related issues in the RAG Templates project. The project uses InterSystems IRIS running in a Docker container with Python development on the host machine.

## Table of Contents

1. [Project Docker Architecture](#project-docker-architecture)
2. [Common Docker Issues](#common-docker-issues)
3. [IRIS-Specific Docker Issues](#iris-specific-docker-issues)
4. [Diagnostic Commands](#diagnostic-commands)
5. [Container Management](#container-management)
6. [Network and Port Issues](#network-and-port-issues)
7. [Volume and Data Persistence Issues](#volume-and-data-persistence-issues)
8. [Resource and Performance Issues](#resource-and-performance-issues)
9. [Alternative Setup Options](#alternative-setup-options)

## Project Docker Architecture

The RAG Templates project uses a hybrid architecture:
- **IRIS Database**: Runs in a Docker container using [`docker-compose.yml`](docker-compose.yml) or [`docker-compose.iris-only.yml`](docker-compose.iris-only.yml)
- **Python Application**: Runs on the host machine, connects to IRIS via JDBC
- **Data Persistence**: Uses Docker named volumes for IRIS data

### Key Files
- [`docker-compose.yml`](docker-compose.yml): Main Docker configuration
- [`docker-compose.iris-only.yml`](docker-compose.iris-only.yml): IRIS-only configuration (commonly used)
- [`.dockerignore`](.dockerignore): Files excluded from Docker context

## Common Docker Issues

### 1. Docker Daemon Not Running

**Symptoms:**
- `Cannot connect to the Docker daemon`
- `docker: command not found`
- `Not supported URL scheme http+docker`

**Solutions:**

#### Check Docker Status
```bash
# Check if Docker daemon is running
sudo systemctl status docker

# Start Docker if not running
sudo systemctl start docker

# Enable Docker to start on boot
sudo systemctl enable docker

# Verify Docker is working
docker --version
docker ps
```

#### Fix Docker Permissions
```bash
# Add your user to docker group
sudo usermod -aG docker $USER

# Apply group changes (logout/login or use newgrp)
newgrp docker

# Test Docker without sudo
docker ps
```

#### Restart Docker Service
```bash
# Restart Docker daemon
sudo systemctl restart docker

# Check Docker status
docker info
```

### 2. Docker Installation Issues

**Symptoms:**
- `docker: command not found`
- Conflicting Docker installations

**Solutions:**

#### Clean Installation (Ubuntu/Debian)
```bash
# Remove conflicting installations
sudo apt-get remove docker docker-engine docker.io containerd runc

# Install using official script
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Start and enable Docker
sudo systemctl start docker
sudo systemctl enable docker

# Test installation
docker run hello-world
```

#### macOS Installation
```bash
# Install Docker Desktop for Mac
# Download from: https://docs.docker.com/desktop/mac/install/

# Or using Homebrew
brew install --cask docker

# Start Docker Desktop application
open /Applications/Docker.app
```

### 3. Docker Compose Issues

**Symptoms:**
- `docker-compose: command not found`
- Version compatibility issues

**Solutions:**

#### Install Docker Compose
```bash
# Install Docker Compose (Linux)
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify installation
docker-compose --version
```

#### Use Docker Compose Plugin
```bash
# Modern Docker installations include compose as a plugin
docker compose --version

# Use 'docker compose' instead of 'docker-compose'
docker compose -f docker-compose.iris-only.yml up -d
```

## IRIS-Specific Docker Issues

### 1. IRIS Container Startup Failures

**Symptoms:**
- Container exits immediately
- IRIS fails to start
- License key issues

**Diagnostic Commands:**
```bash
# Check container status
docker-compose -f docker-compose.iris-only.yml ps

# View container logs
docker-compose -f docker-compose.iris-only.yml logs iris_db

# Check container health
docker inspect iris_db_rag_standalone --format='{{.State.Health.Status}}'
```

**Solutions:**

#### License Key Issues
```bash
# Ensure iris.key file exists (if using licensed version)
ls -la iris.key

# Check volume mount in docker-compose file
# Verify this line exists in docker-compose.yml:
# - ./iris.key:/usr/irissys/mgr/iris.key
```

#### Memory and Resource Issues
```bash
# Check available system resources
docker system df
free -h

# Increase Docker memory limits (Docker Desktop)
# Go to Docker Desktop > Settings > Resources > Advanced
# Increase Memory to at least 4GB for IRIS
```

#### Architecture Compatibility
```bash
# Check your system architecture
uname -m

# For ARM64 systems (Apple Silicon), ensure using ARM64 image:
# image: containers.intersystems.com/intersystems/iris-arm64:2025.1

# For x86_64 systems, use:
# image: containers.intersystems.com/intersystems/iris:2025.1
```

### 2. IRIS Connection Issues

**Symptoms:**
- Cannot connect to IRIS from Python
- Connection timeouts
- Authentication failures

**Diagnostic Commands:**
```bash
# Test IRIS connectivity
docker exec iris_db_rag_standalone iris session iris -U%SYS

# Check IRIS processes
docker exec iris_db_rag_standalone iris list

# Test network connectivity
telnet localhost 1972
telnet localhost 52773
```

**Solutions:**

#### Port Conflicts
```bash
# Check if ports are in use
netstat -tulpn | grep :1972
netstat -tulpn | grep :52773

# Kill processes using the ports
sudo lsof -ti:1972 | xargs kill -9
sudo lsof -ti:52773 | xargs kill -9

# Or modify port mappings in docker-compose.yml:
# ports:
#   - "1973:1972"   # Use different host port
#   - "52774:52773"
```

#### Password Expiration Issues
```bash
# The project handles this automatically, but if needed:
docker exec iris_db_rag_standalone iris session iris -U%SYS \
  "##class(Security.Users).UnExpireUserPasswords(\"*\")"
```

### 3. IRIS Health Check Failures

**Symptoms:**
- Container shows as unhealthy
- Health check timeouts

**Solutions:**

#### Check Health Check Configuration
```yaml
# Verify healthcheck in docker-compose.yml:
healthcheck:
  test: ["CMD", "/usr/irissys/bin/iris", "session", "iris", "-U%SYS", "##class(%SYSTEM.Process).CurrentDirectory()"]
  interval: 15s
  timeout: 10s
  retries: 5
  start_period: 60s
```

#### Manual Health Check
```bash
# Test health check command manually
docker exec iris_db_rag_standalone /usr/irissys/bin/iris session iris -U%SYS "##class(%SYSTEM.Process).CurrentDirectory()"
```

## Diagnostic Commands

### Container Status and Logs
```bash
# List all containers
docker ps -a

# Check specific container status
docker-compose -f docker-compose.iris-only.yml ps

# View container logs
docker logs iris_db_rag_standalone
docker-compose -f docker-compose.iris-only.yml logs -f

# Follow logs in real-time
docker logs -f iris_db_rag_standalone
```

### Container Inspection
```bash
# Inspect container configuration
docker inspect iris_db_rag_standalone

# Check container resource usage
docker stats iris_db_rag_standalone

# Execute commands in container
docker exec -it iris_db_rag_standalone bash
docker exec -it iris_db_rag_standalone iris session iris
```

### Network Diagnostics
```bash
# List Docker networks
docker network ls

# Inspect network configuration
docker network inspect bridge

# Test connectivity from container
docker exec iris_db_rag_standalone ping host.docker.internal
```

### Volume and Storage
```bash
# List Docker volumes
docker volume ls

# Inspect volume details
docker volume inspect iris_db_data

# Check volume usage
docker system df -v
```

## Container Management

### Starting and Stopping Containers
```bash
# Start IRIS container
docker-compose -f docker-compose.iris-only.yml up -d

# Stop IRIS container
docker-compose -f docker-compose.iris-only.yml down

# Restart IRIS container
docker-compose -f docker-compose.iris-only.yml restart

# Stop and remove containers, networks, volumes
docker-compose -f docker-compose.iris-only.yml down -v
```

### Container Cleanup
```bash
# Remove stopped containers
docker container prune

# Remove unused images
docker image prune

# Remove unused volumes
docker volume prune

# Complete system cleanup (use with caution)
docker system prune -a --volumes
```

### Rebuilding Containers
```bash
# Pull latest images
docker-compose -f docker-compose.iris-only.yml pull

# Force recreate containers
docker-compose -f docker-compose.iris-only.yml up -d --force-recreate

# Rebuild from scratch
docker-compose -f docker-compose.iris-only.yml down -v
docker-compose -f docker-compose.iris-only.yml up -d
```

## Network and Port Issues

### Port Conflicts
**Problem:** Ports 1972 or 52773 already in use

**Solutions:**
```bash
# Find processes using the ports
sudo lsof -i :1972
sudo lsof -i :52773

# Kill conflicting processes
sudo kill -9 <PID>

# Or modify docker-compose.yml to use different ports:
ports:
  - "1973:1972"   # IRIS SuperServer
  - "52774:52773" # Management Portal
```

### Network Connectivity Issues
**Problem:** Cannot connect to IRIS from host

**Solutions:**
```bash
# Check Docker network configuration
docker network inspect bridge

# Test connectivity
telnet localhost 1972

# Verify container is listening on correct ports
docker exec iris_db_rag_standalone netstat -tulpn | grep :1972
```

### Firewall Issues
```bash
# Check firewall status (Ubuntu/Debian)
sudo ufw status

# Allow Docker ports if needed
sudo ufw allow 1972
sudo ufw allow 52773

# For macOS, check System Preferences > Security & Privacy > Firewall
```

## Volume and Data Persistence Issues

### Data Loss After Container Restart
**Problem:** IRIS data not persisting between container restarts

**Solutions:**
```bash
# Verify volume configuration in docker-compose.yml:
volumes:
  - iris_db_data:/usr/irissys/mgr

# Check if volume exists
docker volume ls | grep iris_db_data

# Inspect volume
docker volume inspect iris_db_data
```

### Volume Permission Issues
```bash
# Check volume permissions
docker exec iris_db_rag_standalone ls -la /usr/irissys/mgr

# Fix permissions if needed
docker exec iris_db_rag_standalone chown -R irisowner:irisowner /usr/irissys/mgr
```

### Volume Backup and Restore
```bash
# Backup IRIS data
docker run --rm -v iris_db_data:/data -v $(pwd):/backup alpine \
  tar czf /backup/iris_backup.tar.gz -C /data .

# Restore IRIS data
docker run --rm -v iris_db_data:/data -v $(pwd):/backup alpine \
  tar xzf /backup/iris_backup.tar.gz -C /data
```

## Resource and Performance Issues

### Memory Issues
**Symptoms:**
- Container killed by OOM killer
- IRIS startup failures
- Poor performance

**Solutions:**
```bash
# Check system memory
free -h

# Check Docker memory limits
docker stats iris_db_rag_standalone

# Increase Docker memory (Docker Desktop)
# Settings > Resources > Advanced > Memory: 4GB+

# Monitor container memory usage
docker exec iris_db_rag_standalone cat /proc/meminfo
```

### CPU Issues
```bash
# Check CPU usage
docker stats iris_db_rag_standalone

# Limit CPU usage in docker-compose.yml:
deploy:
  resources:
    limits:
      cpus: '2.0'
      memory: 4G
```

### Disk Space Issues
```bash
# Check Docker disk usage
docker system df

# Clean up unused resources
docker system prune -a

# Check available disk space
df -h
```

## Alternative Setup Options

### 1. Local IRIS Installation (No Docker)

If Docker continues to fail, install IRIS directly:

```bash
# Download IRIS Community Edition
wget https://download.intersystems.com/download/iris-community-2025.1.0.225.1-lnxubuntux64.tar.gz

# Extract and install
tar -xzf iris-community-*.tar.gz
cd iris-community-*
sudo ./irisinstall

# Start IRIS
sudo iris start IRIS

# Test connection
python3 -c "
import sys
sys.path.append('.')
from common.iris_connector import get_iris_connection
conn = get_iris_connection()
print('✅ Local IRIS connection working')
conn.close()
"
```

### 2. Cloud IRIS Instance

Use InterSystems Cloud:

```bash
# Sign up at: https://cloud.intersystems.com/

# Configure connection environment variables
export IRIS_HOST="your-cloud-instance.intersystems.com"
export IRIS_PORT="443"
export IRIS_USERNAME="your-username"
export IRIS_PASSWORD="your-password"
export IRIS_NAMESPACE="USER"
```

### 3. Remote IRIS Server

```bash
# Connect to remote server
ssh user@remote-server

# Install IRIS on remote server
wget https://download.intersystems.com/download/iris-community-2025.1.0.225.1-lnxubuntux64.tar.gz
tar -xzf iris-community-*.tar.gz
sudo ./iris-community-*/irisinstall

# Configure local connection to remote IRIS
export IRIS_HOST="remote-server-ip"
export IRIS_PORT="1972"
export IRIS_USERNAME="SuperUser"
export IRIS_PASSWORD="SYS"
```

## Quick Recovery Checklist

When encountering Docker issues, follow this checklist:

### 1. Basic Docker Health Check
```bash
# Check Docker daemon
sudo systemctl status docker

# Test Docker functionality
docker run hello-world

# Check Docker Compose
docker-compose --version
```

### 2. IRIS Container Health Check
```bash
# Check container status
docker-compose -f docker-compose.iris-only.yml ps

# View recent logs
docker-compose -f docker-compose.iris-only.yml logs --tail=50

# Test IRIS connectivity
telnet localhost 1972
```

### 3. Quick Fixes
```bash
# Restart Docker daemon
sudo systemctl restart docker

# Restart IRIS container
docker-compose -f docker-compose.iris-only.yml restart

# Clean restart
docker-compose -f docker-compose.iris-only.yml down
docker-compose -f docker-compose.iris-only.yml up -d
```

### 4. Emergency Fallback
```bash
# Continue development with local IRIS
python3 tests/test_basic_rag_retrieval.py

# Or use mock connections for development
export USE_MOCK_IRIS=true
python3 tests/test_basic_rag_retrieval.py
```

## Getting Help

### Log Collection for Support
```bash
# Collect comprehensive logs
mkdir -p debug_logs
docker-compose -f docker-compose.iris-only.yml logs > debug_logs/docker_logs.txt
docker inspect iris_db_rag_standalone > debug_logs/container_inspect.json
docker system info > debug_logs/docker_info.txt
docker version > debug_logs/docker_version.txt
```

### Useful Resources
- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [InterSystems IRIS Documentation](https://docs.intersystems.com/iris20251/csp/docbook/DocBook.UI.Page.cls)
- [Project README](../README.md)
- [Deployment Guide](DEPLOYMENT_GUIDE.md)

### Common Environment Variables
```bash
# IRIS connection settings
export IRIS_HOST="localhost"
export IRIS_PORT="1972"
export IRIS_USERNAME="SuperUser"
export IRIS_PASSWORD="SYS"
export IRIS_NAMESPACE="USER"

# Docker settings
export DOCKER_HOST="unix:///var/run/docker.sock"
export COMPOSE_PROJECT_NAME="rag-templates"
```

Remember: The key is to not let Docker issues block RAG development progress. Use alternative setups when needed and return to Docker troubleshooting when time permits.