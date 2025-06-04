# Docker Troubleshooting Guide for RAG Templates

## Issue: "Not supported URL scheme http+docker"

This error indicates a Docker daemon connectivity problem. Here are multiple solutions to resolve this issue.

## Quick Docker Fixes

### 1. Check Docker Daemon Status
```bash
# Check if Docker daemon is running
sudo systemctl status docker

# If not running, start it
sudo systemctl start docker

# Enable Docker to start on boot
sudo systemctl enable docker
```

### 2. Fix Docker Socket Permissions
```bash
# Add your user to docker group
sudo usermod -aG docker $USER

# Apply group changes (logout/login or use newgrp)
newgrp docker

# Fix socket permissions
sudo chmod 666 /var/run/docker.sock
```

### 3. Restart Docker Service
```bash
# Restart Docker daemon
sudo systemctl restart docker

# Verify Docker is working
docker --version
docker ps
```

### 4. Alternative Docker Setup (if above fails)
```bash
# Remove any conflicting Docker installations
sudo apt-get remove docker docker-engine docker.io containerd runc

# Install Docker using official script
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Start Docker
sudo systemctl start docker
sudo systemctl enable docker
```

## Fallback: Manual IRIS Setup (No Docker)

If Docker continues to fail, here's how to set up IRIS manually:

### Option 1: Use Local IRIS Installation
```bash
# Download IRIS Community Edition
wget https://download.intersystems.com/download/iris-community-2025.1.0.225.1-lnxubuntux64.tar.gz

# Extract and install
tar -xzf iris-community-*.tar.gz
cd iris-community-*
sudo ./irisinstall

# Start IRIS
sudo iris start IRIS
```

### Option 2: Use TO_VECTOR() Workaround (Continue Local Development)
Since we have a working local system with TO_VECTOR() workaround, let's continue with RAG fixes:

```bash
# Continue with local development
python3 -c "
import sys
sys.path.append('.')
from common.iris_connector import get_iris_connection
conn = get_iris_connection()
print('✅ Local IRIS connection working')
conn.close()
"
```

## Quick Recovery Plan

### Immediate Action: Continue RAG Development Locally
```bash
# 1. Verify local system is working
python3 tests/test_basic_rag_retrieval.py

# 2. Continue fixing remaining RAG techniques
python3 tests/test_crag_retrieval.py
python3 tests/test_colbert_retrieval.py  
python3 tests/test_noderag_retrieval.py

# 3. Run comprehensive benchmark
python3 eval/enterprise_rag_benchmark_final.py
```

### Docker-Free IRIS Setup Script
```bash
#!/bin/bash
# Alternative setup without Docker

echo "🔧 Setting up IRIS without Docker..."

# Check if IRIS is already installed
if command -v iris &> /dev/null; then
    echo "✅ IRIS already installed"
    iris start IRIS
else
    echo "❌ IRIS not found. Please install IRIS Community Edition manually"
    echo "Download from: https://www.intersystems.com/products/intersystems-iris/try-iris/"
    exit 1
fi

# Initialize database
python3 common/db_init_with_indexes.py

# Verify setup
python3 scripts/verify_native_vector_schema.py

echo "✅ IRIS setup complete without Docker"
```

## Remote Server Alternative Setup

### Option 1: Use Cloud IRIS Instance
```bash
# Connect to InterSystems Cloud
# Sign up at: https://cloud.intersystems.com/

# Use cloud connection string in config
export IRIS_HOST="your-cloud-instance.intersystems.com"
export IRIS_PORT="443"
export IRIS_USERNAME="your-username"
export IRIS_PASSWORD="your-password"
```

### Option 2: Simplified Remote Setup
```bash
# Skip Docker, use direct IRIS installation
ssh user@remote-server

# Install IRIS directly
wget https://download.intersystems.com/download/iris-community-2025.1.0.225.1-lnxubuntux64.tar.gz
tar -xzf iris-community-*.tar.gz
sudo ./iris-community-*/irisinstall

# Clone and setup project
git clone your-repo
cd rag-templates
git checkout feature/enterprise-rag-system-complete

# Install Python dependencies
pip3 install -r requirements.txt

# Initialize without Docker
python3 common/db_init_with_indexes.py
```

## Recommended Immediate Action

**Priority 1: Continue RAG Development Locally**
```bash
# Test current working techniques
python3 tests/test_basic_rag_retrieval.py
python3 tests/test_hybrid_ifind_rag_retrieval.py  
python3 tests/test_hyde_retrieval.py

# Fix remaining techniques
python3 tests/test_crag_retrieval.py
python3 tests/test_colbert_retrieval.py
python3 tests/test_noderag_retrieval.py
```

**Priority 2: Docker Fix (Parallel)**
```bash
# Try quick Docker fix
sudo systemctl restart docker
sudo chmod 666 /var/run/docker.sock

# Test Docker
docker run hello-world
```

**Priority 3: Remote Deployment (Later)**
```bash
# Once Docker is fixed OR using manual IRIS setup
./scripts/remote_setup.sh
```

## Next Steps

1. **Immediate**: Continue RAG development locally with working system
2. **Parallel**: Fix Docker daemon issue  
3. **Later**: Deploy to remote server once Docker is resolved

The key is to not let Docker issues block RAG development progress!