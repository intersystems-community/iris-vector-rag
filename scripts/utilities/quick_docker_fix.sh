#!/bin/bash

# Quick Docker Fix Script
# Attempts to resolve common Docker daemon connectivity issues

echo "🔧 Quick Docker Fix Script"
echo "=========================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Step 1: Check Docker daemon status
print_status "Checking Docker daemon status..."
if sudo systemctl is-active --quiet docker; then
    print_success "Docker daemon is running"
else
    print_warning "Docker daemon is not running, attempting to start..."
    sudo systemctl start docker
    sleep 5
    if sudo systemctl is-active --quiet docker; then
        print_success "Docker daemon started successfully"
    else
        print_error "Failed to start Docker daemon"
        exit 1
    fi
fi

# Step 2: Fix Docker socket permissions
print_status "Fixing Docker socket permissions..."
sudo chmod 666 /var/run/docker.sock
print_success "Docker socket permissions fixed"

# Step 3: Add user to docker group (if not already)
print_status "Checking docker group membership..."
if groups $USER | grep -q docker; then
    print_success "User is already in docker group"
else
    print_status "Adding user to docker group..."
    sudo usermod -aG docker $USER
    print_warning "You may need to logout/login or run 'newgrp docker' for group changes to take effect"
fi

# Step 4: Test Docker functionality
print_status "Testing Docker functionality..."
if docker ps &> /dev/null; then
    print_success "Docker is working correctly"
else
    print_warning "Docker test failed, trying with sudo..."
    if sudo docker ps &> /dev/null; then
        print_warning "Docker works with sudo but not without - permission issue"
        print_status "Running: newgrp docker"
        newgrp docker
    else
        print_error "Docker is not working even with sudo"
        exit 1
    fi
fi

# Step 5: Test Docker Compose
print_status "Testing Docker Compose..."
if docker-compose --version &> /dev/null; then
    print_success "Docker Compose is working"
else
    print_error "Docker Compose is not working"
    print_status "Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    if docker-compose --version &> /dev/null; then
        print_success "Docker Compose installed successfully"
    else
        print_error "Failed to install Docker Compose"
        exit 1
    fi
fi

# Step 6: Test with hello-world
print_status "Testing with hello-world container..."
if docker run --rm hello-world &> /dev/null; then
    print_success "Docker hello-world test passed"
else
    print_error "Docker hello-world test failed"
    print_status "Attempting to restart Docker daemon..."
    sudo systemctl restart docker
    sleep 10
    if docker run --rm hello-world &> /dev/null; then
        print_success "Docker working after restart"
    else
        print_error "Docker still not working after restart"
        exit 1
    fi
fi

print_success "🎉 Docker fix completed successfully!"
print_status ""
print_status "You can now run:"
print_status "  ./scripts/remote_setup.sh"
print_status ""
print_status "Or continue with local development:"
print_status "  python3 scripts/continue_rag_development.py"