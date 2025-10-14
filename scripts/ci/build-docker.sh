#!/bin/bash
# Docker build script for RAG templates framework

set -euo pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
IMAGE_NAME="rag-templates"
TAG="latest"
TARGET="full"
PUSH=false
REGISTRY=""
BUILD_ARGS=""
PLATFORM="linux/amd64"
CACHE=true
BUILD_CONTEXT="${PROJECT_ROOT}"

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Docker build script for RAG templates framework

OPTIONS:
    -n, --name NAME         Image name (default: rag-templates)
    -t, --tag TAG           Image tag (default: latest)
    --target TARGET         Build target: base, api, worker, full (default: full)
    -p, --push              Push image to registry
    -r, --registry REGISTRY Registry URL (e.g., ghcr.io/owner)
    --build-arg ARG=VALUE   Build argument
    --platform PLATFORM    Target platform (default: linux/amd64)
    --no-cache              Disable build cache
    -h, --help              Show this help message

EXAMPLES:
    $0                                  # Build basic image
    $0 -t v1.0.0 -p                    # Build and push with tag
    $0 --target api -t api-latest      # Build API target
    $0 --platform linux/arm64         # Build for ARM64
    $0 --build-arg VERSION=1.0.0      # Pass build argument

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        --target)
            TARGET="$2"
            shift 2
            ;;
        -p|--push)
            PUSH=true
            shift
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        --build-arg)
            BUILD_ARGS+=" --build-arg $2"
            shift 2
            ;;
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --no-cache)
            CACHE=false
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Change to project root
cd "${PROJECT_ROOT}"

# Validate target
case $TARGET in
    base|api|worker|full)
        ;;
    *)
        echo -e "${RED}Invalid target: $TARGET${NC}"
        echo "Valid targets: base, api, worker, full"
        exit 1
        ;;
esac

# Construct full image name
FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"
if [[ -n "$REGISTRY" ]]; then
    FULL_IMAGE_NAME="${REGISTRY}/${FULL_IMAGE_NAME}"
fi

echo -e "${BLUE}RAG Templates Docker Build${NC}"
echo -e "${BLUE}=========================${NC}"
echo ""
echo "Image: ${FULL_IMAGE_NAME}"
echo "Target: ${TARGET}"
echo "Platform: ${PLATFORM}"
echo "Context: ${BUILD_CONTEXT}"
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed or not in PATH${NC}"
    exit 1
fi

# Check if Dockerfile exists
if [[ ! -f "${BUILD_CONTEXT}/Dockerfile" ]]; then
    echo -e "${RED}Dockerfile not found in ${BUILD_CONTEXT}${NC}"
    exit 1
fi

# Build Docker command
DOCKER_CMD="docker build"

# Add platform
DOCKER_CMD+=" --platform ${PLATFORM}"

# Add target
DOCKER_CMD+=" --target ${TARGET}"

# Add build args
if [[ -n "$BUILD_ARGS" ]]; then
    DOCKER_CMD+="$BUILD_ARGS"
fi

# Add cache options
if [[ "$CACHE" == false ]]; then
    DOCKER_CMD+=" --no-cache"
fi

# Add tag
DOCKER_CMD+=" -t ${FULL_IMAGE_NAME}"

# Add context
DOCKER_CMD+=" ${BUILD_CONTEXT}"

echo -e "${BLUE}Building Docker image...${NC}"
echo "Command: $DOCKER_CMD"
echo ""

# Execute build
if eval $DOCKER_CMD; then
    echo ""
    echo -e "${GREEN}✓ Docker image built successfully${NC}"
    
    # Show image info
    echo ""
    echo "Image details:"
    docker images "${IMAGE_NAME}" --format "table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.CreatedAt}}\t{{.Size}}"
    
    # Test the image
    echo ""
    echo -e "${BLUE}Testing Docker image...${NC}"
    if docker run --rm "${FULL_IMAGE_NAME}" python -c "
import sys
print(f'Python version: {sys.version}')

try:
    import iris_rag
    print('✓ iris_rag imported successfully')
except ImportError as e:
    print(f'✗ iris_rag import failed: {e}')

try:
    import mem0_integration
    print('✓ mem0_integration imported successfully')  
except ImportError as e:
    print(f'✗ mem0_integration import failed: {e}')
    
print('✓ Container test completed')
"; then
        echo -e "${GREEN}✓ Container test passed${NC}"
    else
        echo -e "${YELLOW}⚠ Container test failed, but image was built${NC}"
    fi
    
    # Push if requested
    if [[ "$PUSH" == true ]]; then
        echo ""
        echo -e "${BLUE}Pushing image to registry...${NC}"
        
        if docker push "${FULL_IMAGE_NAME}"; then
            echo -e "${GREEN}✓ Image pushed successfully${NC}"
        else
            echo -e "${RED}✗ Failed to push image${NC}"
            exit 1
        fi
    fi
    
    echo ""
    echo -e "${GREEN}Build completed successfully!${NC}"
    echo "Image: ${FULL_IMAGE_NAME}"
    
else
    echo ""
    echo -e "${RED}✗ Docker build failed${NC}"
    exit 1
fi