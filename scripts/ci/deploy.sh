#!/bin/bash
# Deployment script for RAG templates framework

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
ENVIRONMENT="staging"
DRY_RUN=false
FORCE=false
VERSION=""
CONFIG_FILE=""
HEALTH_CHECK=true
ROLLBACK_ON_FAILURE=true

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deployment script for RAG templates framework

OPTIONS:
    -e, --environment ENV   Target environment: staging, production (default: staging)
    -v, --version VERSION   Version to deploy (default: latest)
    -c, --config FILE       Configuration file path
    -d, --dry-run          Show what would be deployed without executing
    -f, --force            Force deployment even if health checks fail
    --no-health-check      Skip health checks
    --no-rollback          Don't rollback on failure
    -h, --help             Show this help message

EXAMPLES:
    $0                                  # Deploy latest to staging
    $0 -e production -v v1.2.3         # Deploy specific version to production
    $0 -d                              # Dry run to see what would be deployed
    $0 -f --no-rollback                # Force deploy without rollback

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        --no-health-check)
            HEALTH_CHECK=false
            shift
            ;;
        --no-rollback)
            ROLLBACK_ON_FAILURE=false
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

# Validate environment
case $ENVIRONMENT in
    staging|production)
        ;;
    *)
        echo -e "${RED}Invalid environment: $ENVIRONMENT${NC}"
        echo "Valid environments: staging, production"
        exit 1
        ;;
esac

# Set default config file if not provided
if [[ -z "$CONFIG_FILE" ]]; then
    CONFIG_FILE="config/${ENVIRONMENT}.yml"
fi

# Set default version if not provided
if [[ -z "$VERSION" ]]; then
    if [[ "$ENVIRONMENT" == "production" ]]; then
        # For production, use latest git tag
        VERSION=$(git describe --tags --abbrev=0 2>/dev/null || echo "latest")
    else
        VERSION="latest"
    fi
fi

echo -e "${BLUE}RAG Templates Deployment${NC}"
echo -e "${BLUE}=======================${NC}"
echo ""
echo "Environment: ${ENVIRONMENT}"
echo "Version: ${VERSION}"
echo "Config: ${CONFIG_FILE}"
echo "Dry run: ${DRY_RUN}"
echo ""

# Check if config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo -e "${RED}Configuration file not found: $CONFIG_FILE${NC}"
    exit 1
fi

# Pre-deployment checks
echo -e "${BLUE}Running pre-deployment checks...${NC}"

# Check Git status for production
if [[ "$ENVIRONMENT" == "production" ]] && [[ "$FORCE" == false ]]; then
    if [[ -n "$(git status --porcelain)" ]]; then
        echo -e "${RED}Working directory is not clean. Commit or stash changes before production deployment.${NC}"
        exit 1
    fi
    
    if [[ "$(git rev-parse --abbrev-ref HEAD)" != "main" ]]; then
        echo -e "${YELLOW}Warning: Not on main branch for production deployment${NC}"
        if [[ "$DRY_RUN" == false ]]; then
            read -p "Continue? (y/N) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
    fi
fi

# Function to run command or show what would be run
run_command() {
    local cmd="$1"
    local description="$2"
    
    echo -e "${BLUE}${description}${NC}"
    echo "Command: $cmd"
    
    if [[ "$DRY_RUN" == true ]]; then
        echo -e "${YELLOW}[DRY RUN] Would execute: $cmd${NC}"
        return 0
    else
        if eval "$cmd"; then
            echo -e "${GREEN}✓ ${description} completed${NC}"
            return 0
        else
            echo -e "${RED}✗ ${description} failed${NC}"
            return 1
        fi
    fi
}

# Function to check service health
check_health() {
    local service_url="$1"
    local max_attempts=30
    local attempt=0
    
    echo -e "${BLUE}Checking service health at ${service_url}${NC}"
    
    while [[ $attempt -lt $max_attempts ]]; do
        if curl -f -s "${service_url}/health" >/dev/null 2>&1; then
            echo -e "${GREEN}✓ Service is healthy${NC}"
            return 0
        fi
        
        echo "Attempt $((attempt + 1))/${max_attempts}: Service not ready, waiting..."
        sleep 10
        ((attempt++))
    done
    
    echo -e "${RED}✗ Service health check failed after ${max_attempts} attempts${NC}"
    return 1
}

# Deployment steps
echo -e "${BLUE}Starting deployment...${NC}"

# Step 1: Build and push Docker image
if [[ "$VERSION" != "latest" ]]; then
    run_command \
        "./scripts/ci/build-docker.sh -t ${VERSION} -p" \
        "Building and pushing Docker image"
fi

# Step 2: Deploy to target environment
case $ENVIRONMENT in
    staging)
        SERVICE_URL="https://staging.rag-templates.dev"
        run_command \
            "docker-compose -f docker-compose.staging.yml up -d" \
            "Deploying to staging environment"
        ;;
    production)
        SERVICE_URL="https://rag-templates.dev"
        run_command \
            "docker-compose -f docker-compose.production.yml up -d" \
            "Deploying to production environment"
        ;;
esac

# Step 3: Health check
if [[ "$HEALTH_CHECK" == true ]] && [[ "$DRY_RUN" == false ]]; then
    if ! check_health "$SERVICE_URL"; then
        if [[ "$ROLLBACK_ON_FAILURE" == true ]]; then
            echo -e "${YELLOW}Rolling back deployment...${NC}"
            case $ENVIRONMENT in
                staging)
                    docker-compose -f docker-compose.staging.yml down
                    ;;
                production)
                    docker-compose -f docker-compose.production.yml down
                    ;;
            esac
        fi
        exit 1
    fi
fi

# Step 4: Run smoke tests
echo -e "${BLUE}Running smoke tests...${NC}"
run_command \
    "pytest tests/e2e/test_smoke.py -v" \
    "Running smoke tests"

# Step 5: Update monitoring and alerts
if [[ "$ENVIRONMENT" == "production" ]]; then
    run_command \
        "curl -X POST ${MONITORING_WEBHOOK} -d '{\"version\": \"${VERSION}\", \"environment\": \"${ENVIRONMENT}\"}'" \
        "Updating monitoring dashboard"
fi

echo ""
if [[ "$DRY_RUN" == true ]]; then
    echo -e "${BLUE}Dry run completed successfully!${NC}"
    echo "No actual changes were made."
else
    echo -e "${GREEN}Deployment completed successfully!${NC}"
    echo "Environment: ${ENVIRONMENT}"
    echo "Version: ${VERSION}"
    echo "Service URL: ${SERVICE_URL}"
fi