#!/bin/bash

# Docker ObjectScript Compilation Test
# Tests ObjectScript fixes in the same Docker environment as GitHub CI

set -e

echo "🐳 DOCKER OBJECTSCRIPT COMPILATION TEST"
echo "======================================"
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    log_error "Docker is not running. Please start Docker first."
    exit 1
fi

log_info "Starting IRIS container with project mounted..."

# Start IRIS container using docker-compose
docker-compose up -d

# Wait for IRIS to be ready
log_info "Waiting for IRIS to be ready..."
sleep 30

# Check if container is running
if ! docker-compose ps | grep -q "iris_db.*Up"; then
    log_error "IRIS container failed to start"
    docker-compose logs
    exit 1
fi

log_info "IRIS container is running"

# Test 1: Verify project files are accessible
log_info "Test 1: Checking project file accessibility..."

# Check if project files are mounted correctly
docker-compose exec iris_db ls -la /home/irisowner/dev/objectscript/RAG/ || {
    log_error "Project files not accessible in container"
    exit 1
}

log_info "✅ Project files accessible"

# Test 2: Test ObjectScript class compilation
log_info "Test 2: Testing ObjectScript class compilation..."

# Create a test COS script for ObjectScript compilation
cat > test_compilation.cos << 'EOF'
; Test ObjectScript compilation
; Set namespace to USER
set $namespace="USER"

; Test compilation of SourceDocumentsFixed.CLS
write "Testing SourceDocumentsFixed.CLS compilation...",!
set sc = $System.OBJ.Compile("/home/irisowner/dev/objectscript/RAG/SourceDocumentsFixed.CLS")
if sc {
    write "✅ SourceDocumentsFixed.CLS compiled successfully",!
} else {
    write "❌ SourceDocumentsFixed.CLS compilation failed: "_$System.Status.GetErrorText(sc),!
}

; Test compilation of SourceDocumentsWithIFind.CLS  
write "Testing SourceDocumentsWithIFind.CLS compilation...",!
set sc = $System.OBJ.Compile("/home/irisowner/dev/objectscript/RAG/SourceDocumentsWithIFind.CLS")
if sc {
    write "✅ SourceDocumentsWithIFind.CLS compiled successfully",!
} else {
    write "❌ SourceDocumentsWithIFind.CLS compilation failed: "_$System.Status.GetErrorText(sc),!
}

; Test ZPM module.xml loading
write "Testing module.xml ZPM loading...",!
try {
    set sc = $System.OBJ.Load("/home/irisowner/dev/module.xml")
    if sc {
        write "✅ module.xml loaded successfully",!
    } else {
        write "❌ module.xml loading failed: "_$System.Status.GetErrorText(sc),!
    }
} catch ex {
    write "❌ module.xml loading failed with exception: "_ex.DisplayString(),!
}

halt
EOF

# Copy test script to container (using correct container name from docker-compose.yml)
docker cp test_compilation.cos iris_db_rag_standalone_community:/tmp/test_compilation.cos

# Execute the test script
log_info "Executing ObjectScript compilation test..."
if docker-compose exec iris_db bash -c "iris session IRIS -U USER < /tmp/test_compilation.cos" > compilation_results.txt 2>&1; then
    log_info "✅ ObjectScript test completed"
else
    log_warn "ObjectScript test had issues, checking results..."
fi

# Display results
echo
log_info "=== COMPILATION TEST RESULTS ==="
cat compilation_results.txt

# Check for success indicators
if grep -q "✅.*compiled successfully" compilation_results.txt; then
    log_info "✅ SUCCESS: ObjectScript classes compiled successfully in Docker"
    success=true
else
    log_error "❌ FAILURE: ObjectScript compilation issues detected"
    success=false
fi

# Test 3: Check for syntax errors
log_info "Test 3: Checking for syntax errors..."
if grep -q "SYNTAX" compilation_results.txt; then
    log_error "❌ SYNTAX ERRORS found in ObjectScript"
    success=false
else
    log_info "✅ No syntax errors detected"
fi

# Test 4: Check for XML parsing errors
log_info "Test 4: Checking for XML parsing errors..."
if grep -q "XML.*not in proper format\|Tag expected" compilation_results.txt; then
    log_error "❌ XML PARSING ERRORS found"
    success=false
else
    log_info "✅ No XML parsing errors detected"
fi

# Cleanup
log_info "Cleaning up test files..."
rm -f test_compilation.cos compilation_results.txt
docker-compose exec iris_db rm -f /tmp/test_compilation.cos || true

# Stop container
log_info "Stopping IRIS container..."
docker-compose down

echo
if [ "$success" = true ]; then
    log_info "🎉 DOCKER OBJECTSCRIPT TEST: SUCCESS"
    log_info "   ObjectScript fixes are working correctly in Docker environment"
    log_info "   GitHub CI should now succeed with these fixes"
    exit 0
else
    log_error "❌ DOCKER OBJECTSCRIPT TEST: FAILURE"
    log_error "   ObjectScript fixes need additional work"
    log_error "   Do not push to GitHub until issues are resolved"
    exit 1
fi