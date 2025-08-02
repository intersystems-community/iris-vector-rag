#!/bin/bash

# Script to commit all vector migration work in logical chunks
# Run this from the project root directory

set -e  # Exit on any error

echo "🚀 Starting commit process for vector migration work..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Error: Not in a git repository"
    exit 1
fi

# Check current branch
CURRENT_BRANCH=$(git branch --show-current)
echo "📋 Current branch: $CURRENT_BRANCH"

# 1. Core Vector Migration Infrastructure
print_step "1. Committing core vector migration infrastructure..."
git add scripts/migrate_sourcedocuments_native_vector.py \
        objectscript/RAG.VectorMigration.cls \
        scripts/debug_vector_data.py \
        scripts/test_direct_to_vector.py \
        scripts/compile_vector_migration_class.py \
        scripts/compile_class.cos 2>/dev/null || true

git commit -m "feat: Add comprehensive vector migration infrastructure

- Create migration script for VARCHAR to native VECTOR conversion
- Add ObjectScript utilities for vector data handling
- Include debugging and testing tools for migration process
- Preserve migration research and analysis tools" || echo "No changes to commit for step 1"

print_success "Core migration infrastructure committed"

# 2. Remote Deployment Package
print_step "2. Committing remote deployment package..."
git add REMOTE_DEPLOYMENT_GUIDE.md \
        scripts/remote_setup.sh \
        scripts/verify_native_vector_schema.py \
        scripts/system_health_check.py \
        scripts/create_performance_baseline.py \
        scripts/setup_monitoring.py \
        BRANCH_DEPLOYMENT_CHECKLIST.md \
        COMMIT_STRATEGY.md 2>/dev/null || true

git commit -m "feat: Add complete remote deployment package for native VECTOR

- Automated setup script with branch detection
- Comprehensive deployment guide with branch support
- Schema verification and health monitoring tools
- Performance baseline and monitoring infrastructure
- Branch-specific deployment checklist" || echo "No changes to commit for step 2"

print_success "Remote deployment package committed"

# 3. Migration Documentation
print_step "3. Committing migration documentation..."
git add VECTOR_MIGRATION_COMPLETE_SUMMARY.md \
        V2_TABLE_MIGRATION_SUMMARY.md \
        RAG_SYSTEM_IMPROVEMENT_PLAN.md \
        BASIC_RAG_ANALYSIS.md \
        JDBC_BENCHMARK_FINAL_RESULTS.md 2>/dev/null || true

git commit -m "docs: Add comprehensive vector migration documentation

- Complete migration analysis and decision rationale
- Fresh start approach documentation
- System improvement plans and recommendations
- Migration strategy comparison and outcomes" || echo "No changes to commit for step 3"

print_success "Migration documentation committed"

# 4. RAG Pipeline Updates
print_step "4. Committing RAG pipeline updates..."
git add basic_rag/ \
        crag/ \
        hyde/ \
        noderag/ \
        colbert/ \
        hybrid_ifind_rag/ \
        graphrag/ \
        common/db_vector_search.py \
        common/utils.py \
        common/db_init_complete.sql 2>/dev/null || true

git commit -m "feat: Update all RAG pipelines for native VECTOR compatibility

- Update all 7 RAG techniques for native VECTOR types
- Remove TO_VECTOR() calls on native VECTOR columns
- Optimize database operations for native types
- Ensure compatibility with fresh schema approach" || echo "No changes to commit for step 4"

print_success "RAG pipeline updates committed"

# 5. Benchmark and Evaluation Updates
print_step "5. Committing benchmark and evaluation updates..."
git add eval/ 2>/dev/null || true

git commit -m "feat: Update benchmarking suite for native VECTOR evaluation

- Enhanced enterprise benchmark with native VECTOR support
- Comprehensive evaluation framework updates
- Performance comparison tools and utilities
- Benchmark result preservation and analysis" || echo "No changes to commit for step 5"

print_success "Benchmark updates committed"

# 6. Testing and Validation
print_step "6. Committing testing and validation..."
git add tests/ \
        scripts/*test* \
        scripts/*performance* \
        scripts/*validation* \
        scripts/quick_* \
        scripts/inspect_* 2>/dev/null || true

git commit -m "test: Add comprehensive testing suite for native VECTOR

- Unit tests for RAG pipeline functionality
- Performance validation and testing tools
- Integration tests for vector operations
- Automated testing infrastructure" || echo "No changes to commit for step 6"

print_success "Testing suite committed"

# 7. Performance Results and Analysis
print_step "7. Committing performance results and analysis..."
git add *.json \
        *.png \
        *.html \
        *benchmark* \
        *spider* \
        *performance* \
        *validation* \
        comprehensive_* \
        rag_* 2>/dev/null || true

git commit -m "docs: Add performance analysis results and visualizations

- Comprehensive benchmark results and comparisons
- Performance visualization charts and reports
- HNSW validation and optimization results
- System performance baselines and metrics" || echo "No changes to commit for step 7"

print_success "Performance results committed"

# 8. Remaining files (catch-all)
print_step "8. Committing any remaining files..."
git add . 2>/dev/null || true

git commit -m "chore: Add remaining migration artifacts and analysis files

- Performance analysis results and visualizations
- Migration history and backup files
- Investigation and debugging artifacts
- Complete project state preservation" || echo "No changes to commit for step 8"

print_success "Remaining files committed"

# Show commit summary
print_step "Commit Summary:"
git log --oneline -10

echo ""
echo "🎉 All commits completed successfully!"
echo "📤 Ready to push to remote repository:"
echo "   git push origin $CURRENT_BRANCH"
echo ""
echo "🚀 After pushing, you can deploy to your remote server using:"
echo "   git clone <your-repo-url> rag-templates"
echo "   cd rag-templates"
echo "   git checkout $CURRENT_BRANCH"
echo "   ./scripts/remote_setup.sh"