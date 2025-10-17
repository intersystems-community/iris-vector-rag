#!/bin/bash
# =============================================================================
# Code Quality Checks for RAG API
# =============================================================================
# Runs mypy, flake8, black, and isort to ensure code quality
# =============================================================================

set -e

echo "================================================================================"
echo "RAG API Code Quality Checks"
echo "================================================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

API_DIR="iris_rag/api"
TESTS_DIR="tests/unit/api tests/performance tests/load"

# =============================================================================
# Black - Code Formatting
# =============================================================================
echo ""
echo "================================================================================"
echo "Running Black (Code Formatter)"
echo "================================================================================"

if black --check "$API_DIR" $TESTS_DIR; then
    echo -e "${GREEN}✓ Black: All files are properly formatted${NC}"
else
    echo -e "${YELLOW}⚠ Black: Some files need formatting${NC}"
    echo "Run 'black $API_DIR $TESTS_DIR' to auto-format"
    BLACK_FAILED=1
fi

# =============================================================================
# isort - Import Sorting
# =============================================================================
echo ""
echo "================================================================================"
echo "Running isort (Import Sorter)"
echo "================================================================================"

if isort --check-only "$API_DIR" $TESTS_DIR; then
    echo -e "${GREEN}✓ isort: All imports are properly sorted${NC}"
else
    echo -e "${YELLOW}⚠ isort: Some imports need sorting${NC}"
    echo "Run 'isort $API_DIR $TESTS_DIR' to auto-sort"
    ISORT_FAILED=1
fi

# =============================================================================
# flake8 - Style Guide Enforcement
# =============================================================================
echo ""
echo "================================================================================"
echo "Running flake8 (Style Guide)"
echo "================================================================================"

# Run flake8 with specific configurations
if flake8 "$API_DIR" --max-line-length=100 --exclude=__pycache__,*.pyc,.git --ignore=E203,W503; then
    echo -e "${GREEN}✓ flake8: No style issues found${NC}"
else
    echo -e "${RED}✗ flake8: Style issues detected${NC}"
    FLAKE8_FAILED=1
fi

# =============================================================================
# mypy - Type Checking
# =============================================================================
echo ""
echo "================================================================================"
echo "Running mypy (Type Checker)"
echo "================================================================================"

if mypy "$API_DIR" --ignore-missing-imports --disallow-untyped-defs --no-implicit-optional; then
    echo -e "${GREEN}✓ mypy: No type errors found${NC}"
else
    echo -e "${YELLOW}⚠ mypy: Type errors detected${NC}"
    MYPY_FAILED=1
fi

# =============================================================================
# pylint - Code Analysis
# =============================================================================
echo ""
echo "================================================================================"
echo "Running pylint (Code Analyzer)"
echo "================================================================================"

if pylint "$API_DIR" --disable=C0111,R0903,R0913,W0511 --max-line-length=100; then
    echo -e "${GREEN}✓ pylint: No issues found${NC}"
else
    echo -e "${YELLOW}⚠ pylint: Some issues detected${NC}"
    PYLINT_FAILED=1
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "================================================================================"
echo "Code Quality Check Summary"
echo "================================================================================"

TOTAL_CHECKS=5
PASSED_CHECKS=0

[[ -z "$BLACK_FAILED" ]] && ((PASSED_CHECKS++)) || echo -e "${RED}✗ Black failed${NC}"
[[ -z "$ISORT_FAILED" ]] && ((PASSED_CHECKS++)) || echo -e "${RED}✗ isort failed${NC}"
[[ -z "$FLAKE8_FAILED" ]] && ((PASSED_CHECKS++)) || echo -e "${RED}✗ flake8 failed${NC}"
[[ -z "$MYPY_FAILED" ]] && ((PASSED_CHECKS++)) || echo -e "${YELLOW}⚠ mypy had warnings${NC}"
[[ -z "$PYLINT_FAILED" ]] && ((PASSED_CHECKS++)) || echo -e "${YELLOW}⚠ pylint had warnings${NC}"

echo ""
echo "Results: $PASSED_CHECKS/$TOTAL_CHECKS checks passed"

if [[ $PASSED_CHECKS -eq $TOTAL_CHECKS ]]; then
    echo -e "${GREEN}✓ All code quality checks passed!${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠ Some checks failed or have warnings${NC}"
    exit 1
fi
