#!/bin/bash
# Script to ensure all tests run with real PMC data in a real database
# This satisfies the .clinerules requirement: "Tests must use real PMC documents, not synthetic data. At least 1000 documents should be used."

set -e  # Exit on any error

echo "=================================================="
echo "  Ensuring all RAG tests run with REAL PMC data"
echo "=================================================="

# Clean __pycache__ directories
echo "Cleaning __pycache__ directories..."
find . -type d -name "__pycache__" -exec rm -r {} +
echo "Cleaned __pycache__."

# Step 4: Run tests with real PMC data (Container startup, DB init, and data loading handled by Pytest fixture)
echo "Step 4: Running tests with real PMC data..."
./run_real_pmc_1000_tests.py

# Step 5: Verify results
echo "Step 5: Final verification of real data usage..."
./verify_real_pmc_database.py

echo "=================================================="
echo "  All RAG tests successfully run with REAL PMC data"
echo "  This satisfies the .clinerules requirement."
echo "=================================================="
