#!/bin/bash
# Setup and Demo Real Data Pipeline
# This script demonstrates the full real data pipeline:
# 1. Initialize the database
# 2. Load PMC data
# 3. Generate embeddings
# 4. Run the RAG demo with real data

set -e  # Exit on error

echo "=================================="
echo "    REAL DATA RAG SETUP & DEMO    "
echo "=================================="
echo

# Check if IRIS environment variables are set
if [ -z "$IRIS_HOST" ]; then
    echo "WARNING: IRIS_HOST environment variable not set."
    echo "You may need to set the following environment variables for a real IRIS connection:"
    echo "- IRIS_HOST: Hostname of IRIS instance"
    echo "- IRIS_PORT: Port number (default: 1972)"
    echo "- IRIS_NAMESPACE: Namespace (default: USER)"
    echo "- IRIS_USERNAME: Username"
    echo "- IRIS_PASSWORD: Password"
    echo
    echo "Continuing with setup but will use mock connection if real connection fails."
    echo
fi

# Step 1: Initialize database and load data
echo "Step 1: Loading PMC data into IRIS..."
echo "------------------------------------"
echo "This will initialize the database schema and load PMC XML files."
echo

# Limit to a small number for demo purposes and use --mock flag
python load_pmc_data.py --init-db --limit 20 --mock

echo
echo "Step 2: Generating document embeddings..."
echo "----------------------------------------"
echo "This will generate embeddings for the documents in the database."
echo

python generate_embeddings.py --doc-level --limit 20 --mock

echo
echo "Step 3: Generating token-level embeddings for ColBERT..."
echo "-----------------------------------------------------"
echo "This will generate token-level embeddings for ColBERT retrieval."
echo

python generate_embeddings.py --token-level --limit 20 --mock

echo
echo "Step 4: Running RAG demo with mock data (since no real IRIS connection is available)..."
echo "----------------------------------------"
echo "This will demonstrate retrieval using both Basic RAG and ColBERT."
echo

# Run demo with mock flag since no real IRIS connection is available
python demo_real_data_rag.py --query "What is the role of inflammation in disease?" --top-k 3 --mock

echo
echo "=================================="
echo "     REAL DATA DEMO COMPLETE     "
echo "=================================="
