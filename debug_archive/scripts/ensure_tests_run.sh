#!/bin/bash
# Direct approach to make sure tests run with 1000 documents
# This script applies fixes to the database schema and runs tests

set -e  # Exit on first error

# Ensure colors work in output
RESET="\033[0m"
RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
BLUE="\033[34m"

echo -e "${BLUE}[TEST RUNNER] Making sure all GraphRAG tests run with 1000 documents${RESET}"
echo

# Make sure Docker is running
echo -e "${YELLOW}[STEP 1] Checking Docker status...${RESET}"
if ! docker info &>/dev/null; then
  echo -e "${RED}Docker is not running. Please start Docker and try again.${RESET}"
  exit 1
fi
echo -e "${GREEN}Docker is running${RESET}"

# Fix database schema in db_init.sql if needed
echo -e "${YELLOW}[STEP 2] Checking and fixing database schema if needed...${RESET}"

SCHEMA_FILE="common/db_init.sql"
if grep -q "text_content LONGVARCHAR" "$SCHEMA_FILE"; then
  echo "Fixing SourceDocuments schema in $SCHEMA_FILE"
  sed -i.bak 's/text_content LONGVARCHAR/title VARCHAR(1000),\n    content LONGVARCHAR/' "$SCHEMA_FILE"
fi

if grep -q "content CLOB" "$SCHEMA_FILE" || grep -q "content LONGVARCHAR" "$SCHEMA_FILE"; then
  echo "Fixing KnowledgeGraphNodes schema in $SCHEMA_FILE"
  sed -i.bak 's/content CLOB/description_text LONGVARCHAR/' "$SCHEMA_FILE"
  sed -i.bak 's/content LONGVARCHAR/description_text LONGVARCHAR/' "$SCHEMA_FILE"
fi

echo -e "${GREEN}Schema check complete${RESET}"

# Fix utils.py if needed
UTILS_FILE="tests/utils.py"
if grep -q "INSERT INTO KnowledgeGraphNodes (node_id, node_type, node_name, content, embedding)" "$UTILS_FILE"; then
  echo "Fixing insert statement in $UTILS_FILE"
  sed -i.bak 's/INSERT INTO KnowledgeGraphNodes (node_id, node_type, node_name, content, embedding)/INSERT INTO KnowledgeGraphNodes (node_id, node_type, node_name, description_text, embedding)/' "$UTILS_FILE"
fi

# Set environment variables correctly
echo -e "${YELLOW}[STEP 3] Setting environment variables...${RESET}"

export TEST_IRIS=true
export TEST_DOCUMENT_COUNT=1000
export COLLECT_PERFORMANCE_METRICS=true

# Use real embeddings
export USE_MOCK_EMBEDDINGS=false

echo "TEST_IRIS=$TEST_IRIS"
echo "TEST_DOCUMENT_COUNT=$TEST_DOCUMENT_COUNT"
echo "COLLECT_PERFORMANCE_METRICS=$COLLECT_PERFORMANCE_METRICS"
echo "USE_MOCK_EMBEDDINGS=$USE_MOCK_EMBEDDINGS"

# Run a basic test first to verify testcontainer setup
echo -e "${YELLOW}[STEP 4] Running basic testcontainer setup test...${RESET}"

# Run a simple test that should pass quickly
echo "Running test_iris_testcontainer_setup to verify testcontainer functions..."
poetry run pytest tests/test_graphrag_with_testcontainer.py::test_iris_testcontainer_setup -v

# If successful, run the document loading test
echo -e "${YELLOW}[STEP 5] Running document loading test...${RESET}"
echo "Running test_load_pmc_documents to verify data loading works..."
poetry run pytest tests/test_graphrag_with_testcontainer.py::test_load_pmc_documents -v

# Run the full test suite
echo -e "${YELLOW}[STEP 6] Running full GraphRAG tests...${RESET}"
echo "Running all GraphRAG tests with testcontainer and 1000 documents..."
poetry run pytest tests/test_graphrag_with_testcontainer.py -v

echo
echo -e "${GREEN}[COMPLETED] Tests have been run with 1000 documents${RESET}"
