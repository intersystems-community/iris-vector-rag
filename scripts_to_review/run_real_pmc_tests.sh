#!/bin/bash
# Run tests with real PMC documents (1000 documents)
# This script sets up the environment and runs the real PMC tests

set -e # Exit on error

# Colors for output
GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
BLUE="\033[0;36m"
RESET="\033[0m"

# Print header
echo -e "${BLUE}==================================================================${RESET}"
echo -e "${BLUE}   RUNNING ALL RAG TECHNIQUES WITH 1000 REAL PMC DOCUMENTS       ${RESET}"
echo -e "${BLUE}==================================================================${RESET}"
echo ""

# Check for PMC data directory
PMC_DIR=${PMC_DATA_DIR:-"data/pmc_oas_downloaded"}
if [ ! -d "$PMC_DIR" ]; then
    echo -e "${RED}ERROR: PMC data directory $PMC_DIR does not exist${RESET}"
    echo -e "${YELLOW}Set the PMC_DATA_DIR environment variable to point to your PMC data directory${RESET}"
    exit 1
fi

# Count XML files
XML_COUNT=$(find "$PMC_DIR" -name "*.xml" | wc -l)
echo -e "${BLUE}Found $XML_COUNT PMC XML files in $PMC_DIR${RESET}"

if [ $XML_COUNT -lt 10 ]; then
    echo -e "${RED}ERROR: Not enough PMC XML files found. Need at least 10 files.${RESET}"
    exit 1
fi

# Set environment variables
export TEST_IRIS=true 
export TEST_DOCUMENT_COUNT=1000
export USE_REAL_PMC_DATA=true
export PMC_DATA_DIR="$PMC_DIR"

# Create output directory for test results
OUTPUT_DIR="test_results"
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/real_pmc_tests_$TIMESTAMP.log"

echo -e "${YELLOW}Running tests with:${RESET}"
echo -e "  PMC directory: ${YELLOW}$PMC_DIR${RESET}"
echo -e "  Document count: ${YELLOW}1000${RESET} (or available maximum)"
echo -e "  Log file: ${YELLOW}$LOG_FILE${RESET}"
echo ""

# Run the test
echo -e "${BLUE}Starting test run...${RESET}"
echo -e "${YELLOW}This will take some time as it loads real PMC documents and runs all techniques${RESET}"
echo -e "${YELLOW}Check $LOG_FILE for detailed progress${RESET}"
echo ""

# Run pytest with proper arguments
(
    set -x  # Echo commands
    poetry run python -m pytest tests/test_real_pmc_with_1000_docs.py -v \
        --log-cli-level=INFO \
        -xvs | tee "$LOG_FILE"
) || {
    echo -e "${RED}Tests failed with exit code $?${RESET}"
    echo -e "${YELLOW}Check $LOG_FILE for details${RESET}"
    exit 1
}

# Print success message
echo ""
echo -e "${BLUE}==================================================================${RESET}"
echo -e "${GREEN}All RAG techniques successfully tested with real PMC documents!${RESET}"
echo -e "${BLUE}==================================================================${RESET}"
echo ""
echo -e "${BLUE}The following techniques were tested:${RESET}"
echo -e " - ${GREEN}Basic RAG${RESET}: Standard vector-based retrieval"
echo -e " - ${GREEN}ColBERT${RESET}: Token-level retrieval for more precise matching"
echo -e " - ${GREEN}NodeRAG${RESET}: Knowledge graph node-based retrieval"
echo -e " - ${GREEN}GraphRAG${RESET}: Full knowledge graph traversal for retrieval"
echo -e " - ${GREEN}Context Reduction${RESET}: Reducing context size for more effective LLM use"
echo ""
echo -e "${BLUE}This test used REAL PMC documents, demonstrating that all RAG techniques${RESET}"
echo -e "${BLUE}work properly with real-world medical research data.${RESET}"
echo ""
echo -e "${YELLOW}Detailed test log saved to:${RESET} $LOG_FILE"
echo ""

exit 0
