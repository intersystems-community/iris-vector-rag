#!/bin/bash
# Run tests with 92,000 PMC documents
# This script sets up the environment and runs the tests with a very large document count

set -e # Exit on error

# Colors for output
GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
BLUE="\033[0;36m"
RESET="\033[0m"

# Print header
echo -e "${BLUE}==================================================================${RESET}"
echo -e "${BLUE}   RUNNING ALL RAG TECHNIQUES WITH 92,000 PMC DOCUMENTS          ${RESET}"
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

if [ $XML_COUNT -lt 1000 ]; then
    echo -e "${RED}ERROR: Not enough PMC XML files found. Need at least 1000 files for this test.${RESET}"
    echo -e "${YELLOW}This test requires a large PMC dataset. Please download more PMC files or use a different dataset.${RESET}"
    exit 1
fi

if [ $XML_COUNT -lt 92000 ]; then
    echo -e "${YELLOW}WARNING: Found fewer than 92,000 PMC files. The test will run with all available files ($XML_COUNT).${RESET}"
    echo ""
fi

# Create output directory for test results
OUTPUT_DIR="test_results"
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/92k_docs_tests_$TIMESTAMP.log"
CHECKPOINT_FILE="$OUTPUT_DIR/92k_checkpoint_$TIMESTAMP.json"

echo -e "${YELLOW}Running large-scale tests with:${RESET}"
echo -e "  PMC directory: ${YELLOW}$PMC_DIR${RESET}"
echo -e "  Target document count: ${YELLOW}92,000${RESET} (will use all available, currently $XML_COUNT)"
echo -e "  Log file: ${YELLOW}$LOG_FILE${RESET}"
echo -e "  Checkpoint file: ${YELLOW}$CHECKPOINT_FILE${RESET}"
echo ""

# Ask for confirmation
echo -e "${RED}WARNING: This test will load a very large number of documents and may take several hours to complete.${RESET}"
echo -e "${RED}It will also use significant system resources (CPU, memory, disk space).${RESET}"
echo -e "${YELLOW}Are you sure you want to continue? (y/n)${RESET}"
read -r confirmation

if [[ ! "$confirmation" =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}Test cancelled by user.${RESET}"
    exit 0
fi

# Set environment variables
export TEST_IRIS=true 
export TEST_DOCUMENT_COUNT=92000  # Target 92,000 documents
export USE_REAL_PMC_DATA=true
export PMC_DATA_DIR="$PMC_DIR"
export CHECKPOINT_FILE="$CHECKPOINT_FILE"
export LARGE_SCALE_TEST=true
export LARGE_BATCH_SIZE=500  # Use larger batch size for efficiency
export IRIS_MEMORY="8GB"  # Allocate more memory to IRIS container

# Create test_results directory if it doesn't exist
mkdir -p test_results

# Run the large-scale test in phases

echo -e "${BLUE}Phase 1: Database Initialization and Document Loading${RESET}"
echo -e "${YELLOW}This phase will take considerable time to load all documents${RESET}"

(
    set -x  # Echo commands
    poetry run python -m pytest tests/test_real_pmc_with_1000_docs.py::test_basic_rag_with_real_pmc -v \
        --log-cli-level=INFO \
        -xvs | tee -a "$LOG_FILE"
) || {
    echo -e "${RED}Phase 1 failed with exit code $?${RESET}"
    echo -e "${YELLOW}Check $LOG_FILE for details${RESET}"
    exit 1
}

echo -e "${GREEN}Phase 1 completed successfully!${RESET}"
echo ""

echo -e "${BLUE}Phase 2: ColBERT Testing${RESET}"
(
    set -x  # Echo commands
    poetry run python -m pytest tests/test_real_pmc_with_1000_docs.py::test_colbert_with_real_pmc -v \
        --log-cli-level=INFO \
        -xvs | tee -a "$LOG_FILE"
) || {
    echo -e "${RED}Phase 2 failed with exit code $?${RESET}"
    echo -e "${YELLOW}Check $LOG_FILE for details${RESET}"
    exit 1
}

echo -e "${GREEN}Phase 2 completed successfully!${RESET}"
echo ""

echo -e "${BLUE}Phase 3: NodeRAG Testing${RESET}"
(
    set -x  # Echo commands
    poetry run python -m pytest tests/test_real_pmc_with_1000_docs.py::test_noderag_with_real_pmc -v \
        --log-cli-level=INFO \
        -xvs | tee -a "$LOG_FILE"
) || {
    echo -e "${RED}Phase 3 failed with exit code $?${RESET}"
    echo -e "${YELLOW}Check $LOG_FILE for details${RESET}"
    exit 1
}

echo -e "${GREEN}Phase 3 completed successfully!${RESET}"
echo ""

echo -e "${BLUE}Phase 4: GraphRAG Testing${RESET}"
(
    set -x  # Echo commands
    poetry run python -m pytest tests/test_real_pmc_with_1000_docs.py::test_graphrag_with_real_pmc -v \
        --log-cli-level=INFO \
        -xvs | tee -a "$LOG_FILE"
) || {
    echo -e "${RED}Phase 4 failed with exit code $?${RESET}"
    echo -e "${YELLOW}Check $LOG_FILE for details${RESET}"
    exit 1
}

echo -e "${GREEN}Phase 4 completed successfully!${RESET}"
echo ""

echo -e "${BLUE}Phase 5: Context Reduction Testing${RESET}"
(
    set -x  # Echo commands
    poetry run python -m pytest tests/test_real_pmc_with_1000_docs.py::test_context_reduction_with_real_pmc -v \
        --log-cli-level=INFO \
        -xvs | tee -a "$LOG_FILE"
) || {
    echo -e "${RED}Phase 5 failed with exit code $?${RESET}"
    echo -e "${YELLOW}Check $LOG_FILE for details${RESET}"
    exit 1
}

echo -e "${GREEN}Phase 5 completed successfully!${RESET}"
echo ""

# Print success message
echo ""
echo -e "${BLUE}==================================================================${RESET}"
echo -e "${GREEN}All RAG techniques successfully tested with large-scale document set!${RESET}"
echo -e "${BLUE}==================================================================${RESET}"
echo ""
echo -e "${BLUE}The following techniques were tested with $XML_COUNT documents:${RESET}"
echo -e " - ${GREEN}Basic RAG${RESET}: Standard vector-based retrieval"
echo -e " - ${GREEN}ColBERT${RESET}: Token-level retrieval for more precise matching"
echo -e " - ${GREEN}NodeRAG${RESET}: Knowledge graph node-based retrieval"
echo -e " - ${GREEN}GraphRAG${RESET}: Full knowledge graph traversal for retrieval"
echo -e " - ${GREEN}Context Reduction${RESET}: Reducing context size for more effective LLM use"
echo ""
echo -e "${BLUE}This test used REAL PMC documents at scale, demonstrating that all RAG${RESET}"
echo -e "${BLUE}techniques work properly with large volumes of real-world medical research data.${RESET}"
echo ""
echo -e "${YELLOW}Detailed test log saved to:${RESET} $LOG_FILE"
echo ""

exit 0
