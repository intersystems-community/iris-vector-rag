#!/bin/bash
# Run all RAG techniques with 1000 documents
# This script automates the process of running all the tests with 1000 documents

set -e # Exit on error

# Colors for output
GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
BLUE="\033[0;36m"
RESET="\033[0m"

# Print header
echo -e "${BLUE}==================================================================${RESET}"
echo -e "${BLUE}   RUNNING ALL RAG TECHNIQUES WITH 1000 DOCUMENTS                ${RESET}"
echo -e "${BLUE}==================================================================${RESET}"
echo ""

# Function to run a specific test
run_test() {
    local test_name=$1
    echo -e "${YELLOW}Running test: ${test_name}${RESET}"
    poetry run python run_pytest_with_1000_docs.py --module test_all_techniques_1000_docs --test-name "$test_name"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Test $test_name PASSED!${RESET}"
        return 0
    else
        echo -e "${RED}❌ Test $test_name FAILED!${RESET}"
        return 1
    fi
}

# Overall test status
OVERALL_STATUS=0

# Run database and document loading test first
echo -e "${BLUE}Phase 1: Database Connection and Document Loading${RESET}"
run_test "test_db_connection"
[ $? -ne 0 ] && OVERALL_STATUS=1

run_test "test_load_1000_documents"
[ $? -ne 0 ] && OVERALL_STATUS=1

# Run individual RAG techniques
echo -e "${BLUE}Phase 2: Basic RAG Testing${RESET}"
run_test "test_basic_rag_with_1000_docs"
[ $? -ne 0 ] && OVERALL_STATUS=1

echo -e "${BLUE}Phase 3: ColBERT Testing${RESET}"
run_test "test_colbert_with_1000_docs"
[ $? -ne 0 ] && OVERALL_STATUS=1

echo -e "${BLUE}Phase 4: NodeRAG Testing${RESET}"
run_test "test_noderag_with_1000_docs"
[ $? -ne 0 ] && OVERALL_STATUS=1

echo -e "${BLUE}Phase 5: GraphRAG Testing${RESET}"
run_test "test_graphrag_with_1000_docs"
[ $? -ne 0 ] && OVERALL_STATUS=1

echo -e "${BLUE}Phase 6: Context Reduction Testing${RESET}"
run_test "test_context_reduction_with_1000_docs"
[ $? -ne 0 ] && OVERALL_STATUS=1

# Print final summary
echo ""
echo -e "${BLUE}==================================================================${RESET}"
echo -e "${BLUE}   TEST SUMMARY                                                  ${RESET}"
echo -e "${BLUE}==================================================================${RESET}"

if [ $OVERALL_STATUS -eq 0 ]; then
    echo -e "${GREEN}All RAG techniques successfully tested with 1000 documents!${RESET}"
    echo ""
    echo -e "These tests demonstrate that the following RAG techniques work with 1000 documents:"
    echo -e " - Basic RAG: Standard vector-based retrieval"
    echo -e " - ColBERT: Token-level retrieval for more precise matching"
    echo -e " - NodeRAG: Knowledge graph node-based retrieval"
    echo -e " - GraphRAG: Full knowledge graph traversal for retrieval"
    echo -e " - Context Reduction: Reducing context size for more effective LLM use"
else
    echo -e "${RED}Some tests failed. Please check the output above for details.${RESET}"
    echo -e "${YELLOW}You can run individual tests with:${RESET}"
    echo -e "poetry run python run_pytest_with_1000_docs.py --module test_all_techniques_1000_docs --test-name TEST_NAME"
fi

echo ""
echo -e "${BLUE}==================================================================${RESET}"

exit $OVERALL_STATUS
