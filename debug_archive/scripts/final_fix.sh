#!/bin/bash
# Final approach to fix the testcontainer tests
# This script simplifies everything to focus just on getting tests running

# Define colors
GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
RESET="\033[0m"

echo -e "${GREEN}Final approach to fix RAG tests with 1000 documents${RESET}"
echo "This script will fix database schema issues and run a minimal test"

# Fix database schema issues
echo -e "${YELLOW}Checking database schema...${RESET}"

# Make sure SourceDocuments has correct schema
echo "Fixing SourceDocuments table if needed..."
if grep -q "text_content LONGVARCHAR" common/db_init.sql; then
    sed -i.bak 's/text_content LONGVARCHAR/title VARCHAR(1000),\n    content LONGVARCHAR/' common/db_init.sql
    echo "Fixed SourceDocuments schema"
fi

# Make sure KnowledgeGraphNodes has correct schema
echo "Fixing KnowledgeGraphNodes table if needed..."
if grep -q "content CLOB" common/db_init.sql || grep -q "content LONGVARCHAR" common/db_init.sql; then
    sed -i.bak 's/content CLOB/description_text LONGVARCHAR/' common/db_init.sql
    sed -i.bak 's/content LONGVARCHAR/description_text LONGVARCHAR/' common/db_init.sql
    echo "Fixed KnowledgeGraphNodes schema"
fi

# Make sure utils.py has correct SQL
echo "Fixing SQL in tests/utils.py if needed..."
if grep -q "INSERT INTO KnowledgeGraphNodes (node_id, node_type, node_name, content, embedding)" tests/utils.py; then
    sed -i.bak 's/INSERT INTO KnowledgeGraphNodes (node_id, node_type, node_name, content, embedding)/INSERT INTO KnowledgeGraphNodes (node_id, node_type, node_name, description_text, embedding)/' tests/utils.py
    echo "Fixed tests/utils.py"
fi

# Create a simple test file
echo -e "${YELLOW}Creating minimal test file...${RESET}"
cat > test_minimal.py << 'EOF'
"""
Minimal test file to verify testcontainer setup.
"""
import pytest
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variables directly in the test file
os.environ["TEST_IRIS"] = "true"
os.environ["TEST_DOCUMENT_COUNT"] = "1000"

@pytest.mark.force_testcontainer
def test_minimal_testcontainer(iris_testcontainer_connection):
    """Very basic test to verify testcontainer is working."""
    assert iris_testcontainer_connection is not None
    
    # Test a simple query
    with iris_testcontainer_connection.cursor() as cursor:
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        assert result is not None
        assert result[0] == 1
        
        # Create a simple table and insert data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS TestTable (
                id VARCHAR(255) PRIMARY KEY,
                name VARCHAR(100)
            )
        """)
        
        cursor.execute("INSERT INTO TestTable VALUES ('test1', 'Test Name')")
        
        # Verify data was inserted
        cursor.execute("SELECT * FROM TestTable")
        rows = cursor.fetchall()
        assert len(rows) > 0
        logger.info(f"Retrieved row: {rows[0]}")
EOF

# Run the minimal test
echo -e "${YELLOW}Running minimal test with poetry...${RESET}"
export TEST_IRIS=true
export TEST_DOCUMENT_COUNT=1000

poetry run pytest test_minimal.py -v

# If that succeeds, try running a small subset of the real tests
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Minimal test succeeded! Now trying GraphRAG setup test...${RESET}"
    poetry run pytest tests/test_graphrag_with_testcontainer.py::test_iris_testcontainer_setup -v
    
    # If the testcontainer setup test passes, try loading docs
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Testcontainer setup test passed! Now trying document loading...${RESET}"
        poetry run pytest tests/test_graphrag_with_testcontainer.py::test_load_pmc_documents -v
    fi
fi

echo -e "${GREEN}Script completed. Check output above for results.${RESET}"
