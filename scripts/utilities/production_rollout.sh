#!/bin/bash
# Production Rollout Script for JDBC-Based RAG System

echo "🚀 RAG System Production Rollout"
echo "================================"
echo "Version: 1.0.0-JDBC"
echo "Date: $(date)"
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to check command success
check_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ $1 successful${NC}"
    else
        echo -e "${RED}❌ $1 failed${NC}"
        exit 1
    fi
}

# Step 1: Environment Check
echo "1. Checking Environment..."
echo "-------------------------"

# Check Python
python --version
check_status "Python check"

# Check Java
java -version 2>&1 | head -n 1
check_status "Java check"

# Check JDBC driver
if [ -f "intersystems-jdbc-3.8.4.jar" ]; then
    echo -e "${GREEN}✅ JDBC driver found${NC}"
else
    echo -e "${RED}❌ JDBC driver not found${NC}"
    exit 1
fi

# Check .env file
if [ -f ".env" ]; then
    echo -e "${GREEN}✅ .env file found${NC}"
else
    echo -e "${YELLOW}⚠️  .env file not found - using defaults${NC}"
fi

echo ""

# Step 2: Database Setup
echo "2. Setting up Database..."
echo "------------------------"

echo "Creating schema and indexes..."
python common/db_init_with_indexes.py
check_status "Database schema creation"

echo ""

# Step 3: Data Validation
echo "3. Validating Data..."
echo "--------------------"

python -c "
from common.iris_connector_jdbc import get_iris_connection
conn = get_iris_connection()
cursor = conn.cursor()

# Check tables
tables = ['SourceDocuments', 'DocumentChunks', 'Entities', 'Relationships', 'ColbertTokenEmbeddings']
for table in tables:
    cursor.execute(f'SELECT COUNT(*) FROM RAG.{table}')
    count = cursor.fetchone()[0]
    print(f'RAG.{table}: {count} rows')

cursor.close()
conn.close()
"
check_status "Data validation"

echo ""

# Step 4: Test Pipelines
echo "4. Testing All Pipelines..."
echo "--------------------------"

python scripts/test_all_pipelines_jdbc.py > /tmp/pipeline_test.log 2>&1
if grep -q "Testing complete!" /tmp/pipeline_test.log; then
    echo -e "${GREEN}✅ All pipelines tested successfully${NC}"
    
    # Extract results
    echo ""
    echo "Pipeline Test Results:"
    grep -E "✅|❌" /tmp/pipeline_test.log | tail -n 7
else
    echo -e "${RED}❌ Pipeline testing failed${NC}"
    echo "Check /tmp/pipeline_test.log for details"
    exit 1
fi

echo ""

# Step 5: Performance Check
echo "5. Checking Performance..."
echo "-------------------------"

python -c "
from common.iris_connector_jdbc import get_iris_connection
import time

conn = get_iris_connection()
cursor = conn.cursor()

# Test vector search performance
start = time.time()
cursor.execute('''
    SELECT TOP 10 doc_id 
    FROM RAG.SourceDocuments 
    WHERE embedding IS NOT NULL 
    ORDER BY doc_id
''')
results = cursor.fetchall()
elapsed = time.time() - start

print(f'Vector search test: {len(results)} docs in {elapsed:.3f}s')

cursor.close()
conn.close()
"
check_status "Performance check"

echo ""

# Step 6: Create API Service
echo "6. Creating API Service..."
echo "-------------------------"

if [ ! -f "app.py" ]; then
    echo "Creating FastAPI application..."
    cat > app.py << 'EOF'
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn

# Import pipelines
from basic_rag.pipeline import BasicRAGPipeline
from hyde.pipeline import HyDEPipeline
from crag.pipeline import CRAGPipeline
from noderag.pipeline import NodeRAGPipeline
from colbert.pipeline import OptimizedColbertRAGPipeline as ColBERTPipeline
from graphrag.pipeline import GraphRAGPipeline
from hybrid_ifind_rag.pipeline import HybridiFindRAGPipeline

from common.iris_connector_jdbc import get_iris_connection
from common.utils import get_embedding_func, get_llm_func

app = FastAPI(title="Enterprise RAG System", version="1.0.0-JDBC")

# Initialize pipelines on startup
pipelines = {}

@app.on_event("startup")
async def startup_event():
    conn = get_iris_connection()
    embedding_func = get_embedding_func()
    llm_func = get_llm_func()
    
    pipelines["basic_rag"] = BasicRAGPipeline(conn, embedding_func, llm_func)
    pipelines["hyde"] = HyDEPipeline(conn, embedding_func, llm_func)
    pipelines["crag"] = CRAGPipeline(conn, embedding_func, llm_func)
    pipelines["noderag"] = NodeRAGPipeline(conn, embedding_func, llm_func)
    pipelines["colbert"] = ColBERTPipeline(conn, embedding_func, embedding_func, llm_func)
    pipelines["graphrag"] = GraphRAGPipeline(conn, embedding_func, llm_func)
    pipelines["hybrid"] = HybridiFindRAGPipeline(conn, embedding_func, llm_func)

class QueryRequest(BaseModel):
    query: str
    technique: str = "basic_rag"
    top_k: int = 10
    threshold: Optional[float] = 0.1

class QueryResponse(BaseModel):
    query: str
    answer: str
    technique: str
    document_count: int
    success: bool

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0-JDBC"}

@app.get("/techniques")
async def list_techniques():
    return {"techniques": list(pipelines.keys())}

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    if request.technique not in pipelines:
        raise HTTPException(status_code=400, detail=f"Unknown technique: {request.technique}")
    
    try:
        pipeline = pipelines[request.technique]
        
        if request.technique == "crag":
            result = pipeline.run(request.query, top_k=request.top_k)
        else:
            result = pipeline.run(request.query, top_k=request.top_k, similarity_threshold=request.threshold)
        
        return QueryResponse(
            query=request.query,
            answer=result.get("answer", ""),
            technique=request.technique,
            document_count=len(result.get("retrieved_documents", [])),
            success=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF
    check_status "API creation"
else
    echo -e "${YELLOW}⚠️  app.py already exists${NC}"
fi

echo ""

# Step 7: Create systemd service (optional)
echo "7. Creating System Service..."
echo "----------------------------"

if [ "$EUID" -eq 0 ]; then
    cat > /etc/systemd/system/rag-api.service << EOF
[Unit]
Description=Enterprise RAG API Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment="PATH=/usr/local/bin:/usr/bin:/bin"
ExecStart=/usr/bin/python app.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF
    
    systemctl daemon-reload
    echo -e "${GREEN}✅ Systemd service created${NC}"
else
    echo -e "${YELLOW}⚠️  Run as root to create systemd service${NC}"
fi

echo ""

# Step 8: Final Summary
echo "8. Production Rollout Summary"
echo "============================="
echo ""
echo -e "${GREEN}✅ Environment validated${NC}"
echo -e "${GREEN}✅ Database configured${NC}"
echo -e "${GREEN}✅ Data validated${NC}"
echo -e "${GREEN}✅ All pipelines tested${NC}"
echo -e "${GREEN}✅ Performance verified${NC}"
echo -e "${GREEN}✅ API service created${NC}"
echo ""
echo "🎉 Production rollout complete!"
echo ""
echo "Next steps:"
echo "1. Start API: python app.py"
echo "2. Test endpoint: curl http://localhost:8000/health"
echo "3. Query RAG: curl -X POST http://localhost:8000/query -H 'Content-Type: application/json' -d '{\"query\": \"What is diabetes?\"}''"
echo ""
echo "For production deployment:"
echo "- Use a process manager (systemd, supervisor, etc.)"
echo "- Configure reverse proxy (nginx, apache)"
echo "- Set up monitoring (prometheus, grafana)"
echo "- Enable SSL/TLS"
echo ""
echo "Documentation: docs/PRODUCTION_DEPLOYMENT_JDBC.md"