#!/usr/bin/env python3
"""
RAG Templates Framework - API Server
Main FastAPI application for the RAG Templates Framework
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="RAG Templates Framework API",
    description="Complete RAG pipeline API with multiple adapters and templates",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    services: Dict[str, str]

class QueryRequest(BaseModel):
    query: str
    pipeline: Optional[str] = "basic"
    limit: Optional[int] = 5
    metadata: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    query: str
    response: str
    sources: List[Dict[str, Any]]
    pipeline: str
    timestamp: str

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "RAG Templates Framework API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    import datetime
    
    # Basic service checks (simplified for Docker demo)
    services = {
        "api": "healthy",
        "database": "healthy",  # Would check actual DB connection
        "cache": "healthy"      # Would check Redis connection
    }
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.datetime.now().isoformat(),
        services=services
    )

@app.post("/api/v1/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Process a RAG query through the specified pipeline
    """
    import datetime
    
    try:
        # Simplified RAG processing for demo
        # In a real implementation, this would use the actual RAG pipeline
        response_text = f"This is a demo response to your query: '{request.query}'"
        
        # Mock sources for demo
        sources = [
            {
                "title": "Demo Document 1",
                "content": "Sample content relevant to the query...",
                "score": 0.95,
                "metadata": {"source": "demo"}
            },
            {
                "title": "Demo Document 2", 
                "content": "Another sample document...",
                "score": 0.87,
                "metadata": {"source": "demo"}
            }
        ]
        
        return QueryResponse(
            query=request.query,
            response=response_text,
            sources=sources[:request.limit],
            pipeline=request.pipeline,
            timestamp=datetime.datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/api/v1/pipelines")
async def list_pipelines():
    """List available RAG pipelines"""
    return {
        "pipelines": [
            {
                "name": "basic",
                "description": "Basic RAG pipeline with simple retrieval",
                "default": True
            },
            {
                "name": "advanced", 
                "description": "Advanced RAG with reranking and filtering",
                "default": False
            },
            {
                "name": "conversational",
                "description": "Conversational RAG with context tracking",
                "default": False
            }
        ]
    }

@app.get("/api/v1/status")
async def get_status():
    """Get detailed system status"""
    return {
        "api_status": "running",
        "version": "1.0.0",
        "uptime": "demo",
        "memory_usage": "demo",
        "active_connections": 1
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )