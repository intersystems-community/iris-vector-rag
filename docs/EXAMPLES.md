# Comprehensive Examples

Real-world examples demonstrating the Library Consumption Framework across different use cases and complexity levels.

## Table of Contents

1. [Quick Start Examples](#quick-start-examples)
2. [Simple API Examples](#simple-api-examples)
3. [Standard API Examples](#standard-api-examples)
4. [Enterprise API Examples](#enterprise-api-examples)
5. [MCP Integration Examples](#mcp-integration-examples)
6. [Real-World Applications](#real-world-applications)
7. [Performance Optimization Examples](#performance-optimization-examples)

## Quick Start Examples

### 30-Second RAG Application

#### Python
```python
from rag_templates import RAG

# Dead simple - works immediately
rag = RAG()
rag.add_documents([
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing enables computers to understand text."
])

answer = rag.query("What is machine learning?")
print(answer)
# Output: "Machine learning is a subset of artificial intelligence..."
```

#### JavaScript
```javascript
import { RAG } from '@rag-templates/core';

// Dead simple - works immediately
const rag = new RAG();
await rag.addDocuments([
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing enables computers to understand text."
]);

const answer = await rag.query("What is machine learning?");
console.log(answer);
// Output: "Machine learning is a subset of artificial intelligence..."
```

### 5-Minute Document Q&A System

#### Python
```python
from rag_templates import RAG
import os

# Initialize RAG
rag = RAG()

# Load documents from a directory
documents = []
for filename in os.listdir("./documents"):
    if filename.endswith('.txt'):
        with open(f"./documents/{filename}", 'r') as f:
            content = f.read()
            documents.append({
                "content": content,
                "title": filename,
                "source": filename
            })

rag.add_documents(documents)

# Interactive Q&A
while True:
    question = input("Ask a question (or 'quit' to exit): ")
    if question.lower() == 'quit':
        break
    
    answer = rag.query(question)
    print(f"Answer: {answer}\n")
```

#### JavaScript
```javascript
import { RAG } from '@rag-templates/core';
import fs from 'fs/promises';
import path from 'path';
import readline from 'readline';

// Initialize RAG
const rag = new RAG();

// Load documents from a directory
const documentsDir = "./documents";
const files = await fs.readdir(documentsDir);
const documents = [];

for (const filename of files) {
    if (filename.endsWith('.txt')) {
        const content = await fs.readFile(path.join(documentsDir, filename), 'utf8');
        documents.push({
            content: content,
            title: filename,
            source: filename
        });
    }
}

await rag.addDocuments(documents);

// Interactive Q&A
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

const askQuestion = () => {
    rl.question("Ask a question (or 'quit' to exit): ", async (question) => {
        if (question.toLowerCase() === 'quit') {
            rl.close();
            return;
        }
        
        const answer = await rag.query(question);
        console.log(`Answer: ${answer}\n`);
        askQuestion();
    });
};

askQuestion();
```

## Simple API Examples

### Basic Document Management

#### Python
```python
from rag_templates import RAG

# Initialize with zero configuration
rag = RAG()

# Add different types of documents
documents = [
    # Simple string
    "Python is a programming language.",
    
    # Document with metadata
    {
        "content": "JavaScript is used for web development.",
        "title": "JavaScript Overview",
        "source": "web_dev_guide.pdf",
        "metadata": {"category": "programming", "difficulty": "beginner"}
    },
    
    # Document with custom fields
    {
        "content": "Machine learning algorithms learn from data.",
        "title": "ML Basics",
        "author": "Dr. Smith",
        "publication_date": "2024-01-15"
    }
]

rag.add_documents(documents)

# Query the system
questions = [
    "What is Python?",
    "How is JavaScript used?",
    "What do ML algorithms do?"
]

for question in questions:
    answer = rag.query(question)
    print(f"Q: {question}")
    print(f"A: {answer}\n")

# Check system status
print(f"Total documents: {rag.get_document_count()}")
print(f"Database host: {rag.get_config('database.iris.host')}")
```

### File Processing Pipeline

#### Python
```python
from rag_templates import RAG
import os
import json

def process_knowledge_base(directory_path):
    """Process a directory of documents into a RAG knowledge base."""
    
    rag = RAG()
    processed_files = []
    
    # Supported file types
    supported_extensions = ['.txt', '.md', '.json']
    
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            
            if file_ext in supported_extensions:
                try:
                    if file_ext == '.json':
                        # Handle JSON files
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                # Array of documents
                                for i, item in enumerate(data):
                                    if isinstance(item, dict) and 'content' in item:
                                        rag.add_documents([item])
                                    elif isinstance(item, str):
                                        rag.add_documents([{
                                            "content": item,
                                            "source": f"{file}[{i}]"
                                        }])
                            elif isinstance(data, dict) and 'content' in data:
                                # Single document
                                rag.add_documents([data])
                    else:
                        # Handle text files
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            rag.add_documents([{
                                "content": content,
                                "title": os.path.basename(file),
                                "source": file_path,
                                "metadata": {
                                    "file_type": file_ext,
                                    "file_size": os.path.getsize(file_path)
                                }
                            }])
                    
                    processed_files.append(file_path)
                    print(f"✅ Processed: {file_path}")
                    
                except Exception as e:
                    print(f"❌ Error processing {file_path}: {e}")
    
    print(f"\n📊 Processing complete:")
    print(f"   Files processed: {len(processed_files)}")
    print(f"   Total documents: {rag.get_document_count()}")
    
    return rag

# Usage
if __name__ == "__main__":
    knowledge_base = process_knowledge_base("./company_docs")
    
    # Test the knowledge base
    test_queries = [
        "What are our company policies?",
        "How do I submit expenses?",
        "What is our remote work policy?"
    ]
    
    for query in test_queries:
        answer = knowledge_base.query(query)
        print(f"\nQ: {query}")
        print(f"A: {answer}")
```

#### JavaScript
```javascript
import { RAG } from '@rag-templates/core';
import fs from 'fs/promises';
import path from 'path';

async function processKnowledgeBase(directoryPath) {
    /**
     * Process a directory of documents into a RAG knowledge base.
     */
    
    const rag = new RAG();
    const processedFiles = [];
    
    // Supported file types
    const supportedExtensions = ['.txt', '.md', '.json'];
    
    async function processDirectory(dirPath) {
        const entries = await fs.readdir(dirPath, { withFileTypes: true });
        
        for (const entry of entries) {
            const fullPath = path.join(dirPath, entry.name);
            
            if (entry.isDirectory()) {
                await processDirectory(fullPath);
            } else if (entry.isFile()) {
                const fileExt = path.extname(entry.name).toLowerCase();
                
                if (supportedExtensions.includes(fileExt)) {
                    try {
                        if (fileExt === '.json') {
                            // Handle JSON files
                            const content = await fs.readFile(fullPath, 'utf8');
                            const data = JSON.parse(content);
                            
                            if (Array.isArray(data)) {
                                // Array of documents
                                for (let i = 0; i < data.length; i++) {
                                    const item = data[i];
                                    if (typeof item === 'object' && item.content) {
                                        await rag.addDocuments([item]);
                                    } else if (typeof item === 'string') {
                                        await rag.addDocuments([{
                                            content: item,
                                            source: `${entry.name}[${i}]`
                                        }]);
                                    }
                                }
                            } else if (typeof data === 'object' && data.content) {
                                // Single document
                                await rag.addDocuments([data]);
                            }
                        } else {
                            // Handle text files
                            const content = await fs.readFile(fullPath, 'utf8');
                            const stats = await fs.stat(fullPath);
                            
                            await rag.addDocuments([{
                                content: content,
                                title: entry.name,
                                source: fullPath,
                                metadata: {
                                    fileType: fileExt,
                                    fileSize: stats.size
                                }
                            }]);
                        }
                        
                        processedFiles.push(fullPath);
                        console.log(`✅ Processed: ${fullPath}`);
                        
                    } catch (error) {
                        console.error(`❌ Error processing ${fullPath}: ${error.message}`);
                    }
                }
            }
        }
    }
    
    await processDirectory(directoryPath);
    
    console.log(`\n📊 Processing complete:`);
    console.log(`   Files processed: ${processedFiles.length}`);
    console.log(`   Total documents: ${await rag.getDocumentCount()}`);
    
    return rag;
}

// Usage
async function main() {
    const knowledgeBase = await processKnowledgeBase("./company_docs");
    
    // Test the knowledge base
    const testQueries = [
        "What are our company policies?",
        "How do I submit expenses?",
        "What is our remote work policy?"
    ];
    
    for (const query of testQueries) {
        const answer = await knowledgeBase.query(query);
        console.log(`\nQ: ${query}`);
        console.log(`A: ${answer}`);
    }
}

main().catch(console.error);
```

## Standard API Examples

### Advanced RAG Configuration

#### Python
```python
from rag_templates import ConfigurableRAG

# Advanced configuration with technique selection
rag = ConfigurableRAG({
    "technique": "colbert",
    "llm_provider": "openai",
    "llm_config": {
        "model": "gpt-4o-mini",
        "temperature": 0.1,
        "max_tokens": 1000
    },
    "embedding_model": "text-embedding-3-large",
    "embedding_config": {
        "dimension": 3072,
        "batch_size": 50
    },
    "technique_config": {
        "max_query_length": 512,
        "doc_maxlen": 180,
        "top_k": 15
    },
    "caching": {
        "enabled": True,
        "ttl": 3600
    }
})

# Load documents with metadata
documents = [
    {
        "content": "Quantum computing uses quantum mechanical phenomena to process information.",
        "title": "Quantum Computing Basics",
        "category": "technology",
        "difficulty": "advanced",
        "tags": ["quantum", "computing", "physics"]
    },
    {
        "content": "Artificial intelligence mimics human cognitive functions in machines.",
        "title": "AI Overview",
        "category": "technology", 
        "difficulty": "intermediate",
        "tags": ["ai", "machine learning", "cognition"]
    }
]

rag.add_documents(documents)

# Advanced querying with options
result = rag.query("How does quantum computing work?", {
    "max_results": 10,
    "include_sources": True,
    "min_similarity": 0.8,
    "source_filter": "technology",
    "response_format": "detailed"
})

print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Processing time: {result.metadata.get('processing_time_ms', 0)}ms")

print("\nSources:")
for i, source in enumerate(result.sources, 1):
    print(f"{i}. {source.title} (similarity: {source.similarity:.2f})")
    print(f"   Tags: {source.metadata.get('tags', [])}")
    print(f"   Difficulty: {source.metadata.get('difficulty', 'unknown')}")
```

### Multi-Technique Comparison

#### Python
```python
from rag_templates import ConfigurableRAG

def compare_rag_techniques(query, documents):
    """Compare different RAG techniques on the same query."""
    
    techniques = ["basic", "colbert", "hyde", "crag"]
    results = {}
    
    for technique in techniques:
        print(f"Testing {technique} technique...")
        
        rag = ConfigurableRAG({
            "technique": technique,
            "llm_provider": "openai",
            "max_results": 5
        })
        
        # Add documents
        rag.add_documents(documents)
        
        # Query with timing
        import time
        start_time = time.time()
        
        result = rag.query(query, {
            "include_sources": True,
            "min_similarity": 0.7
        })
        
        end_time = time.time()
        
        results[technique] = {
            "answer": result.answer,
            "confidence": result.confidence,
            "sources_count": len(result.sources) if result.sources else 0,
            "processing_time": (end_time - start_time) * 1000,  # ms
            "technique_info": rag.get_technique_info(technique)
        }
    
    return results

# Test documents
test_documents = [
    {
        "content": "Machine learning is a method of data analysis that automates analytical model building.",
        "title": "ML Definition",
        "category": "ai"
    },
    {
        "content": "Deep learning is a subset of machine learning that uses neural networks with multiple layers.",
        "title": "Deep Learning Explained",
        "category": "ai"
    },
    {
        "content": "Natural language processing enables computers to understand and interpret human language.",
        "title": "NLP Overview",
        "category": "ai"
    }
]

# Compare techniques
query = "What is the relationship between machine learning and deep learning?"
comparison_results = compare_rag_techniques(query, test_documents)

# Display results
print(f"\nQuery: {query}\n")
print("Technique Comparison Results:")
print("=" * 50)

for technique, result in comparison_results.items():
    print(f"\n{technique.upper()}:")
    print(f"  Answer: {result['answer'][:100]}...")
    print(f"  Confidence: {result['confidence']:.2f}")
    print(f"  Sources: {result['sources_count']}")
    print(f"  Time: {result['processing_time']:.1f}ms")
    print(f"  Best for: {result['technique_info'].get('best_for', 'N/A')}")

# Find best technique
best_technique = max(comparison_results.items(), 
                    key=lambda x: x[1]['confidence'])
print(f"\nBest technique for this query: {best_technique[0]} "
      f"(confidence: {best_technique[1]['confidence']:.2f})")
```

### Dynamic Technique Switching

#### JavaScript
```javascript
import { ConfigurableRAG } from '@rag-templates/core';

class AdaptiveRAG {
    constructor() {
        this.techniques = {
            basic: new ConfigurableRAG({ technique: 'basic' }),
            colbert: new ConfigurableRAG({ technique: 'colbert' }),
            hyde: new ConfigurableRAG({ technique: 'hyde' }),
            crag: new ConfigurableRAG({ technique: 'crag' })
        };
        
        this.queryPatterns = [
            { pattern: /code|programming|function|class/i, technique: 'colbert' },
            { pattern: /research|study|analysis|hypothesis/i, technique: 'hyde' },
            { pattern: /fact|definition|what is|explain/i, technique: 'crag' },
            { pattern: /.*/, technique: 'basic' } // default
        ];
    }
    
    async addDocuments(documents) {
        // Add documents to all techniques
        for (const rag of Object.values(this.techniques)) {
            await rag.addDocuments(documents);
        }
    }
    
    selectTechnique(query) {
        for (const { pattern, technique } of this.queryPatterns) {
            if (pattern.test(query)) {
                return technique;
            }
        }
        return 'basic';
    }
    
    async query(queryText, options = {}) {
        const selectedTechnique = this.selectTechnique(queryText);
        const rag = this.techniques[selectedTechnique];
        
        console.log(`Using ${selectedTechnique} technique for query: "${queryText}"`);
        
        const result = await rag.query(queryText, {
            ...options,
            includeSources: true
        });
        
        return {
            ...result,
            technique: selectedTechnique,
            techniqueInfo: rag.getTechniqueInfo(selectedTechnique)
        };
    }
    
    async compareAllTechniques(queryText) {
        const results = {};
        
        for (const [name, rag] of Object.entries(this.techniques)) {
            const start = Date.now();
            const result = await rag.query(queryText, { includeSources: true });
            const end = Date.now();
            
            results[name] = {
                answer: result.answer,
                confidence: result.confidence,
                sourcesCount: result.sources?.length || 0,
                processingTime: end - start
            };
        }
        
        return results;
    }
}

// Usage example
async function demonstrateAdaptiveRAG() {
    const adaptiveRAG = new AdaptiveRAG();
    
    // Add sample documents
    await adaptiveRAG.addDocuments([
        {
            content: "Python is a high-level programming language known for its simplicity.",
            title: "Python Programming",
            category: "programming"
        },
        {
            content: "Recent studies show that machine learning improves healthcare outcomes.",
            title: "ML in Healthcare Research",
            category: "research"
        },
        {
            content: "Artificial intelligence is the simulation of human intelligence in machines.",
            title: "AI Definition",
            category: "definition"
        }
    ]);
    
    // Test different query types
    const testQueries = [
        "How do you write a Python function?",  // Should use ColBERT
        "What does research show about ML in healthcare?",  // Should use HyDE
        "What is artificial intelligence?",  // Should use CRAG
        "Tell me about technology trends"  // Should use Basic
    ];
    
    for (const query of testQueries) {
        console.log(`\n${'='.repeat(60)}`);
        const result = await adaptiveRAG.query(query);
        
        console.log(`Query: ${query}`);
        console.log(`Selected Technique: ${result.technique}`);
        console.log(`Answer: ${result.answer}`);
        console.log(`Confidence: ${result.confidence?.toFixed(2) || 'N/A'}`);
        console.log(`Best for: ${result.techniqueInfo?.bestFor || 'N/A'}`);
    }
    
    // Compare all techniques on one query
    console.log(`\n${'='.repeat(60)}`);
    console.log("TECHNIQUE COMPARISON");
    console.log(`${'='.repeat(60)}`);
    
    const comparisonQuery = "How does machine learning work?";
    const comparison = await adaptiveRAG.compareAllTechniques(comparisonQuery);
    
    console.log(`Query: ${comparisonQuery}\n`);
    
    for (const [technique, result] of Object.entries(comparison)) {
        console.log(`${technique.toUpperCase()}:`);
        console.log(`  Answer: ${result.answer.substring(0, 100)}...`);
        console.log(`  Confidence: ${result.confidence?.toFixed(2) || 'N/A'}`);
        console.log(`  Sources: ${result.sourcesCount}`);
        console.log(`  Time: ${result.processingTime}ms\n`);
    }
}

demonstrateAdaptiveRAG().catch(console.error);
```

## Enterprise API Examples

### Production-Ready RAG System

#### Python
```python
from rag_templates import ConfigurableRAG
from rag_templates.config import ConfigManager
import logging
import time
from typing import Dict, List, Optional

class EnterpriseRAGSystem:
    """Production-ready RAG system with enterprise features."""
    
    def __init__(self, config_path: str):
        # Load enterprise configuration
        self.config = ConfigManager.from_file(config_path)
        
        # Initialize RAG with enterprise features
        self.rag = ConfigurableRAG(self.config)
        
        # Setup logging
        self.setup_logging()
        
        # Performance metrics
        self.metrics = {
            "queries_processed": 0,
            "total_processing_time": 0,
            "cache_hits": 0,
            "errors": 0
        }
        
        self.logger.info("Enterprise RAG system initialized")
    
    def setup_logging(self):
        """Setup structured logging for production."""
        logging.basicConfig(
            level=getattr(logging, self.config.get("logging.level", "INFO")),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('rag_system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def add_documents_with_validation(self, documents: List[Dict]) -> Dict:
        """Add documents with validation and error handling."""
        try:
            # Validate documents
            validated_docs = []
            for i, doc in enumerate(documents):
                if not isinstance(doc, dict):
                    raise ValueError(f"Document {i} must be a dictionary")
                
                if "content" not in doc:
                    raise ValueError(f"Document {i} missing required 'content' field")
                
                if len(doc["content"].strip()) < 10:
                    self.logger.warning(f"Document {i} has very short content")
                
                # Add metadata
                doc["metadata"] = doc.get("metadata", {})
                doc["metadata"]["added_at"] = time.time()
                doc["metadata"]["validated"] = True
                
                validated_docs.append(doc)
            
            # Add to RAG system
            self.rag.add_documents(validated_docs)
            
            self.logger.info(f"Successfully added {len(validated_docs)} documents")
            
            return {
                "success": True,
                "documents_added": len(validated_docs),
                "total_documents": self.rag.get_document_count()
            }
            
        except Exception as e:
            self.logger.error(f"Error adding documents: {e}")
            self.metrics["errors"] += 1
            return {
                "success": False,
                "error": str(e),
                "documents_added": 0
            }
    
    def query_with_monitoring(self, 
                            query: str, 
                            options: Optional[Dict] = None,
                            user_id: Optional[str] = None) -> Dict:
        """Query with comprehensive monitoring and error handling."""
        
        start_time = time.time()
        query_id = f"query_{int(start_time * 1000)}"
        
        try:
            # Log query
            self.logger.info(f"Processing query {query_id}: {query[:100]}...")
            
            # Security validation
            if len(query) > 1000:
                raise ValueError("Query too long (max 1000 characters)")
            
            if any(word in query.lower() for word in ["drop", "delete", "truncate"]):
                raise ValueError("Query contains potentially harmful content")
            
            # Process query
            result = self.rag.query(query, {
                **(options or {}),
                "include_sources": True,
                "pipeline_config": {
                    "monitoring": True,
                    "security": True,
                    "caching": True
                }
            })
            
            # Calculate metrics
            processing_time = (time.time() - start_time) * 1000
            self.metrics["queries_processed"] += 1
            self.metrics["total_processing_time"] += processing_time
            
            if result.metadata and result.metadata.get("cache_hit"):
                self.metrics["cache_hits"] += 1
            
            # Log success
            self.logger.info(f"Query {query_id} completed in {processing_time:.1f}ms")
            
            return {
                "success": True,
                "query_id": query_id,
                "answer": result.answer,
                "confidence": result.confidence,
                "sources": [
                    {
                        "title": s.title,
                        "similarity": s.similarity,
                        "source": s.source
                    } for s in (result.sources or [])
                ],
                "metadata": {
                    "processing_time_ms": processing_time,
                    "cache_hit": result.metadata.get("cache_hit", False),
                    "user_id": user_id,
                    "timestamp": time.time()
                }
            }
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.metrics["errors"] += 1
            
            self.logger.error(f"Query {query_id} failed after {processing_time:.1f}ms: {e}")
            
            return {
                "success": False,
                "query_id": query_id,
                "error": str(e),
                "metadata": {
                    "processing_time_ms": processing_time,
                    "user_id": user_id,
                    "timestamp": time.time()
                }
            }
    
    def get_system_metrics(self) -> Dict:
        """Get comprehensive system metrics."""
        avg_processing_time = (
            self.metrics["total_processing_time"] / self.metrics["queries_processed"]
            if self.metrics["queries_processed"] > 0 else 0
        )
        
        cache_hit_rate = (
            self.metrics["cache_hits"] / self.metrics["queries_processed"]
            if self.metrics["queries_processed"] > 0 else 0
        )
        
        return {
            "queries_processed": self.metrics["queries_processed"],
            "average_processing_time_ms": avg_processing_time,
            "cache_hit_rate": cache_hit_rate,
            "error_rate": self.metrics["errors"] / max(self.metrics["queries_processed"], 1),
            "total_documents": self.rag.get_document_count(),
            "system_status": "healthy" if self.metrics["errors"] < 10 else "degraded"
        }
    
    def health_check(self) -> Dict:
        """Perform system health check."""
        try:
            # Test query
            test_result = self.rag.query("health check test", {"max_results": 1})
            
            # Check database connection
            doc_count = self.rag.get_document_count()
            
            return {
                "status": "healthy",
                "database_connected": True,
                "document_count": doc_count,
                "test_query_successful": True,
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }

# Usage example
def main():
    # Initialize enterprise system
    rag_system = EnterpriseRAGSystem("enterprise-config.yaml")
    
    # Add documents with validation
    documents = [
        {
            "content": "Enterprise RAG systems require robust error handling and monitoring.",
            "title": "Enterprise RAG Best Practices",
            "category": "enterprise",
            "metadata": {"department": "engineering", "classification": "internal"}
        },
        {
            "content": "Production systems must handle high query volumes with low latency.",
            "title": "Production System Requirements",
            "category": "enterprise",
            "metadata": {"department": "engineering", "classification": "internal"}
        }
    ]
    
    add_result = rag_system.add_documents_with_validation(documents)
    print(f"Document addition result: {add_result}")
    
    # Process queries with monitoring
    queries = [
        "What are enterprise RAG best practices?",
        "How should production systems handle high volumes?",
        "What are the monitoring requirements?"
    ]
    
    for query in queries:
        result = rag_system.query_with_monitoring(
            query, 
            {"max_results": 5},
            user_id="demo_user"
        )
        
        if result["success"]:
            print(f"\nQuery: {query}")
            print(f"Answer: {result['answer']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Processing time: {result['metadata']['processing_time_ms']:.1f}ms")
            print(f"Sources: {len(result['sources'])}")
        else:
            print(f"\nQuery failed: {result['error']}")
    
    # Display system metrics
    metrics = rag_system.get_system_metrics()
    print(f"\nSystem Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Health check
    health = rag_system.health_check()
    print(f"\nHealth Check: {health}")

if __name__ == "__main__":
    main()
```

## MCP Integration Examples

### Claude Desktop Integration

#### Complete MCP Server Example

```javascript
// claude-rag-server.js
import { createMCPServer } from '@rag-templates/mcp';