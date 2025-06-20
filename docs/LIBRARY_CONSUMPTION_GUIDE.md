# Library Consumption Guide

A comprehensive guide for consuming rag-templates as a library, transforming from complex setup to dead-simple integration.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Progressive Complexity](#progressive-complexity)
3. [Language Parity Examples](#language-parity-examples)
4. [Common Use Cases](#common-use-cases)
5. [Configuration Patterns](#configuration-patterns)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)
8. [FAQ](#faq)

## Quick Start

### Installation

#### Python
```bash
pip install rag-templates
```

#### JavaScript/Node.js
```bash
npm install @rag-templates/core
```

### Your First RAG Application

#### Python - 30 Seconds to RAG
```python
from rag_templates import RAG

# Zero configuration - works immediately
rag = RAG()

# Add your documents
rag.add_documents([
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing enables computers to understand text."
])

# Ask questions
answer = rag.query("What is machine learning?")
print(answer)
# Output: "Machine learning is a subset of artificial intelligence..."
```

#### JavaScript - 30 Seconds to RAG
```javascript
import { RAG } from '@rag-templates/core';

// Zero configuration - works immediately
const rag = new RAG();

// Add your documents
await rag.addDocuments([
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing enables computers to understand text."
]);

// Ask questions
const answer = await rag.query("What is machine learning?");
console.log(answer);
// Output: "Machine learning is a subset of artificial intelligence..."
```

## Progressive Complexity

The framework provides three tiers of complexity to match your needs:

### Tier 1: Simple API (Zero Configuration)

**Perfect for**: Prototypes, demos, learning, simple applications

**Philosophy**: Works immediately with zero setup

#### Python
```python
from rag_templates import RAG

# Instant RAG - no configuration needed
rag = RAG()

# Add documents from various sources
rag.add_documents([
    "Document content as string",
    {"content": "Document with metadata", "source": "file.pdf"},
    {"title": "Custom Title", "content": "More content"}
])

# Simple querying
answer = rag.query("Your question")
print(answer)  # String response

# Check status
count = rag.get_document_count()
print(f"Documents in knowledge base: {count}")
```

#### JavaScript
```javascript
import { RAG } from '@rag-templates/core';

// Instant RAG - no configuration needed
const rag = new RAG();

// Add documents from various sources
await rag.addDocuments([
    "Document content as string",
    {content: "Document with metadata", source: "file.pdf"},
    {title: "Custom Title", content: "More content"}
]);

// Simple querying
const answer = await rag.query("Your question");
console.log(answer);  // String response

// Check status
const count = await rag.getDocumentCount();
console.log(`Documents in knowledge base: ${count}`);
```

### Tier 2: Standard API (Basic Configuration)

**Perfect for**: Production applications, technique selection, custom configuration

**Philosophy**: Simple configuration for powerful features

#### Python
```python
from rag_templates import ConfigurableRAG

# Technique selection and basic configuration
rag = ConfigurableRAG({
    'technique': 'colbert',           # Choose RAG technique
    'llm_provider': 'openai',         # LLM provider
    'embedding_model': 'text-embedding-3-small',
    'max_results': 5,                 # Default result count
    'temperature': 0.1                # LLM temperature
})

# Advanced querying with options
result = rag.query("What is neural network architecture?", {
    'max_results': 10,
    'include_sources': True,
    'min_similarity': 0.8,
    'source_filter': 'academic_papers'
})

# Rich result object
print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence}")
print(f"Sources: {len(result.sources)}")
for source in result.sources:
    print(f"  - {source.title} (similarity: {source.similarity:.2f})")
```

#### JavaScript
```javascript
import { ConfigurableRAG } from '@rag-templates/core';

// Technique selection and basic configuration
const rag = new ConfigurableRAG({
    technique: 'colbert',             // Choose RAG technique
    llmProvider: 'openai',            // LLM provider
    embeddingModel: 'text-embedding-3-small',
    maxResults: 5,                    // Default result count
    temperature: 0.1                  // LLM temperature
});

// Advanced querying with options
const result = await rag.query("What is neural network architecture?", {
    maxResults: 10,
    includeSources: true,
    minSimilarity: 0.8,
    sourceFilter: 'academic_papers'
});

// Rich result object
console.log(`Answer: ${result.answer}`);
console.log(`Confidence: ${result.confidence}`);
console.log(`Sources: ${result.sources.length}`);
result.sources.forEach(source => {
    console.log(`  - ${source.title} (similarity: ${source.similarity.toFixed(2)})`);
});
```

### Tier 3: Enterprise API (Full Control)

**Perfect for**: Enterprise deployments, advanced features, custom pipelines

**Philosophy**: Complete control with enterprise features

#### Python
```python
from rag_templates import ConfigurableRAG
from rag_templates.config import ConfigManager

# Load enterprise configuration
config = ConfigManager.from_file('enterprise-config.yaml')
rag = ConfigurableRAG(config)

# Enterprise query with full pipeline control
result = rag.query("Complex enterprise query", {
    'pipeline_config': {
        'caching': True,              # Enable response caching
        'monitoring': True,           # Enable metrics collection
        'reconciliation': True,       # Enable data consistency checks
        'security': {
            'input_validation': True,
            'output_filtering': True
        }
    },
    'retrieval_config': {
        'hybrid_search': True,        # Combine multiple search methods
        'reranking': True,           # Apply reranking algorithms
        'query_expansion': True       # Expand query with synonyms
    },
    'generation_config': {
        'fact_checking': True,        # Verify generated facts
        'citation_mode': 'detailed',  # Include detailed citations
        'response_format': 'structured' # Structured response format
    }
})

# Enterprise result with full metadata
print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence}")
print(f"Processing time: {result.metadata.processing_time_ms}ms")
print(f"Cache hit: {result.metadata.cache_hit}")
print(f"Security score: {result.metadata.security_score}")
```

## Language Parity Examples

The framework provides feature-equivalent APIs across Python and JavaScript:

### Document Management

#### Python
```python
from rag_templates import RAG

rag = RAG()

# Add documents
rag.add_documents([
    "Simple string document",
    {
        "content": "Document with metadata",
        "title": "Research Paper",
        "source": "academic_journal.pdf",
        "metadata": {"author": "Dr. Smith", "year": 2024}
    }
])

# Bulk document loading
rag.load_from_directory("./documents", {
    "file_types": [".pdf", ".txt", ".md"],
    "chunk_size": 1000,
    "chunk_overlap": 200
})

# Document management
count = rag.get_document_count()
rag.clear_knowledge_base()  # Warning: irreversible
```

#### JavaScript
```javascript
import { RAG } from '@rag-templates/core';

const rag = new RAG();

// Add documents
await rag.addDocuments([
    "Simple string document",
    {
        content: "Document with metadata",
        title: "Research Paper",
        source: "academic_journal.pdf",
        metadata: {author: "Dr. Smith", year: 2024}
    }
]);

// Bulk document loading
await rag.loadFromDirectory("./documents", {
    fileTypes: [".pdf", ".txt", ".md"],
    chunkSize: 1000,
    chunkOverlap: 200
});

// Document management
const count = await rag.getDocumentCount();
await rag.clearKnowledgeBase();  // Warning: irreversible
```

### Configuration Management

#### Python
```python
from rag_templates import ConfigurableRAG

# Configuration object
config = {
    'technique': 'colbert',
    'llm_provider': 'anthropic',
    'llm_config': {
        'model': 'claude-3-sonnet',
        'temperature': 0.1,
        'max_tokens': 2000
    },
    'embedding_config': {
        'model': 'text-embedding-3-large',
        'dimension': 3072
    },
    'database': {
        'host': 'localhost',
        'port': 52773,
        'namespace': 'RAG_DEMO'
    }
}

rag = ConfigurableRAG(config)

# Runtime configuration access
llm_model = rag.get_config('llm_config.model')
rag.set_config('temperature', 0.2)
```

#### JavaScript
```javascript
import { ConfigurableRAG } from '@rag-templates/core';

// Configuration object
const config = {
    technique: 'colbert',
    llmProvider: 'anthropic',
    llmConfig: {
        model: 'claude-3-sonnet',
        temperature: 0.1,
        maxTokens: 2000
    },
    embeddingConfig: {
        model: 'text-embedding-3-large',
        dimension: 3072
    },
    database: {
        host: 'localhost',
        port: 52773,
        namespace: 'RAG_DEMO'
    }
};

const rag = new ConfigurableRAG(config);

// Runtime configuration access
const llmModel = rag.getConfig('llmConfig.model');
rag.setConfig('temperature', 0.2);
```

## Common Use Cases

### 1. Document Q&A System

#### Python
```python
from rag_templates import RAG
import os

# Initialize RAG
rag = RAG()

# Load company documents
document_dir = "./company_docs"
for filename in os.listdir(document_dir):
    if filename.endswith('.txt'):
        with open(os.path.join(document_dir, filename), 'r') as f:
            content = f.read()
            rag.add_documents([{
                "content": content,
                "source": filename,
                "type": "company_policy"
            }])

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

// Load company documents
const documentDir = "./company_docs";
const files = await fs.readdir(documentDir);

for (const filename of files) {
    if (filename.endsWith('.txt')) {
        const content = await fs.readFile(path.join(documentDir, filename), 'utf8');
        await rag.addDocuments([{
            content: content,
            source: filename,
            type: "company_policy"
        }]);
    }
}

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

### 2. Research Assistant

#### Python
```python
from rag_templates import ConfigurableRAG

# Configure for research use case
rag = ConfigurableRAG({
    'technique': 'hyde',  # Good for complex reasoning
    'llm_provider': 'openai',
    'llm_config': {
        'model': 'gpt-4',
        'temperature': 0.1  # Low temperature for factual responses
    },
    'max_results': 10,
    'include_citations': True
})

# Load research papers
research_papers = [
    {"content": "Paper 1 content...", "title": "AI in Healthcare", "authors": ["Dr. A", "Dr. B"]},
    {"content": "Paper 2 content...", "title": "Machine Learning Ethics", "authors": ["Dr. C"]},
    # ... more papers
]

rag.add_documents(research_papers)

# Research query with detailed analysis
result = rag.query("What are the ethical implications of AI in healthcare?", {
    'analysis_depth': 'comprehensive',
    'include_sources': True,
    'citation_style': 'academic'
})

print(f"Research Summary: {result.answer}")
print(f"Key Sources: {len(result.sources)}")
for source in result.sources:
    print(f"  - {source.title} by {', '.join(source.authors)}")
```

### 3. Customer Support Bot

#### JavaScript
```javascript
import { ConfigurableRAG } from '@rag-templates/core';

// Configure for customer support
const supportBot = new ConfigurableRAG({
    technique: 'basic',  // Fast responses for customer support
    llmProvider: 'openai',
    llmConfig: {
        model: 'gpt-3.5-turbo',
        temperature: 0.3,  // Slightly creative for helpful responses
        maxTokens: 500     // Concise responses
    },
    responseStyle: 'helpful_and_concise'
});

// Load support documentation
await supportBot.addDocuments([
    {content: "How to reset password...", category: "account"},
    {content: "Billing information...", category: "billing"},
    {content: "Product features...", category: "product"},
    // ... more support docs
]);

// Handle customer queries
async function handleCustomerQuery(query, customerContext = {}) {
    const result = await supportBot.query(query, {
        maxResults: 3,
        includeSources: true,
        customerTier: customerContext.tier || 'standard',
        urgency: customerContext.urgency || 'normal'
    });
    
    return {
        answer: result.answer,
        confidence: result.confidence,
        suggestedActions: result.suggestedActions,
        escalateToHuman: result.confidence < 0.7
    };
}

// Example usage
const response = await handleCustomerQuery(
    "How do I cancel my subscription?",
    {tier: 'premium', urgency: 'high'}
);

console.log(response);
```

### 4. Code Documentation Assistant

#### Python
```python
from rag_templates import ConfigurableRAG
import ast
import os

# Configure for code documentation
code_assistant = ConfigurableRAG({
    'technique': 'colbert',  # Good for precise code matching
    'llm_provider': 'anthropic',
    'llm_config': {
        'model': 'claude-3-sonnet',
        'temperature': 0.0  # Deterministic for code
    },
    'code_understanding': True
})

# Index codebase
def index_python_files(directory):
    documents = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r') as f:
                    content = f.read()
                
                # Parse AST for better understanding
                try:
                    tree = ast.parse(content)
                    functions = [node.name for node in ast.walk(tree) 
                               if isinstance(node, ast.FunctionDef)]
                    classes = [node.name for node in ast.walk(tree) 
                             if isinstance(node, ast.ClassDef)]
                    
                    documents.append({
                        'content': content,
                        'filepath': filepath,
                        'functions': functions,
                        'classes': classes,
                        'type': 'python_code'
                    })
                except:
                    pass  # Skip files with syntax errors
    
    return documents

# Index the codebase
codebase_docs = index_python_files('./src')
code_assistant.add_documents(codebase_docs)

# Query code documentation
def ask_about_code(question):
    result = code_assistant.query(question, {
        'include_sources': True,
        'code_context': True,
        'max_results': 5
    })
    
    print(f"Answer: {result.answer}")
    print("\nRelevant Code Files:")
    for source in result.sources:
        print(f"  - {source.filepath}")
        if source.functions:
            print(f"    Functions: {', '.join(source.functions)}")
        if source.classes:
            print(f"    Classes: {', '.join(source.classes)}")

# Example usage
ask_about_code("How do I implement user authentication?")
ask_about_code("What's the database connection pattern used?")
```

## Configuration Patterns

### Environment-Based Configuration

#### Python
```python
import os
from rag_templates import ConfigurableRAG

# Environment-based configuration (recommended for production)
rag = ConfigurableRAG({
    'database': {
        'host': os.getenv('IRIS_HOST', 'localhost'),
        'port': int(os.getenv('IRIS_PORT', '52773')),
        'username': os.getenv('IRIS_USERNAME', 'demo'),
        'password': os.getenv('IRIS_PASSWORD', 'demo'),
        'namespace': os.getenv('IRIS_NAMESPACE', 'RAG_PROD')
    },
    'llm_provider': os.getenv('LLM_PROVIDER', 'openai'),
    'llm_config': {
        'api_key': os.getenv('OPENAI_API_KEY'),
        'model': os.getenv('LLM_MODEL', 'gpt-4o-mini')
    },
    'embedding_model': os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
})
```

#### JavaScript
```javascript
import { ConfigurableRAG } from '@rag-templates/core';

// Environment-based configuration (recommended for production)
const rag = new ConfigurableRAG({
    database: {
        host: process.env.IRIS_HOST || 'localhost',
        port: parseInt(process.env.IRIS_PORT || '52773'),
        username: process.env.IRIS_USERNAME || 'demo',
        password: process.env.IRIS_PASSWORD || 'demo',
        namespace: process.env.IRIS_NAMESPACE || 'RAG_PROD'
    },
    llmProvider: process.env.LLM_PROVIDER || 'openai',
    llmConfig: {
        apiKey: process.env.OPENAI_API_KEY,
        model: process.env.LLM_MODEL || 'gpt-4o-mini'
    },
    embeddingModel: process.env.EMBEDDING_MODEL || 'text-embedding-3-small'
});
```

### Configuration Files

#### YAML Configuration
```yaml
# config/production.yaml
technique: "colbert"
llm_provider: "openai"

llm_config:
  model: "gpt-4o-mini"
  temperature: 0.1
  max_tokens: 1000

embedding_config:
  model: "text-embedding-3-small"
  dimension: 1536
  batch_size: 100

database:
  host: "${IRIS_HOST}"
  port: "${IRIS_PORT}"
  username: "${IRIS_USERNAME}"
  password: "${IRIS_PASSWORD}"
  namespace: "RAG_PRODUCTION"

vector_index:
  type: "HNSW"
  M: 16
  efConstruction: 200

caching:
  enabled: true
  ttl: 3600
  max_size: 1000

monitoring:
  enabled: true
  metrics_endpoint: "${METRICS_ENDPOINT}"
  log_level: "INFO"
```

#### Loading Configuration Files

##### Python
```python
from rag_templates import ConfigurableRAG
from rag_templates.config import ConfigManager

# Load from YAML file
config = ConfigManager.from_file('config/production.yaml')
rag = ConfigurableRAG(config)

# Or load directly
rag = ConfigurableRAG.from_config_file('config/production.yaml')
```

##### JavaScript
```javascript
import { ConfigurableRAG, ConfigManager } from '@rag-templates/core';

// Load from YAML file
const config = await ConfigManager.fromFile('config/production.yaml');
const rag = new ConfigurableRAG(config);

// Or load directly
const rag = await ConfigurableRAG.fromConfigFile('config/production.yaml');
```

## Best Practices

### 1. Start Simple, Scale Up

```python
# Start with Simple API for prototyping
from rag_templates import RAG

rag = RAG()
# ... prototype and test

# Upgrade to Standard API when you need more control
from rag_templates import ConfigurableRAG

rag = ConfigurableRAG({'technique': 'colbert'})
# ... production deployment

# Move to Enterprise API for advanced features
config = ConfigManager.from_file('enterprise-config.yaml')
rag = ConfigurableRAG(config)
# ... enterprise deployment
```

### 2. Environment-Based Configuration

```bash
# .env file
IRIS_HOST=production-iris.company.com
IRIS_PORT=52773
IRIS_USERNAME=rag_service
IRIS_PASSWORD=secure_password
IRIS_NAMESPACE=RAG_PRODUCTION

OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

# Optional: Advanced settings
RAG_TECHNIQUE=colbert
RAG_MAX_RESULTS=10
RAG_CACHE_TTL=3600
```

### 3. Error Handling

#### Python
```python
from rag_templates import RAG, RAGError, ConfigurationError

try:
    rag = RAG()
    rag.add_documents(documents)
    answer = rag.query("Question")
except ConfigurationError as e:
    print(f"Configuration issue: {e}")
    # Handle configuration problems
except RAGError as e:
    print(f"RAG operation failed: {e}")
    # Handle RAG-specific errors
except Exception as e:
    print(f"Unexpected error: {e}")
    # Handle other errors
```

#### JavaScript
```javascript
import { RAG, RAGError, ConfigurationError } from '@rag-templates/core';

try {
    const rag = new RAG();
    await rag.addDocuments(documents);
    const answer = await rag.query("Question");
} catch (error) {
    if (error instanceof ConfigurationError) {
        console.log(`Configuration issue: ${error.message}`);
        // Handle configuration problems
    } else if (error instanceof RAGError) {
        console.log(`RAG operation failed: ${error.message}`);
        // Handle RAG-specific errors
    } else {
        console.log(`Unexpected error: ${error.message}`);
        // Handle other errors
    }
}
```

### 4. Performance Optimization

```python
from rag_templates import ConfigurableRAG

# Optimize for performance
rag = ConfigurableRAG({
    'technique': 'basic',  # Fastest technique
    'embedding_config': {
        'batch_size': 100,  # Batch embeddings for efficiency
        'cache_embeddings': True
    },
    'caching': {
        'enabled': True,
        'ttl': 3600,  # Cache responses for 1 hour
        'max_size': 1000
    },
    'database': {
        'connection_pool_size': 10,  # Connection pooling
        'query_timeout': 30
    }
})
```

### 5. Security Best Practices

```python
from rag_templates import ConfigurableRAG

# Security-focused configuration
rag = ConfigurableRAG({
    'security': {
        'input_validation': True,      # Validate all inputs
        'output_filtering': True,      # Filter sensitive outputs
        'rate_limiting': True,         # Prevent abuse
        'audit_logging': True          # Log all operations
    },
    'database': {
        'ssl_enabled': True,           # Use SSL connections
        'connection_timeout': 30
    },
    'llm_config': {
        'content_filter': True,        # Filter inappropriate content
        'max_tokens': 1000            # Limit response length
    }
})
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: `ImportError: No module named 'rag_templates'`

**Solution**:
```bash
# Python
pip install rag-templates

# JavaScript
npm install @rag-templates/core
```

#### 2. Database Connection Issues

**Problem**: `ConnectionError: Failed to connect to IRIS database`

**Solutions**:
```python
# Check environment variables
import os
print(f"IRIS_HOST: {os.getenv('IRIS_HOST')}")
print(f"IRIS_PORT: {os.getenv('IRIS_PORT')}")

# Test connection manually
from rag_templates.config import ConfigManager
config = ConfigManager()
db_config = config.get_database_config()
print(f"Database config: {db_config}")

# Use explicit configuration
rag = ConfigurableRAG({
    'database': {
        'host': 'localhost',
        'port': 52773,
        'username': 'demo',
        'password': 'demo'
    }
})
```

#### 3. LLM API Issues

**Problem**: `APIError: Invalid API key`

**Solutions**:
```bash
# Set API key
export OPENAI_API_KEY=your-api-key

# Or use configuration
```

```python
rag = ConfigurableRAG({
    'llm_config': {
        'api_key': 'your-api-key',
        'model': 'gpt-4o-mini'
    }
})
```

#### 4. Memory Issues

**Problem**: `MemoryError: Out of memory during embedding generation`

**Solutions**:
```python
# Reduce batch size
rag = ConfigurableRAG({
    'embedding_config': {
        'batch_size': 10,  # Reduce from default 100
        'max_sequence_length': 512  # Reduce sequence length
    }
})

# Process documents in smaller chunks
documents = [...]  # Large document list
chunk_size = 100

for i in range(0, len(documents), chunk_size):
    chunk = documents[i:i + chunk_size]
    rag.add_documents(chunk)
```

### Debug Mode

#### Python
```python
import logging
from rag_templates import RAG

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Create RAG with debug mode
rag = RAG(debug=True)

# All operations will now show detailed logs
rag.add_documents(["Test document"])
answer = rag.query("Test query")
```

#### JavaScript
```javascript
import { RAG } from '@rag-templates/core';

// Enable debug mode
const rag = new RAG(null, {debug: true});

// All operations will now show detailed logs
await rag.addDocuments(["Test document"]);
const answer = await rag.query("Test query");
```

## FAQ

### General Questions

**Q: What's the difference between Simple and Standard APIs?**

A: The Simple API provides zero-configuration RAG with string responses, perfect for prototypes. The Standard API offers technique selection, advanced configuration, and rich result objects for production use.

**Q: Can I use both Python and JavaScript APIs in the same project?**

A: Yes! The APIs are designed for interoperability. You can use Python for data processing and JavaScript for web interfaces, sharing the same IRIS database.

**Q: How do I migrate from the old complex setup to the new Simple API?**

A: See our [Migration Guide](MIGRATION_GUIDE.md) for step-by-step instructions and automated migration tools.

### Technical Questions

**Q: Which RAG technique should I choose?**

A: 
- **basic**: General purpose, fastest
- **colbert**: High precision, good for factual queries
- **hyde**: Complex reasoning, research applications
- **graphrag**: Structured knowledge, enterprise data
- **crag**: Self-correcting, accuracy-critical applications

**Q: How do I handle large document collections?**

A: Use batch processing and consider the Enterprise API:

```python
# Batch processing
for batch in document_batches:
    rag.add_documents(batch)
    
# Enterprise features
rag = ConfigurableRAG({
    'indexing': {
        'batch_size': 1000,
        'parallel_workers': 4,
        'incremental_updates': True
    }
})
```

**Q: Can I customize the embedding model?**

A: Yes, through configuration:

```python
rag = ConfigurableRAG({
    'embedding_model': 'sentence-transformers/all-mpnet-base-v2',
    'embedding_config': {
        'dimension': 768,
        'normalize': True
    }
})
```

**Q: How do I implement custom RAG techniques?**

A: The framework supports custom techniques:

```python
from rag_templates.core import BaseTechnique

class MyCustomTechnique(BaseTechnique):
    def retrieve(self, query, top_k=5):
        # Custom retrieval logic
        pass
    
    def generate(self, query, context):
        # Custom generation logic
        pass

# Register and use
rag = ConfigurableRAG({
    'technique': 'my_custom',
    'custom_techniques': {'my_custom': MyCustomTechnique}
})
```

### Performance Questions

**Q: How can I improve query performance?**

A: Several optimization strategies:

```python
rag = ConfigurableRAG({
    'caching': {'enabled': True, 'ttl': 3600},
    'embedding_config': {'cache_embeddings': True},
    'database': {'connection_pool_size': 10},
    'technique': 'basic'  # Fastest technique
})
```

**Q: What's the recommended setup for production?**

A: Use the Enterprise API with:
- Environment-based configuration
- Connection pooling
- Caching enabled
- Monitoring and logging
- Security features enabled

```python
# Production configuration
rag = ConfigurableRAG.from_config_file('production-config.yaml')
```

---

**Next Steps**: 
- [MCP Integration Guide](MCP_INTEGRATION_GUIDE.md) - Create MCP servers
- [API Reference](API_REFERENCE.md) - Complete API documentation
- [Migration Guide](MIGRATION_GUIDE.md) - Migrate from complex setup