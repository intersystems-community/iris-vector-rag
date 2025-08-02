# Integrating RAG with Existing Data

This guide explains how to add RAG capabilities to existing InterSystems IRIS databases and tables without modifying your original data or schema.

## Table of Contents

1. [Overview](#overview)
2. [Configuration-Based Table Mapping](#configuration-based-table-mapping)
3. [RAG Overlay System (Non-Destructive)](#rag-overlay-system-non-destructive)
4. [Field Mapping Requirements](#field-mapping-requirements)
5. [Examples](#examples)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

## Overview

RAG Templates provides two approaches for integrating with existing data:

1. **Configuration-Based Mapping**: Use existing tables directly by configuring table names
2. **RAG Overlay System**: Create views and auxiliary tables that expose existing data in RAG format

Both approaches preserve your original data and schema integrity.

## Configuration-Based Table Mapping

### Simple Table Name Configuration

The easiest way to use existing tables is to configure the table name in your RAG configuration:

```yaml
# config.yaml
storage:
  iris:
    table_name: "MyCompany.Documents"  # Your existing table
```

### Python Usage

Both storage classes support custom table names:

#### Enterprise API (Manual Schema Control)
```python
from iris_rag.storage.enterprise_storage import IRISStorage
from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.connection import ConnectionManager

# Load config with custom table name
config = ConfigurationManager("config.yaml")
connection = ConnectionManager(config)

# Enterprise storage with full control
storage = IRISStorage(connection, config)

# Add missing columns to existing table
storage.initialize_schema()  # Adds doc_id, metadata, embedding columns if missing
```

#### Standard API (LangChain Compatible)
```python
from iris_rag.storage.vector_store_iris import IRISVectorStore

# Standard storage with LangChain compatibility
vector_store = IRISVectorStore(connection, config)

# Works with existing table automatically
documents = vector_store.similarity_search("query", k=5)
```

### Required Schema Compatibility

Your existing table needs these minimum requirements:

**Required Fields:**
- **Text content field**: Contains the main document text
- **Unique ID field**: Primary key or unique identifier

**Optional Fields (will be added if missing):**
- `doc_id VARCHAR(255)`: Document identifier (maps to your ID field)
- `metadata VARCHAR(MAX)`: JSON metadata storage
- `embedding VECTOR(FLOAT, dimension)`: Vector embeddings

## RAG Overlay System (Non-Destructive)

For complex scenarios or when you cannot modify existing tables, use the RAG Overlay System.

### How It Works

1. **Discovers** existing tables with text content
2. **Creates views** that map your schema to RAG format
3. **Preserves** original data completely
4. **Adds** only necessary auxiliary tables for embeddings

### Overlay Configuration

Create an overlay configuration file:

```yaml
# overlay_config.yaml
source_tables:
  - name: "CustomerDocs.Documents"
    id_field: "document_id"           # Maps to doc_id
    title_field: "title"
    content_field: "content"          # Main text content
    metadata_fields: ["author", "created_date", "category"]
    enabled: true
  
  - name: "KnowledgeBase.Articles"
    id_field: "article_id"
    title_field: "article_title"
    content_field: "full_text"
    metadata_fields: ["topic", "last_updated"]
    enabled: true

rag_schema: "RAG"
view_prefix: "RAG_Overlay_"
embedding_table: "RAG.OverlayEmbeddings"
ifind_table: "RAG.OverlayIFindIndex"
```

### Running the Overlay Installer

```bash
# Install overlay system
python scripts/rag_overlay_installer.py --config overlay_config.yaml

# Or use programmatically
```

```python
from scripts.rag_overlay_installer import RAGOverlayInstaller

# Install RAG overlay
installer = RAGOverlayInstaller("overlay_config.yaml")

# Discover existing tables automatically
discovered = installer.discover_existing_tables()
print(f"Found {len(discovered)} tables with text content")

# Create overlay views and tables
installer.create_overlay_views()
installer.create_overlay_embedding_table()
installer.create_overlay_ifind_table()
installer.create_unified_rag_view()
```

### What Gets Created

The overlay system creates:

1. **Views** (one per source table):
   ```sql
   CREATE VIEW RAG.RAG_Overlay_CustomerDocs_Documents AS
   SELECT 
       document_id as doc_id,
       title as title,
       content as text_content,
       -- ... standard RAG schema mapping
   FROM CustomerDocs.Documents
   ```

2. **Embedding Table** (stores computed embeddings):
   ```sql
   CREATE TABLE RAG.OverlayEmbeddings (
       doc_id VARCHAR(255) PRIMARY KEY,
       source_table VARCHAR(255),
       embedding VARCHAR(32000),
       created_at TIMESTAMP
   )
   ```

3. **IFind Table** (for keyword search):
   ```sql
   CREATE TABLE RAG.OverlayIFindIndex (
       doc_id VARCHAR(255) PRIMARY KEY,
       source_table VARCHAR(255),
       text_content LONGVARCHAR
   )
   ```

## Field Mapping Requirements

### Required Fields

| RAG Schema | Your Field | Purpose |
|------------|------------|---------|
| `doc_id` | Any unique ID | Document identifier |
| `text_content` | Any text field | Main content for search |

### Optional Fields

| RAG Schema | Your Field | Purpose | Default if Missing |
|------------|------------|---------|-------------------|
| `title` | Title/Name field | Document title | Empty string |
| `metadata` | JSON or multiple fields | Searchable metadata | Auto-generated JSON |
| `embedding` | N/A | Vector embeddings | Generated automatically |

### Field Type Compatibility

| Your Field Type | RAG Schema Type | Notes |
|-----------------|-----------------|-------|
| `VARCHAR`, `LONGVARCHAR` | `text_content` | ✅ Direct mapping |
| `INTEGER`, `BIGINT` | `doc_id` | ✅ Converted to string |
| `JSON`, `VARCHAR` | `metadata` | ✅ Parsed or wrapped |
| `TIMESTAMP`, `DATE` | `metadata` | ✅ Included in JSON |

## Examples

### Example 1: Simple Customer Documents

**Your existing table:**
```sql
CREATE TABLE Sales.CustomerDocuments (
    id INTEGER PRIMARY KEY,
    customer_name VARCHAR(255),
    document_text LONGVARCHAR,
    upload_date TIMESTAMP
)
```

**Configuration:**
```yaml
storage:
  iris:
    table_name: "Sales.CustomerDocuments"
```

**Usage:**
```python
# The system automatically maps:
# id -> doc_id
# document_text -> text_content  
# customer_name, upload_date -> metadata

from iris_rag.storage.vector_store_iris import IRISVectorStore

vector_store = IRISVectorStore(connection, config)
results = vector_store.similarity_search("contract terms", k=5)
```

### Example 2: Complex Multi-Table Setup

**Your existing tables:**
```sql
-- Table 1: Product documentation
CREATE TABLE Products.Documentation (
    product_id VARCHAR(50) PRIMARY KEY,
    product_name VARCHAR(255),
    documentation TEXT,
    version VARCHAR(20),
    last_updated TIMESTAMP
)

-- Table 2: Support tickets
CREATE TABLE Support.Tickets (
    ticket_id INTEGER PRIMARY KEY,
    subject VARCHAR(500),
    description LONGVARCHAR,
    resolution LONGVARCHAR,
    category VARCHAR(100)
)
```

**Overlay configuration:**
```yaml
source_tables:
  - name: "Products.Documentation"
    id_field: "product_id"
    title_field: "product_name"
    content_field: "documentation"
    metadata_fields: ["version", "last_updated"]
    enabled: true
    
  - name: "Support.Tickets"
    id_field: "ticket_id"
    title_field: "subject"
    content_field: "description"  # Could combine with resolution
    metadata_fields: ["category", "resolution"]
    enabled: true
```

**Usage:**
```python
# After overlay installation, query across all sources
from iris_rag.storage.vector_store_iris import IRISVectorStore

# Configure to use the unified overlay view
config_data = {
    "storage": {
        "iris": {
            "table_name": "RAG.UnifiedOverlayView"
        }
    }
}

vector_store = IRISVectorStore(connection, config)
results = vector_store.similarity_search("product installation issues", k=10)

# Results will include both product docs and support tickets
for doc in results:
    print(f"Source: {doc.metadata['source_table']}")
    print(f"Content: {doc.page_content}")
```

## Best Practices

### 1. Data Preparation

- **Clean text content**: Ensure text fields don't contain binary data
- **Consistent encoding**: Use UTF-8 encoding for text content
- **Reasonable size limits**: Very large documents may need chunking

### 2. Performance Optimization

```yaml
# Configure appropriate vector dimensions
storage:
  iris:
    vector_dimension: 384  # Match your embedding model

# Use appropriate chunking for large documents
chunking:
  enabled: true
  chunk_size: 1000
  chunk_overlap: 200
```

### 3. Security Considerations

- **Field mapping**: Only expose necessary fields to RAG system
- **Access control**: Use IRIS security features on source tables
- **Data sensitivity**: Consider which fields to include in metadata

### 4. Monitoring and Maintenance

```python
# Check overlay health
installer = RAGOverlayInstaller("config.yaml")
discovered = installer.discover_existing_tables()

# Monitor embedding generation progress
from iris_rag.storage.vector_store_iris import IRISVectorStore
vector_store = IRISVectorStore(connection, config)
doc_count = vector_store.get_document_count()
print(f"Indexed {doc_count} documents")
```

## Troubleshooting

### Common Issues

**1. "Table not found" errors**
```python
# Verify table name and schema
config_manager = ConfigurationManager()
table_name = config_manager.get("storage:iris:table_name")
print(f"Looking for table: {table_name}")

# Check table exists
connection = get_iris_connection()
cursor = connection.cursor()
cursor.execute("SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = ?", [table_name])
```

**2. "Column not found" errors**
```sql
-- Check your table schema
DESCRIBE YourSchema.YourTable

-- Or use information schema
SELECT COLUMN_NAME, DATA_TYPE 
FROM INFORMATION_SCHEMA.COLUMNS 
WHERE TABLE_NAME = 'YourTable'
```

**3. "No embeddings generated"**
```python
# Check embedding table
cursor.execute("SELECT COUNT(*) FROM RAG.OverlayEmbeddings")
embedding_count = cursor.fetchone()[0]

if embedding_count == 0:
    # Trigger embedding generation
    vector_store = IRISVectorStore(connection, config)
    # Add documents to trigger embedding generation
```

### Performance Issues

**Large table scanning:**
```yaml
# Add indexes to your source tables
# CREATE INDEX idx_content ON YourTable (text_content)
# CREATE INDEX idx_updated ON YourTable (last_updated)
```

**Slow embedding generation:**
```yaml
# Configure batch processing
embeddings:
  batch_size: 32  # Reduce if memory constrained
  
# Use appropriate model
embedding_model:
  name: "all-MiniLM-L6-v2"  # Faster, smaller model
  dimension: 384
```

### Configuration Validation

```python
# Validate configuration before deployment
def validate_overlay_config(config_path):
    installer = RAGOverlayInstaller(config_path)
    
    for table_config in installer.config["source_tables"]:
        table_name = table_config["name"]
        
        # Check table exists
        try:
            cursor.execute(f"SELECT 1 FROM {table_name} LIMIT 1")
            print(f"✅ Table {table_name} accessible")
        except Exception as e:
            print(f"❌ Table {table_name} error: {e}")
            
        # Check required fields exist
        required_fields = ["id_field", "content_field"]
        for field in required_fields:
            if not table_config.get(field):
                print(f"❌ Missing required field: {field}")
```

## Migration from Legacy Systems

If you're migrating from other RAG systems or databases:

1. **Map your existing schema** to RAG requirements
2. **Use overlay system** for gradual migration
3. **Test with subset** of data first
4. **Validate results** against your existing system
5. **Gradually expand** to full dataset

The overlay system allows you to run both systems in parallel during migration, ensuring zero downtime and data safety.

---

For more information, see:
- [Configuration Guide](CONFIGURATION.md)
- [API Reference](API_REFERENCE.md) 
- [Developer Guide](DEVELOPER_GUIDE.md)