# Framework Migration Guide

Migrate from LangChain, LlamaIndex, and other RAG frameworks to rag-templates with zero-configuration simplicity. **Special focus on IRIS customers with existing data.**

## Table of Contents

1. [Migration Overview](#migration-overview)
2. [IRIS Existing Data Migration](#iris-existing-data-migration)
3. [LangChain Migration](#langchain-migration)
4. [LlamaIndex Migration](#llamaindex-migration)
5. [LangGraph Migration](#langgraph-migration)
6. [Haystack Migration](#haystack-migration)
7. [Custom RAG Migration](#custom-rag-migration)
8. [Framework Comparison](#framework-comparison)
9. [Migration Tools](#migration-tools)

## Migration Overview

### Why Migrate to rag-templates?

| Feature | LangChain | LlamaIndex | rag-templates |
|---------|-----------|------------|---------------|
| **Setup Time** | 30+ min config | 20+ min setup | 30 seconds |
| **Lines of Code** | 50+ lines | 40+ lines | 3 lines |
| **Database** | Multiple configs | External setup | Built-in IRIS |
| **Vector Store** | Choose & config | Choose & config | Production-ready |
| **Enterprise Ready** | Custom setup | Custom setup | Built-in |
| **8 RAG Techniques** | Manual impl | Manual impl | One-line switch |
| **Existing IRIS Data** | Complex setup | Not supported | Native integration |

### Migration Benefits

- **Instant Productivity**: Start building in minutes, not hours
- **Zero Configuration**: Works immediately with production defaults
- **Enterprise Vector DB**: Built-in InterSystems IRIS with proven scalability
- **8 RAG Techniques**: Switch between techniques with one parameter
- **Production Ready**: Battle-tested in enterprise environments
- **Existing Data**: **Non-destructive integration with your current IRIS data**

## IRIS Existing Data Migration

### Customer Scenario: Healthcare System with Patient Data

Many IRIS customers already have valuable data in production databases and want to add RAG capabilities without disrupting existing systems.

#### Before: Complex Custom Integration
```python
# 100+ lines of complex integration code
import iris
from sentence_transformers import SentenceTransformer
import numpy as np
import openai

class CustomIRISRAG:
    def __init__(self, connection_string):
        self.connection = iris.connect(connection_string)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def create_rag_schema(self):
        """Manually create RAG tables - risky for production"""
        cursor = self.connection.cursor()
        
        # Create new tables (potential conflicts with existing schema)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rag_documents (
                id INTEGER IDENTITY,
                content VARCHAR(MAX),
                embedding VECTOR(DOUBLE, 384),
                source_table VARCHAR(100),
                source_id VARCHAR(50)
            )
        """)
        
        # Manual indexing
        cursor.execute("""
            CREATE INDEX embedding_idx ON rag_documents 
            USING VECTOR_COSINE(embedding)
        """)
        
    def extract_existing_data(self):
        """Manually extract from existing tables"""
        cursor = self.connection.cursor()
        
        # Extract patient records
        cursor.execute("""
            SELECT PatientID, FirstName, LastName, Diagnosis, Notes
            FROM Hospital.Patient
        """)
        
        patients = cursor.fetchall()
        
        for patient in patients:
            # Manual text assembly
            text = f"Patient {patient[1]} {patient[2]}: {patient[3]}. Notes: {patient[4]}"
            
            # Manual embedding generation
            embedding = self.model.encode(text).tolist()
            
            # Manual insertion
            cursor.execute("""
                INSERT INTO rag_documents (content, embedding, source_table, source_id)
                VALUES (?, VECTOR_FORMAT(?, 'LIST'), 'Hospital.Patient', ?)
            """, [text, embedding, patient[0]])
            
    def query_rag(self, question):
        """Manual RAG implementation"""
        # Generate query embedding
        query_embedding = self.model.encode(question).tolist()
        
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT TOP 5 content, VECTOR_COSINE(embedding, VECTOR_FORMAT(?, 'LIST')) as similarity
            FROM rag_documents
            ORDER BY similarity DESC
        """, [query_embedding])
        
        results = cursor.fetchall()
        context = "\n".join([r[0] for r in results])
        
        # Manual LLM call
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Answer based on patient data context"},
                {"role": "user", "content": f"Context: {context}\nQuestion: {question}"}
            ]
        )
        
        return response.choices[0].message.content

# Usage - risky and complex
rag = CustomIRISRAG("iris://localhost:1972/HEALTHCARE")
rag.create_rag_schema()  # Potential schema conflicts
rag.extract_existing_data()  # Manual data extraction
answer = rag.query_rag("What patients have diabetes complications?")
```

#### After: rag-templates with RAG Overlay

```python
# 5 lines - non-destructive integration
from rag_templates import ConfigurableRAG

# Option 1: Configuration-based integration
rag = ConfigurableRAG({
    "technique": "basic",
    "database": {
        "existing_tables": {
            "Hospital.Patient": {
                "content_fields": ["FirstName", "LastName", "Diagnosis", "Notes"],
                "id_field": "PatientID",
                "template": "Patient {FirstName} {LastName}: {Diagnosis}. Notes: {Notes}"
            }
        }
    }
})

# Automatically integrates existing data without schema changes
answer = rag.query("What patients have diabetes complications?")
```

**Or use the RAG Overlay System:**

```python
# Option 2: RAG Overlay System (Enterprise API)
from rag_templates.overlay import RAGOverlayInstaller
from rag_templates import ConfigurableRAG

# Install RAG overlay on existing database
installer = RAGOverlayInstaller("iris://localhost:1972/HEALTHCARE")
installer.install_overlay({
    "tables": ["Hospital.Patient", "Hospital.Diagnosis", "Hospital.Treatment"],
    "content_mapping": {
        "Hospital.Patient": {
            "content_template": "Patient {FirstName} {LastName}: {Diagnosis}. Notes: {Notes}",
            "metadata_fields": ["PatientID", "AdmissionDate", "Department"]
        }
    },
    "non_destructive": True  # No changes to existing schema
})

# Use with zero configuration
rag = ConfigurableRAG({"technique": "hybrid_ifind"})
answer = rag.query("What patients have diabetes complications?")
```

### Customer Scenario: Financial Services with Transaction Data

#### Before: Custom Integration
```python
# Complex manual integration with transaction data
class FinancialRAG:
    def extract_transactions(self):
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT t.TransactionID, t.Amount, t.Description,
                   c.CustomerName, c.AccountType,
                   m.MerchantName, m.Category
            FROM Banking.Transaction t
            JOIN Banking.Customer c ON t.CustomerID = c.CustomerID
            JOIN Banking.Merchant m ON t.MerchantID = m.MerchantID
            WHERE t.TransactionDate >= DATEADD(month, -12, GETDATE())
        """)
        
        transactions = cursor.fetchall()
        
        for txn in transactions:
            # Manual text construction
            text = f"Transaction {txn[0]}: ${txn[1]} at {txn[6]} ({txn[7]}). Customer: {txn[3]} ({txn[4]}). Description: {txn[2]}"
            
            # Manual embedding and storage
            embedding = self.model.encode(text).tolist()
            self.store_embedding(text, embedding, 'Banking.Transaction', txn[0])
```

#### After: rag-templates with Multi-Table Integration
```python
from rag_templates import ConfigurableRAG

rag = ConfigurableRAG({
    "technique": "sql_rag",  # SQL-aware RAG for relational data
    "database": {
        "existing_tables": {
            "Banking.Transaction": {
                "joins": [
                    "Banking.Customer ON Transaction.CustomerID = Customer.CustomerID",
                    "Banking.Merchant ON Transaction.MerchantID = Merchant.MerchantID"
                ],
                "content_template": "Transaction ${Amount} at {MerchantName} ({Category}). Customer: {CustomerName} ({AccountType}). {Description}",
                "filters": "TransactionDate >= DATEADD(month, -12, GETDATE())"
            }
        }
    }
})

answer = rag.query("Show me suspicious transaction patterns for high-value customers")
```

### Customer Scenario: Manufacturing with IoT Sensor Data

#### Before: Time-Series Data Integration Challenge
```python
# Complex IoT data integration
class ManufacturingRAG:
    def extract_sensor_data(self):
        """Extract and aggregate time-series sensor data"""
        cursor = self.connection.cursor()
        
        # Complex aggregation query
        cursor.execute("""
            SELECT 
                s.SensorID, s.SensorType, s.Location,
                AVG(r.Temperature) as AvgTemp,
                MAX(r.Pressure) as MaxPressure,
                COUNT(a.AlarmID) as AlarmCount,
                STRING_AGG(a.AlarmType, ', ') as AlarmTypes
            FROM Manufacturing.Sensor s
            LEFT JOIN Manufacturing.SensorReading r ON s.SensorID = r.SensorID
            LEFT JOIN Manufacturing.Alarm a ON s.SensorID = a.SensorID
            WHERE r.ReadingTime >= DATEADD(day, -30, GETDATE())
            GROUP BY s.SensorID, s.SensorType, s.Location
        """)
        
        sensor_data = cursor.fetchall()
        
        for sensor in sensor_data:
            # Manual aggregation and text creation
            text = f"Sensor {sensor[0]} ({sensor[1]}) at {sensor[2]}: Avg temp {sensor[3]}°C, Max pressure {sensor[4]} PSI. {sensor[5]} alarms: {sensor[6]}"
            
            # Manual processing...
```

#### After: rag-templates with Time-Series Aggregation
```python
from rag_templates import ConfigurableRAG

rag = ConfigurableRAG({
    "technique": "graphrag",  # Graph RAG for connected IoT data
    "database": {
        "existing_tables": {
            "Manufacturing.Sensor": {
                "aggregation": {
                    "time_window": "30 days",
                    "metrics": ["AVG(Temperature)", "MAX(Pressure)", "COUNT(Alarms)"],
                    "joins": [
                        "Manufacturing.SensorReading ON Sensor.SensorID = SensorReading.SensorID",
                        "Manufacturing.Alarm ON Sensor.SensorID = Alarm.SensorID"
                    ]
                },
                "content_template": "Sensor {SensorID} ({SensorType}) at {Location}: Avg temp {AvgTemp}°C, Max pressure {MaxPressure} PSI. {AlarmCount} alarms",
                "relationships": {
                    "location_hierarchy": "Location",
                    "sensor_network": "SensorType"
                }
            }
        }
    }
})

answer = rag.query("Which production line sensors show correlation between temperature spikes and quality issues?")
```

### Migration Benefits for IRIS Customers

#### Zero-Risk Integration
- **Non-destructive**: No changes to existing schema
- **Incremental**: Add RAG to one table at a time
- **Reversible**: Easy to remove RAG overlay if needed
- **Performance**: No impact on existing applications

#### Enterprise Features
- **Security**: Inherits existing IRIS security model
- **Scalability**: Uses existing IRIS clustering and scaling
- **Backup**: RAG data included in existing backup procedures
- **Monitoring**: Integrates with existing IRIS monitoring

#### ROI Acceleration
- **Immediate Value**: Query existing data in natural language
- **No Migration**: Leverage existing data investments
- **Reduced Development**: 95% less code vs custom solutions
- **Faster Time-to-Market**: Days instead of months

### Migration Process for IRIS Customers

#### Phase 1: Assessment (1 day)
```python
# Quick assessment of existing data
from rag_templates.assessment import DataSuitabilityAnalyzer

analyzer = DataSuitabilityAnalyzer("iris://your-connection")
report = analyzer.analyze_tables([
    "YourSchema.MainTable",
    "YourSchema.SecondaryTable"
])

print(f"RAG Suitability Score: {report.suitability_score}/10")
print(f"Recommended Technique: {report.recommended_technique}")
print(f"Estimated Setup Time: {report.setup_time}")
```

#### Phase 2: Pilot Implementation (1 day)
```python
# Start with one table
from rag_templates import ConfigurableRAG

pilot_rag = ConfigurableRAG({
    "technique": "basic",
    "database": {
        "existing_tables": {
            "YourSchema.MainTable": {
                "content_fields": ["TextField1", "TextField2"],
                "id_field": "ID"
            }
        }
    }
})

# Test queries
test_result = pilot_rag.query("Your domain-specific question")
```

#### Phase 3: Production Deployment (2-3 days)
```python
# Scale to multiple tables with advanced techniques
production_rag = ConfigurableRAG({
    "technique": "hybrid_ifind",  # Best for enterprise
    "database": {
        "existing_tables": {
            "Schema1.Table1": {...},
            "Schema2.Table2": {...},
            "Schema3.Table3": {...}
        },
        "performance": {
            "caching": True,
            "index_optimization": True,
            "batch_processing": True
        }
    }
})
```

## LangChain Migration

### Basic RAG Pipeline

#### Before: LangChain
```python
# 50+ lines of setup and configuration
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.schema import Document
import os

# Initialize components
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Setup vector store
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# Initialize LLM
llm = OpenAI(
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Create retrieval chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True
)

# Add documents
documents = [
    "Machine learning is a subset of AI...",
    "Deep learning uses neural networks..."
]

# Process and store documents
docs = [Document(page_content=text) for text in documents]
chunks = text_splitter.split_documents(docs)
vectorstore.add_documents(chunks)

# Query
result = qa_chain({"query": "What is machine learning?"})
answer = result["result"]
sources = result["source_documents"]
```

#### After: rag-templates
```python
# 3 lines - zero configuration
from rag_templates import RAG

rag = RAG()
rag.add_documents([
    "Machine learning is a subset of AI...",
    "Deep learning uses neural networks..."
])
answer = rag.query("What is machine learning?")
```

### Advanced RAG with Custom Embeddings

#### Before: LangChain
```python
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Custom embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector store with custom embeddings
vectorstore = FAISS.from_texts(
    texts=documents,
    embedding=embeddings
)

# Compression retriever
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever()
)

# Conversational chain with memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=compression_retriever,
    memory=memory
)

# Query with conversation history
result = qa({"question": "What is machine learning?"})
```

#### After: rag-templates
```python
from rag_templates import ConfigurableRAG

rag = ConfigurableRAG({
    "technique": "crag",  # Corrective RAG with compression
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "memory": True
})
rag.add_documents(documents)
result = rag.query("What is machine learning?", {
    "include_sources": True,
    "conversation_history": True
})
```

### Document Loading and Processing

#### Before: LangChain
```python
from langchain.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader, 
    DirectoryLoader, UnstructuredLoader
)
from langchain.text_splitter import CharacterTextSplitter

# Multiple loaders for different file types
pdf_loader = PyPDFLoader("document.pdf")
text_loader = TextLoader("document.txt")
csv_loader = CSVLoader("data.csv")

# Directory loading
directory_loader = DirectoryLoader(
    "./documents",
    glob="**/*.txt",
    loader_cls=TextLoader
)

# Load and split documents
all_documents = []
for loader in [pdf_loader, text_loader, csv_loader, directory_loader]:
    docs = loader.load()
    all_documents.extend(docs)

# Split documents
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(all_documents)

# Add to vector store
vectorstore.add_documents(chunks)
```

#### After: rag-templates
```python
# Built-in support for multiple file types
rag = RAG()
rag.load_from_directory("./documents", {
    "file_types": [".pdf", ".txt", ".csv", ".md"],
    "chunk_size": 1000,
    "chunk_overlap": 200
})
```

## LlamaIndex Migration

### Basic RAG Setup

#### Before: LlamaIndex
```python
# 40+ lines of configuration
from llama_index import (
    VectorStoreIndex, SimpleDirectoryReader, 
    ServiceContext, StorageContext
)
from llama_index.embeddings import OpenAIEmbedding
from llama_index.llms import OpenAI
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
import chromadb

# Configure LLM and embeddings
llm = OpenAI(model="gpt-4", temperature=0)
embedding = OpenAIEmbedding()

# Setup service context
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embedding,
    chunk_size=1000,
    chunk_overlap=200
)

# Configure vector store
chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection("documents")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Setup storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Load documents
documents = SimpleDirectoryReader("./documents").load_data()

# Create index
index = VectorStoreIndex.from_documents(
    documents,
    service_context=service_context,
    storage_context=storage_context
)

# Create query engine
query_engine = index.as_query_engine(
    similarity_top_k=5,
    response_mode="compact"
)

# Query
response = query_engine.query("What is machine learning?")
answer = str(response)
```

#### After: rag-templates
```python
from rag_templates import RAG

rag = RAG()
rag.load_from_directory("./documents")
answer = rag.query("What is machine learning?")
```

## Framework Comparison

### Feature Matrix

| Feature | LangChain | LlamaIndex | rag-templates |
|---------|-----------|------------|---------------|
| **Setup Complexity** | High | Medium | None |
| **IRIS Integration** | Manual | Not supported | Native |
| **Existing Data** | Complex | Not supported | RAG Overlay |
| **Production Ready** | Custom | Custom | Built-in |
| **8 RAG Techniques** | Manual | Manual | One parameter |
| **Enterprise Features** | Extensions | Custom | Built-in |

### Code Comparison

| Task | LangChain | LlamaIndex | rag-templates |
|------|-----------|------------|---------------|
| **Basic Setup** | 50+ lines | 40+ lines | 3 lines |
| **IRIS Integration** | 100+ lines | Not supported | 5 lines |
| **Existing Data RAG** | 200+ lines | Not supported | 3 lines |

## ObjectScript and Embedded Python Integration

### IRIS Customers: Native ObjectScript vs Embedded Python

IRIS customers have unique advantages with rag-templates through native ObjectScript integration and high-performance embedded Python capabilities.

#### Option 1: Pure ObjectScript Integration

```objectscript
/// Native ObjectScript RAG integration
Class YourApp.RAGService Extends %RegisteredObject
{

/// Invoke RAG techniques directly from ObjectScript
ClassMethod QueryRAG(query As %String, technique As %String = "basic") As %String
{
    // Use MCP bridge for ObjectScript -> Python RAG
    Set config = {"technique": (technique), "top_k": 5}
    Set configJSON = ##class(%ZEN.Auxiliary.jsonProvider).%ConvertJSONToObject(config)
    
    // Call Python RAG through embedded Python
    Set result = ##class(rag.templates).InvokeRAG(query, configJSON)
    
    Return result.answer
}

/// Batch process multiple queries
ClassMethod BatchQuery(queries As %List, technique As %String = "basic") As %List
{
    Set results = ##class(%ListOfDataTypes).%New()
    
    For i=1:1:queries.Count() {
        Set query = queries.GetAt(i)
        Set answer = ..QueryRAG(query, technique)
        Do results.Insert(answer)
    }
    
    Return results
}

/// Integration with existing IRIS business logic
ClassMethod PatientInsightQuery(patientID As %String, query As %String) As %String
{
    // Get patient context from existing IRIS tables
    &sql(SELECT FirstName, LastName, Diagnosis, Notes
         INTO :firstName, :lastName, :diagnosis, :notes
         FROM Hospital.Patient 
         WHERE PatientID = :patientID)
    
    // Enhance query with patient context
    Set enhancedQuery = query_" for patient "_firstName_" "_lastName_" with "_diagnosis
    
    // Use RAG with existing data integration
    Set answer = ..QueryRAG(enhancedQuery, "hybrid_ifind")
    
    Return answer
}

}
```

#### Option 2: Embedded Python with IRIS Performance

```python
# Embedded Python in IRIS - 2x faster than external Python
import iris
from rag_templates import ConfigurableRAG

class IRISEmbeddedRAG:
    def __init__(self):
        # Leverage IRIS embedded Python performance
        self.rag = ConfigurableRAG({
            "technique": "hybrid_ifind",
            "database": {"embedded_mode": True}  # Use IRIS embedded capabilities
        })
    
    def query_with_iris_data(self, query: str, patient_id: str = None):
        """Enhanced RAG with direct IRIS data access"""
        
        if patient_id:
            # Direct IRIS SQL through embedded Python
            rs = iris.sql.exec("""
                SELECT FirstName, LastName, Diagnosis, Notes, AdmissionDate
                FROM Hospital.Patient p
                JOIN Hospital.Admission a ON p.PatientID = a.PatientID
                WHERE p.PatientID = ?
                ORDER BY a.AdmissionDate DESC
            """, patient_id)
            
            # Build context from IRIS data
            context_parts = []
            for row in rs:
                context = f"Patient {row[0]} {row[1]}: {row[2]}. Notes: {row[3]} (Admitted: {row[4]})"
                context_parts.append(context)
            
            # Enhanced query with patient context
            enhanced_query = f"{query}\n\nPatient Context:\n" + "\n".join(context_parts)
            return self.rag.query(enhanced_query)
        
        return self.rag.query(query)
    
    def bulk_analysis(self, query_template: str):
        """Bulk analysis of all patients using IRIS performance"""
        
        # Efficient IRIS bulk query
        rs = iris.sql.exec("""
            SELECT PatientID, FirstName, LastName, Diagnosis
            FROM Hospital.Patient
            WHERE Diagnosis LIKE '%diabetes%'
        """)
        
        results = []
        for row in rs:
            patient_query = query_template.format(
                patient=f"{row[1]} {row[2]}",
                diagnosis=row[3]
            )
            answer = self.query_with_iris_data(patient_query, row[0])
            results.append({
                "patient_id": row[0],
                "query": patient_query,
                "answer": answer
            })
        
        return results

# Usage in IRIS embedded Python
rag_service = IRISEmbeddedRAG()
answer = rag_service.query_with_iris_data(
    "What are the latest treatment protocols?", 
    patient_id="12345"
)
```

#### Option 3: IRIS WSGI High-Performance Web Apps

IRIS's new WSGI facility provides **2x faster performance than Gunicorn** for Python web applications:

```python
# High-performance RAG web service using IRIS WSGI
from flask import Flask, request, jsonify
from rag_templates import ConfigurableRAG

app = Flask(__name__)

# Initialize RAG with IRIS embedded performance
rag = ConfigurableRAG({
    "technique": "colbert",
    "database": {
        "embedded_mode": True,  # Use IRIS embedded Python
        "performance_mode": "wsgi"  # Optimize for WSGI serving
    }
})

@app.route('/rag/query', methods=['POST'])
def rag_query():
    """High-performance RAG endpoint"""
    data = request.json
    query = data.get('query')
    technique = data.get('technique', 'basic')
    
    # Switch technique dynamically
    rag.configure({"technique": technique})
    
    # Direct IRIS data integration
    if 'patient_id' in data:
        # Embedded Python direct database access
        import iris
        rs = iris.sql.exec(
            "SELECT * FROM Hospital.Patient WHERE PatientID = ?", 
            data['patient_id']
        )
        patient_data = rs.fetchone()
        
        enhanced_query = f"{query}\nPatient: {patient_data[1]} {patient_data[2]}"
        result = rag.query(enhanced_query)
    else:
        result = rag.query(query)
    
    return jsonify({
        "answer": result.answer if hasattr(result, 'answer') else result,
        "technique": technique,
        "performance": "iris_wsgi_optimized"
    })

@app.route('/rag/techniques', methods=['GET'])
def list_techniques():
    """List available RAG techniques"""
    return jsonify({
        "techniques": ["basic", "colbert", "crag", "hyde", "graphrag", "hybrid_ifind", "noderag", "sql_rag"],
        "performance": "2x faster than gunicorn",
        "integration": "native_iris"
    })

# Deploy with IRIS WSGI (2x faster than external gunicorn)
if __name__ == '__main__':
    # IRIS automatically handles WSGI serving with superior performance
    app.run()
```

#### Deploy to IRIS WSGI:

```objectscript
/// Deploy Python RAG app to IRIS WSGI facility
Class YourApp.RAGWebService Extends %RegisteredObject
{

/// Configure WSGI application
ClassMethod SetupWSGI() As %Status
{
    // Configure IRIS WSGI for Python RAG app
    Set config = ##class(%Library.DynamicObject).%New()
    Do config.%Set("app_module", "rag_web_service")
    Do config.%Set("app_variable", "app")
    Do config.%Set("performance_mode", "high")
    Do config.%Set("embedded_python", 1)
    
    // Deploy to IRIS WSGI (2x faster than gunicorn)
    Set status = ##class(%SYS.Python.WSGI).Deploy("rag-api", config)
    
    Return status
}

/// Health check for RAG service
ClassMethod HealthCheck() As %String
{
    Set response = ##class(%Net.HttpRequest).%New()
    Do response.Get("http://localhost:52773/rag-api/health")
    
    Return response.HttpResponse.Data.Read()
}

}
```

### Performance Comparison: IRIS vs External Solutions

| Deployment Method | Performance | Setup Complexity | IRIS Integration |
|-------------------|-------------|------------------|------------------|
| **IRIS WSGI** | **2x faster than Gunicorn** | **Minimal** | **Native** |
| **IRIS Embedded Python** | **Native speed** | **Zero** | **Direct** |
| **ObjectScript Integration** | **Maximum** | **Native** | **Seamless** |
| External Gunicorn | Baseline | High | API calls |
| External Flask | Baseline | High | API calls |
| Docker Deployment | Container overhead | Very High | Network calls |

### Migration Paths for IRIS Customers

#### Path 1: Start with Embedded Python (Recommended)
```python
# Immediate value with existing data
from rag_templates import ConfigurableRAG

rag = ConfigurableRAG({
    "database": {"embedded_mode": True},
    "existing_tables": {"YourSchema.YourTable": {...}}
})

answer = rag.query("Your domain question")
```

#### Path 2: Add ObjectScript Integration
```objectscript
// Call from existing ObjectScript applications
Set answer = ##class(YourApp.RAGService).QueryRAG("Your question", "colbert")
```

#### Path 3: Deploy High-Performance Web Service
```python
# 2x faster than external solutions
# Deploy Python RAG app with IRIS WSGI facility
# Automatic embedded Python optimization
```

### Key Advantages for IRIS Customers

1. **Performance**: 2x faster than external solutions with IRIS WSGI
2. **Integration**: Native ObjectScript and embedded Python
3. **Security**: Inherits IRIS security model and access controls
4. **Scalability**: Leverages IRIS clustering and high availability
5. **Operations**: Single system to manage, monitor, and backup
6. **Cost**: No additional infrastructure or licensing required

## Migration Tools

### IRIS Customer Assessment Tool

```python
from rag_templates.assessment import IRISCustomerAnalyzer

# Analyze existing IRIS database for RAG potential
analyzer = IRISCustomerAnalyzer("iris://your-connection")
assessment = analyzer.full_assessment()

print(f"Tables suitable for RAG: {len(assessment.suitable_tables)}")
print(f"Estimated ROI: {assessment.roi_estimate}")
print(f"Recommended migration path: {assessment.migration_strategy}")
print(f"ObjectScript integration potential: {assessment.objectscript_readiness}")
print(f"WSGI deployment benefits: {assessment.wsgi_performance_gain}")
```

**The migration to rag-templates is especially powerful for IRIS customers because it provides immediate value from existing data investments with zero risk, minimal effort, and maximum performance through native IRIS capabilities.**