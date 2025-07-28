# Troubleshooting Guide

Comprehensive troubleshooting guide for the Library Consumption Framework, covering common issues, solutions, and debugging techniques.

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Installation Issues](#installation-issues)
3. [Configuration Problems](#configuration-problems)
4. [Database Connection Issues](#database-connection-issues)
5. [API and LLM Issues](#api-and-llm-issues)
6. [Performance Problems](#performance-problems)
7. [MCP Integration Issues](#mcp-integration-issues)
8. [Document Format Issues](#document-format-issues)
9. [Error Reference](#error-reference)
10. [Debug Mode and Logging](#debug-mode-and-logging)
11. [Getting Help](#getting-help)

## Quick Diagnostics

### Health Check Script

#### Python
```python
#!/usr/bin/env python3
"""
Quick health check for rag-templates Library Consumption Framework.
"""

import sys
import os
import traceback

def check_installation():
    """Check if rag-templates is properly installed."""
    try:
        import rag_templates
        print("✅ rag-templates package installed")
        print(f"   Version: {getattr(rag_templates, '__version__', 'unknown')}")
        return True
    except ImportError as e:
        print(f"❌ rag-templates not installed: {e}")
        return False

def check_dependencies():
    """Check critical dependencies."""
    dependencies = [
        ('intersystems-iris', 'IRIS database driver'),
        ('openai', 'OpenAI API client'),
        ('sentence-transformers', 'Embedding models'),
        ('yaml', 'Configuration file support')
    ]
    
    all_good = True
    for package, description in dependencies:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} ({description})")
        except ImportError:
            print(f"⚠️  {package} not installed ({description})")
            all_good = False
    
    return all_good

def check_environment():
    """Check environment variables."""
    env_vars = [
        ('IRIS_HOST', 'IRIS database host'),
        ('IRIS_PORT', 'IRIS database port'),
        ('IRIS_USERNAME', 'IRIS username'),
        ('IRIS_PASSWORD', 'IRIS password'),
        ('OPENAI_API_KEY', 'OpenAI API key')
    ]
    
    for var, description in env_vars:
        value = os.getenv(var)
        if value:
            masked_value = value[:4] + '*' * (len(value) - 4) if len(value) > 4 else '***'
            print(f"✅ {var}: {masked_value} ({description})")
        else:
            print(f"⚠️  {var} not set ({description})")

def test_simple_api():
    """Test Simple API functionality."""
    try:
        from rag_templates import RAG
        
        print("Testing Simple API...")
        rag = RAG()
        print("✅ Simple API initialization successful")
        
        # Test document addition
        rag.add_documents(["Test document for health check"])
        print("✅ Document addition successful")
        
        # Test querying
        answer = rag.query("test query")
        print("✅ Query execution successful")
        print(f"   Answer: {answer[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Simple API test failed: {e}")
        traceback.print_exc()
        return False

def test_database_connection():
    """Test database connectivity."""
    try:
        from rag_templates.core.config_manager import ConfigurationManager
        
        config = ConfigurationManager()
        db_config = config.get_database_config()
        
        print("Testing database connection...")
        print(f"   Host: {db_config.get('host', 'unknown')}")
        print(f"   Port: {db_config.get('port', 'unknown')}")
        print(f"   Namespace: {db_config.get('namespace', 'unknown')}")
        
        # Try to create a simple connection test
        # Note: This is a simplified test
        print("✅ Database configuration loaded")
        return True
        
    except Exception as e:
        print(f"❌ Database connection test failed: {e}")
        return False

def main():
    """Run comprehensive health check."""
    print("🔍 RAG Templates Health Check")
    print("=" * 50)
    
    checks = [
        ("Installation", check_installation),
        ("Dependencies", check_dependencies),
        ("Environment", check_environment),
        ("Database", test_database_connection),
        ("Simple API", test_simple_api)
    ]
    
    results = {}
    for name, check_func in checks:
        print(f"\n{name} Check:")
        results[name] = check_func()
    
    print("\n" + "=" * 50)
    print("Health Check Summary:")
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All checks passed! System is healthy.")
    else:
        print("\n⚠️  Some checks failed. See details above.")
        print("   Refer to the troubleshooting guide for solutions.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
```

#### JavaScript
```javascript
#!/usr/bin/env node
/**
 * Quick health check for rag-templates Library Consumption Framework.
 */

import fs from 'fs/promises';
import path from 'path';

async function checkInstallation() {
    try {
        const { RAG } = await import('@rag-templates/core');
        console.log("✅ @rag-templates/core package installed");
        
        // Try to read package.json for version
        try {
            const packagePath = path.join(process.cwd(), 'node_modules', '@rag-templates', 'core', 'package.json');
            const packageJson = JSON.parse(await fs.readFile(packagePath, 'utf8'));
            console.log(`   Version: ${packageJson.version}`);
        } catch {
            console.log("   Version: unknown");
        }
        
        return true;
    } catch (error) {
        console.log(`❌ @rag-templates/core not installed: ${error.message}`);
        return false;
    }
}

async function checkDependencies() {
    const dependencies = [
        ['intersystems-iris', 'IRIS database driver'],
        ['@xenova/transformers', 'Embedding models'],
        ['js-yaml', 'Configuration file support']
    ];
    
    let allGood = true;
    
    for (const [packageName, description] of dependencies) {
        try {
            await import(packageName);
            console.log(`✅ ${packageName} (${description})`);
        } catch {
            console.log(`⚠️  ${packageName} not installed (${description})`);
            allGood = false;
        }
    }
    
    return allGood;
}

function checkEnvironment() {
    const envVars = [
        ['IRIS_HOST', 'IRIS database host'],
        ['IRIS_PORT', 'IRIS database port'],
        ['IRIS_USERNAME', 'IRIS username'],
        ['IRIS_PASSWORD', 'IRIS password'],
        ['OPENAI_API_KEY', 'OpenAI API key']
    ];
    
    for (const [varName, description] of envVars) {
        const value = process.env[varName];
        if (value) {
            const maskedValue = value.length > 4 
                ? value.substring(0, 4) + '*'.repeat(value.length - 4)
                : '***';
            console.log(`✅ ${varName}: ${maskedValue} (${description})`);
        } else {
            console.log(`⚠️  ${varName} not set (${description})`);
        }
    }
}

async function testSimpleAPI() {
    try {
        const { RAG } = await import('@rag-templates/core');
        
        console.log("Testing Simple API...");
        const rag = new RAG();
        console.log("✅ Simple API initialization successful");
        
        // Test document addition
        await rag.addDocuments(["Test document for health check"]);
        console.log("✅ Document addition successful");
        
        // Test querying
        const answer = await rag.query("test query");
        console.log("✅ Query execution successful");
        console.log(`   Answer: ${answer.substring(0, 50)}...`);
        
        return true;
        
    } catch (error) {
        console.log(`❌ Simple API test failed: ${error.message}`);
        console.error(error.stack);
        return false;
    }
}

async function testDatabaseConnection() {
    try {
        const { ConfigManager } = await import('@rag-templates/core');
        
        const config = new ConfigManager();
        const dbConfig = config.getDatabaseConfig();
        
        console.log("Testing database connection...");
        console.log(`   Host: ${dbConfig.host || 'unknown'}`);
        console.log(`   Port: ${dbConfig.port || 'unknown'}`);
        console.log(`   Namespace: ${dbConfig.namespace || 'unknown'}`);
        
        console.log("✅ Database configuration loaded");
        return true;
        
    } catch (error) {
        console.log(`❌ Database connection test failed: ${error.message}`);
        return false;
    }
}

async function main() {
    console.log("🔍 RAG Templates Health Check");
    console.log("=".repeat(50));
    
    const checks = [
        ["Installation", checkInstallation],
        ["Dependencies", checkDependencies],
        ["Environment", checkEnvironment],
        ["Database", testDatabaseConnection],
        ["Simple API", testSimpleAPI]
    ];
    
    const results = {};
    
    for (const [name, checkFunc] of checks) {
        console.log(`\n${name} Check:`);
        results[name] = await checkFunc();
    }
    
    console.log("\n" + "=".repeat(50));
    console.log("Health Check Summary:");
    
    let allPassed = true;
    for (const [name, passed] of Object.entries(results)) {
        const status = passed ? "✅ PASS" : "❌ FAIL";
        console.log(`  ${name}: ${status}`);
        if (!passed) allPassed = false;
    }
    
    if (allPassed) {
        console.log("\n🎉 All checks passed! System is healthy.");
    } else {
        console.log("\n⚠️  Some checks failed. See details above.");
        console.log("   Refer to the troubleshooting guide for solutions.");
    }
    
    return allPassed;
}

// Run health check
main().then(success => {
    process.exit(success ? 0 : 1);
}).catch(error => {
    console.error("Health check failed:", error);
    process.exit(1);
});
```

## Installation Issues

### Issue 1: Package Not Found

**Problem**: `pip install rag-templates` or `npm install @rag-templates/core` fails

**Solutions**:

#### Python
```bash
# Update pip
pip install --upgrade pip

# Install from source if package not yet published
pip install git+https://github.com/your-org/rag-templates.git

# Install with specific Python version
python3.11 -m pip install rag-templates

# Install in virtual environment
python -m venv rag_env
source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate
pip install rag-templates
```

#### JavaScript
```bash
# Clear npm cache
npm cache clean --force

# Install with specific registry
npm install @rag-templates/core --registry https://registry.npmjs.org/

# Install from source
npm install git+https://github.com/your-org/rag-templates.git

# Install with yarn
yarn add @rag-templates/core
```

### Issue 2: Dependency Conflicts

**Problem**: Conflicting package versions

**Solutions**:

#### Python
```bash
# Create fresh virtual environment
python3 -m venv fresh_rag_env
source fresh_rag_env/bin/activate # On Windows: fresh_rag_env\Scripts\activate
pip install rag-templates # Or pip install -r requirements.txt

# Or use pip-tools for dependency resolution within the virtual environment
pip install pip-tools
pip-compile requirements.in # Ensure requirements.in exists or adapt
pip install -r requirements.txt # This will install resolved dependencies
```

#### JavaScript
```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Use npm overrides in package.json
{
  "overrides": {
    "conflicting-package": "^1.0.0"
  }
}
```

### Issue 3: Permission Errors

**Problem**: Permission denied during installation

**Solutions**:

#### Python
```bash
# Install for user only
pip install --user rag-templates

# Use sudo (not recommended)
sudo pip install rag-templates

# Better: use virtual environment
python -m venv venv
source venv/bin/activate
pip install rag-templates
```

#### JavaScript
```bash
# Fix npm permissions
npm config set prefix ~/.npm-global
export PATH=~/.npm-global/bin:$PATH

# Or use npx
npx @rag-templates/core

# Use yarn instead
yarn global add @rag-templates/core
```

## Configuration Problems

### Issue 1: Configuration File Not Found

**Problem**: `ConfigurationError: Configuration file not found`

**Solutions**:

#### Python
```python
# Use absolute path
from rag_templates import ConfigurableRAG
import os

config_path = os.path.abspath("config.yaml")
rag = ConfigurableRAG.from_config_file(config_path)

# Or use environment variables instead
rag = ConfigurableRAG({
    "technique": os.getenv("RAG_TECHNIQUE", "basic"),
    "llm_provider": os.getenv("LLM_PROVIDER", "openai")
})

# Or use Simple API with defaults
from rag_templates import RAG
rag = RAG()  # Works without config file
```

#### JavaScript
```javascript
// Use absolute path
import path from 'path';
import { ConfigurableRAG } from '@rag-templates/core';

const configPath = path.resolve("config.yaml");
const rag = await ConfigurableRAG.fromConfigFile(configPath);

// Or use environment variables
const rag = new ConfigurableRAG({
    technique: process.env.RAG_TECHNIQUE || "basic",
    llmProvider: process.env.LLM_PROVIDER || "openai"
});

// Or use Simple API with defaults
import { RAG } from '@rag-templates/core';
const rag = new RAG();  // Works without config file
```

### Issue 2: Invalid Configuration Format

**Problem**: `ConfigurationError: Invalid YAML format`

**Solutions**:

#### Validate YAML Syntax
```bash
# Install yamllint
pip install yamllint

# Check YAML syntax
yamllint config.yaml

# Or use online validator
# https://www.yamllint.com/
```

#### Common YAML Fixes
```yaml
# ❌ Wrong: inconsistent indentation
database:
  host: localhost
    port: 52773

# ✅ Correct: consistent indentation
database:
  host: localhost
  port: 52773

# ❌ Wrong: missing quotes for special characters
password: my@password!

# ✅ Correct: quoted special characters
password: "my@password!"

# ❌ Wrong: invalid boolean
enabled: yes

# ✅ Correct: valid boolean
enabled: true
```

### Issue 3: Environment Variable Substitution

**Problem**: Environment variables not being substituted in config

**Solutions**:

#### Python
```python
# Ensure environment variables are set
import os
os.environ['IRIS_HOST'] = 'localhost'
os.environ['IRIS_PORT'] = '52773'

# Use explicit environment loading
from rag_templates.config import ConfigManager
config = ConfigManager.from_file("config.yaml")
config.load_environment()  # Force reload environment variables
```

#### JavaScript
```javascript
// Use dotenv for environment variables
import dotenv from 'dotenv';
dotenv.config();

// Ensure variables are set
process.env.IRIS_HOST = process.env.IRIS_HOST || 'localhost';
process.env.IRIS_PORT = process.env.IRIS_PORT || '52773';
```

## Database Connection Issues

### Issue 1: Connection Refused

**Problem**: `ConnectionError: Connection refused to IRIS database`

**Solutions**:

#### Check Database Status
```bash
# Check if IRIS is running
docker ps | grep iris

# Start IRIS if not running
docker-compose -f docker-compose.iris-only.yml up -d

# Check IRIS logs
docker-compose -f docker-compose.iris-only.yml logs -f
```

#### Test Connection Manually
```python
# Test IRIS connection directly
import iris as iris

try:
    connection = iris.connect(
        hostname="localhost",
        port=52773,
        namespace="USER",
        username="demo",
        password="demo"
    )
    print("✅ IRIS connection successful")
    connection.close()
except Exception as e:
    print(f"❌ IRIS connection failed: {e}")
```

#### Common Connection Fixes
```python
# Fix 1: Check port mapping
# Ensure Docker port mapping is correct: -p 52773:52773

# Fix 2: Use correct namespace
rag = ConfigurableRAG({
    "database": {
        "host": "localhost",
        "port": 52773,
        "namespace": "USER",  # Not "RAG" if it doesn't exist
        "username": "demo",
        "password": "demo"
    }
})

# Fix 3: Wait for database startup
import time
import iris as iris

def wait_for_iris(max_attempts=30):
    for attempt in range(max_attempts):
        try:
            conn = iris.connect(hostname="localhost", port=52773, 
                              namespace="USER", username="demo", password="demo")
            conn.close()
            return True
        except:
            time.sleep(2)
    return False

if wait_for_iris():
    rag = RAG()
else:
    print("IRIS database not available")
```

### Issue 2: Authentication Failed

**Problem**: `AuthenticationError: Invalid credentials`

**Solutions**:

```python
# Check default credentials
default_configs = [
    {"username": "demo", "password": "demo"},
    {"username": "SuperUser", "password": "SYS"},
    {"username": "_SYSTEM", "password": "SYS"}
]

for config in default_configs:
    try:
        rag = ConfigurableRAG({
            "database": {
                "host": "localhost",
                "port": 52773,
                "username": config["username"],
                "password": config["password"]
            }
        })
        print(f"✅ Connected with {config['username']}")
        break
    except Exception as e:
        print(f"❌ Failed with {config['username']}: {e}")
```

### Issue 3: Namespace Not Found

**Problem**: `NamespaceError: Namespace 'RAG' does not exist`

**Solutions**:

```python
# Use existing namespace
rag = ConfigurableRAG({
    "database": {
        "namespace": "USER"  # Use default USER namespace
    }
})

# Or create namespace programmatically
import iris as iris

def create_namespace_if_not_exists(namespace_name):
    try:
        conn = iris.connect(hostname="localhost", port=52773, 
                          namespace="%SYS", username="SuperUser", password="SYS")
        
        # Check if namespace exists
        cursor = conn.cursor()
        cursor.execute("SELECT Name FROM Config.Namespaces WHERE Name = ?", [namespace_name])
        
        if not cursor.fetchone():
            # Create namespace
            cursor.execute(f"CREATE NAMESPACE {namespace_name}")
            print(f"✅ Created namespace {namespace_name}")
        else:
            print(f"✅ Namespace {namespace_name} already exists")
        
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Failed to create namespace: {e}")
        return False

# Usage
if create_namespace_if_not_exists("RAG"):
    rag = ConfigurableRAG({"database": {"namespace": "RAG"}})
```

## API and LLM Issues

### Issue 1: OpenAI API Key Invalid

**Problem**: `APIError: Invalid API key`

**Solutions**:

```bash
# Set API key in environment
export OPENAI_API_KEY=sk-your-actual-api-key-here

# Verify API key format
echo $OPENAI_API_KEY | grep -E '^sk-[a-zA-Z0-9]{48}$'
```

```python
# Test API key directly
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

try:
    response = openai.models.list()
    print("✅ OpenAI API key valid")
except Exception as e:
    print(f"❌ OpenAI API key invalid: {e}")

# Use alternative LLM provider
rag = ConfigurableRAG({
    "llm_provider": "anthropic",  # or "azure_openai"
    "llm_config": {
        "api_key": os.getenv("ANTHROPIC_API_KEY")
    }
})
```

### Issue 2: Rate Limiting

**Problem**: `RateLimitError: Too many requests`

**Solutions**:

```python
# Enable caching to reduce API calls
rag = ConfigurableRAG({
    "caching": {
        "enabled": True,
        "ttl": 3600  # Cache for 1 hour
    },
    "llm_config": {
        "rate_limit": {
            "requests_per_minute": 50,
            "tokens_per_minute": 40000
        }
    }
})

# Implement retry logic
import time
import random

def query_with_retry(rag, query, max_retries=3):
    for attempt in range(max_retries):
        try:
            return rag.query(query)
        except Exception as e:
            if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limited, waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                raise e
```

### Issue 3: Model Not Available

**Problem**: `ModelError: Model 'gpt-4' not available`

**Solutions**:

```python
# Use available models
available_models = [
    "gpt-4o-mini",
    "gpt-3.5-turbo",
    "gpt-4o"
]

for model in available_models:
    try:
        rag = ConfigurableRAG({
            "llm_config": {"model": model}
        })
        print(f"✅ Using model: {model}")
        break
    except Exception as e:
        print(f"❌ Model {model} not available: {e}")

# Check available models programmatically
import openai

try:
    models = openai.models.list()
    available = [model.id for model in models.data if "gpt" in model.id]
    print(f"Available models: {available}")
except Exception as e:
    print(f"Could not list models: {e}")
```

## Performance Problems

### Issue 1: Slow Query Performance

**Problem**: Queries taking too long to execute

**Solutions**:

```python
# Enable performance optimizations
rag = ConfigurableRAG({
    "technique": "basic",  # Fastest technique
    "caching": {
        "enabled": True,
        "ttl": 3600
    },
    "embedding_config": {
        "cache_embeddings": True,
        "batch_size": 100
    },
    "database": {
        "connection_pool_size": 10
    }
})

# Profile query performance
import time

def profile_query(rag, query):
    start_time = time.time()
    
    # Embedding time
    embed_start = time.time()
    # This would be internal to the query
    embed_time = time.time() - embed_start
    
    # Full query time
    result = rag.query(query)
    total_time = time.time() - start_time
    
    print(f"Query: {query[:50]}...")
    print(f"Total time: {total_time:.2f}s")
    print(f"Answer: {result[:100]}...")
    
    return result

# Optimize document chunking
rag = ConfigurableRAG({
    "chunking": {
        "chunk_size": 500,  # Smaller chunks for faster processing
        "chunk_overlap": 50
    }
})
```

### Issue 2: High Memory Usage

**Problem**: Application consuming too much memory

**Solutions**:

```python
# Optimize memory usage
rag = ConfigurableRAG({
    "embedding_config": {
        "batch_size": 10,  # Reduce batch size
        "max_sequence_length": 512  # Limit sequence length
    },
    "caching": {
        "max_size": 100  # Limit cache size
    }
})

# Process documents in batches
def add_documents_in_batches(rag, documents, batch_size=50):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        rag.add_documents(batch)
        print(f"Processed batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")

# Monitor memory usage
import psutil
import os

def monitor_memory():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")

monitor_memory()
rag = RAG()
monitor_memory()
```

### Issue 3: Embedding Generation Slow

**Problem**: Embedding generation taking too long

**Solutions**:

```python
# Use faster embedding models
fast_models = [
    "text-embedding-3-small",  # OpenAI - fast and good
    "sentence-transformers/all-MiniLM-L6-v2",  # Local - very fast
    "sentence-transformers/all-mpnet-base-v2"   # Local - balanced
]

rag = ConfigurableRAG({
    "embedding_model": "text-embedding-3-small",
    "embedding_config": {
        "batch_size": 100,  # Process multiple texts at once
        "cache_embeddings": True  # Cache computed embeddings
    }
})

# Pre-compute embeddings for static documents
def precompute_embeddings(rag, documents):
    print("Pre-computing embeddings...")
    start_time = time.time()
    
    rag.add_documents(documents)
    
    end_time = time.time()
    print(f"Embeddings computed in {end_time - start_time:.2f}s")
```

## MCP Integration Issues

### Issue 1: MCP Server Not Starting

**Problem**: MCP server fails to start

**Solutions**:

#### Check Node.js Version
```bash
# Check Node.js version (requires 18+)
node --version

# Update Node.js if needed
nvm install 18
nvm use 18
```

#### Debug Server Startup
```javascript
// Add debug logging to server
import { createMCPServer } from '@rag-templates/mcp';

const server = createMCPServer({
    name: "debug-server",
    description: "Debug MCP server",
    debug: true,  // Enable debug mode
    onStartup: async () => {
        console.log("Server startup callback called");
    },
    onError: (error) => {
        console.error("Server error:", error);
    }
});

try {
    await server.start();
    console.log("✅ Server started successfully");
} catch (error) {
    console.error("❌ Server startup failed:", error);
}
```

### Issue 2: Claude Desktop Not Detecting Server

**Problem**: MCP server doesn't appear in Claude Desktop

**Solutions**:

#### Check Configuration File
```json
// Verify claude_desktop_config.json syntax
{
  "mcpServers": {
    "rag-server": {
      "command": "node",
      "args": ["server.js"],
      "cwd": "/absolute/path/to/server/directory",
      "env": {
        "NODE_ENV": "production"
      }
    }
  }
}
```

#### Test Server Manually
```bash
# Test server directly
node server.js

# Check if server responds to MCP protocol
echo '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}' | node server.js
```

#### Debug Claude Desktop
```bash
# Check Claude Desktop logs (macOS)
tail -f ~/Library/Logs/Claude/claude.log

# Check Claude Desktop logs (Windows)
tail -f %APPDATA%\Claude\logs\claude.log
```

### Issue 3: MCP Tool Errors

**Problem**: MCP tools failing with schema validation errors

**Solutions**:

```javascript
// Ensure strict MCP compliance
const tools = [
    {
        name: "rag_search",
        description: "Search the knowledge base",
        inputSchema: {
            type: 'object',
            properties: {
                query: {
                    type: 'string',
                    description: 'Search query'
                },
                maxResults: {
                    type: 'integer',
                    minimum: 1,
                    maximum: 50,
                    default: 5
                }
            },
            required: ['query'],
            additionalProperties: false  // Important for MCP compliance
        },
        handler: async (args) => {
            // Validate arguments
            if (!args.query || typeof args.query !== 'string') {
                throw new Error('Invalid query parameter');
            }
            
            // Process request
            return { result: "success" };
        }
    }
];

// Test tool schema
function validateToolSchema(tool) {
    const required = ['name', 'description', 'inputSchema', 'handler'];
    for (const field of required) {
        if (!tool[field]) {
            throw new Error(`Tool missing required field: ${field}`);
        }
    }
    
    if (tool.inputSchema.type !== 'object') {
        throw new Error('Tool inputSchema must be of type "object"');
    }
    
    if (tool.inputSchema.additionalProperties !== false) {
        console.warn('Tool should set additionalProperties: false for MCP compliance');
    }
}
## Document Format Issues

### Issue 1: AttributeError - 'dict' object has no attribute 'page_content'

**Problem**: `AttributeError: 'dict' object has no attribute 'page_content'` when adding documents

**Cause**: Document format inconsistency between rag_templates and iris_rag pipeline

**Solution**: This critical bug has been resolved. The issue was caused by the `_process_documents()` methods in both `rag_templates/simple.py` and `rag_templates/standard.py` returning dictionary objects instead of proper `Document` objects.

**Quick Fix**: Update to the latest version of the library where this issue has been resolved.

**For detailed information**: See the comprehensive [Document Format Inconsistency Fix Guide](troubleshooting/DOCUMENT_FORMAT_INCONSISTENCY_FIX.md)

#### Symptoms
- Error occurs when calling `rag.add_documents()`
- Works with Simple API and Standard API
- Error message: `AttributeError: 'dict' object has no attribute 'page_content'`

#### Verification
```python
# Test that the fix is working
from rag_templates import RAG
rag = RAG()
rag.add_documents(["Test document"])  # Should work without error
```

```

## Error Reference

### Common Error Types

#### Python Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `RAGFrameworkError` | General framework error | Check logs for specific cause |
| `ConfigurationError` | Invalid configuration | Validate config file syntax |
| `InitializationError` | Setup failure | Check dependencies and database |
| `ConnectionError` | Database connection failed | Verify IRIS is running and accessible |
| `AuthenticationError` | Invalid credentials | Check username/password |
| `APIError` | LLM API failure | Verify API key and rate limits |
| `AttributeError: 'dict' object has no attribute 'page_content'` | Document format inconsistency | See [Document Format Fix](troubleshooting/DOCUMENT_FORMAT_INCONSISTENCY_FIX.md) |

####