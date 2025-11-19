# Configuration Examples

This directory contains example configuration files for deploying iris-vector-rag in different environments.

## Available Examples

### 1. Local Development (`local.yaml`)

For local development with Docker or native IRIS installation.

**Features:**
- Localhost connection
- Default credentials (_SYSTEM/SYS)
- 384-dimensional vectors (SentenceTransformers)
- Debug logging
- Fast HNSW indexing

**Usage:**
```bash
# With Docker
docker-compose up -d
python -m iris_vector_rag.cli.init_tables --config config/examples/local.yaml

# With environment variables
export IRIS_HOST="localhost"
export IRIS_PORT="1974"
export VECTOR_DIMENSION="384"
python -m iris_vector_rag.cli.init_tables
```

### 2. AWS Cloud Deployment (`aws.yaml`)

For AWS-hosted InterSystems IRIS instances.

**Features:**
- %SYS namespace (required for schema creation on AWS)
- SQLUser schema prefix (AWS requirement)
- 1024-dimensional vectors (NVIDIA NIM)
- Environment variable support for secrets

**Usage:**
```bash
# Set AWS credentials
export IRIS_HOST="aws-iris.your-domain.com"
export IRIS_USERNAME="AppUser"
export IRIS_PASSWORD="YourSecurePassword"
export IRIS_NAMESPACE="%SYS"
export VECTOR_DIMENSION="1024"
export TABLE_SCHEMA="SQLUser"

# Initialize tables
python -m iris_vector_rag.cli.init_tables --config config/examples/aws.yaml
```

### 3. Azure Cloud Deployment (`azure.yaml`)

For Azure-hosted InterSystems IRIS instances.

**Features:**
- USER namespace (Azure standard)
- RAG schema prefix
- 1536-dimensional vectors (OpenAI ada-002)
- Azure Key Vault integration support

**Usage:**
```bash
# Set Azure credentials (use Azure Key Vault)
export IRIS_HOST="azure-iris.cloudapp.azure.com"
export IRIS_USERNAME="AzureAppUser"
export IRIS_PASSWORD=$(az keyvault secret show --name iris-password --vault-name your-vault --query value -o tsv)
export VECTOR_DIMENSION="1536"

# Initialize tables
python -m iris_vector_rag.cli.init_tables --config config/examples/azure.yaml
```

## Configuration Priority

Configuration values are resolved in this order (highest to lowest priority):

1. **Environment Variables** (`IRIS_HOST`, `VECTOR_DIMENSION`, etc.)
2. **Configuration File** (`--config config/examples/aws.yaml`)
3. **Defaults** (localhost:1972, 384 dimensions, RAG schema)

This follows the [12-factor app](https://12factor.net/config) methodology for cloud-native applications.

## Common Vector Dimensions

Choose vector dimensions based on your embedding model:

| Model | Dimensions | Use Case |
|-------|-----------|----------|
| SentenceTransformers (all-MiniLM-L6-v2) | 384 | Development, lightweight |
| SentenceTransformers (all-mpnet-base-v2) | 768 | Balanced performance |
| NVIDIA NIM, Cohere embed-v3.0 | 1024 | AWS deployments |
| OpenAI text-embedding-ada-002 | 1536 | Azure/OpenAI |
| OpenAI text-embedding-3-large | 3072 | High accuracy |

## Security Best Practices

### Never Commit Secrets

❌ **Bad:**
```yaml
database:
  iris:
    password: "MySecretPassword123"  # DON'T DO THIS!
```

✅ **Good:**
```yaml
database:
  iris:
    password: ${IRIS_PASSWORD}  # Reads from environment
```

### Use Cloud Secret Management

**AWS:**
```bash
# Store in AWS Secrets Manager
aws secretsmanager create-secret \
  --name iris-password \
  --secret-string "YourSecurePassword"

# Retrieve in code
export IRIS_PASSWORD=$(aws secretsmanager get-secret-value \
  --secret-id iris-password \
  --query SecretString \
  --output text)
```

**Azure:**
```bash
# Store in Azure Key Vault
az keyvault secret set \
  --vault-name your-vault \
  --name iris-password \
  --value "YourSecurePassword"

# Retrieve in code
export IRIS_PASSWORD=$(az keyvault secret show \
  --vault-name your-vault \
  --name iris-password \
  --query value \
  --output tsv)
```

**Docker:**
```bash
# Use Docker secrets
echo "YourSecurePassword" | docker secret create iris_password -

# Reference in docker-compose.yml
services:
  app:
    secrets:
      - iris_password
    environment:
      IRIS_PASSWORD_FILE: /run/secrets/iris_password
```

## Namespace Requirements

### AWS IRIS
- **Namespace:** `%SYS` (required for CREATE TABLE permissions)
- **Schema:** `SQLUser` (required for table isolation)
- **Full table name:** `SQLUser.Entities`

### Azure IRIS
- **Namespace:** `USER` (standard for Azure)
- **Schema:** `RAG` (flexible, can be customized)
- **Full table name:** `RAG.Entities`

### Local Development
- **Namespace:** `USER` (default)
- **Schema:** `RAG` (default)
- **Full table name:** `RAG.Entities`

## Troubleshooting

### Port Already in Use

```bash
# Check what's using port 1972/1974
lsof -i :1972
lsof -i :1974

# Stop conflicting services
docker-compose down
```

### Vector Dimension Mismatch

```
ConfigValidationError: Vector dimension mismatch detected
Configured dimension: 1024
Existing table dimension: 384
```

**Solution 1: Match existing tables**
```bash
export VECTOR_DIMENSION="384"
```

**Solution 2: Recreate tables (⚠️ deletes all data)**
```bash
python -m iris_vector_rag.cli.init_tables --drop --config config/examples/aws.yaml
```

### Namespace Permission Denied

```
PermissionError: User lacks CREATE TABLE permission in namespace %SYS
```

**Solution:** Contact your IRIS administrator for CREATE TABLE permissions, or use a namespace where you have permissions (e.g., USER).

## Testing Configuration

Test your configuration before deploying:

```bash
# Test connection only
python -c "
from iris_vector_rag.config.manager import ConfigurationManager
config = ConfigurationManager(config_path='config/examples/aws.yaml')
cloud_config = config.get_cloud_config()
print(f'Host: {cloud_config.connection.host}')
print(f'Namespace: {cloud_config.connection.namespace}')
print(f'Vector dimension: {cloud_config.vector.vector_dimension}')
print(f'Schema: {cloud_config.tables.table_schema}')
"

# Test with preflight validation
python -m iris_vector_rag.cli.init_tables --config config/examples/aws.yaml --dry-run
```

## More Information

- [Feature Specification](../../specs/058-number-058-short/spec.md)
- [Implementation Plan](../../specs/058-number-058-short/plan.md)
- [Quickstart Guide](../../specs/058-number-058-short/quickstart.md)
- [Vector Dimension Migration](../../docs/migration/vector-dimensions.md)
- [Namespace Configuration](../../docs/configuration/namespace-permissions.md)
