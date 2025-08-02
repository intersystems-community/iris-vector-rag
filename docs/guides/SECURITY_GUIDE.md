# RAG Templates Production Security Guide

## Table of Contents
1. [Configuration Security](#configuration-security)
2. [Database Security (IRIS)](#database-security)
3. [SQL Injection Prevention](#sql-injection-prevention)
4. [API Key Management](#api-key-management)
5. [LLM & AI Security](#llm-ai-security)
6. [Vector Database Security](#vector-database-security)
7. [Network Security](#network-security)
8. [Data Encryption](#data-encryption)
9. [Input Validation](#input-validation)
10. [Dependency Security](#dependency-security)
11. [Audit Logging](#audit-logging)
12. [Compliance](#compliance)
13. [Incident Response](#incident-response)
14. [Security Testing](#security-testing)

---

## Configuration Security

### YAML Configuration Protection
The project uses [`iris_rag/config/manager.py`](iris_rag/config/manager.py) for configuration management with environment variable overrides:

```python
# Secure configuration loading from config/config.yaml
# Environment variables override YAML with prefix mapping:
# RAG_DATABASE__IRIS__HOST overrides database.iris.host
# RAG_EMBEDDING__OPENAI__API_KEY overrides embedding.openai.api_key

# Example secure environment setup:
export RAG_DATABASE__IRIS__HOST="secure-iris-host.internal"
export RAG_DATABASE__IRIS__PASSWORD="$(openssl rand -base64 32)"
export RAG_EMBEDDING__OPENAI__API_KEY="sk-..."
```

### Configuration File Security
```bash
# Secure config file permissions
chmod 600 config/config.yaml
chown app:app config/config.yaml

# Never commit sensitive values to version control
echo "config/config.yaml" >> .gitignore
```

---

## Database Security (InterSystems IRIS)

### Secure Connection Management
The [`common/iris_connector.py`](common/iris_connector.py) implements secure IRIS connections:

```python
# From common/iris_connector.py - secure connection pattern
def create_secure_connection():
    return iris.connect(
        f"{config.database.iris.host}:{config.database.iris.port}/{config.database.iris.namespace}",
        config.database.iris.username,
        config.database.iris.password,
        timeout=30,
        ssl=True  # Always use TLS
    )
```

### IRIS-Specific Security Configuration
```sql
-- Enable audit logging
SET ^%SYS("Audit",1,"Enabled")=1
SET ^%SYS("Audit",1,"Events","SQL")=1
SET ^%SYS("Audit",1,"Events","Login")=1

-- Configure encryption at rest
SET ^%SYS("Config","Encryption","Enabled")=1

-- Create least-privilege roles
CREATE ROLE rag_reader;
GRANT SELECT ON RAG.* TO rag_reader;

CREATE ROLE rag_writer;
GRANT SELECT, INSERT, UPDATE ON RAG.* TO rag_writer;
```

---

## SQL Injection Prevention

### Comprehensive Parameterized Query Implementation
The codebase implements extensive SQL injection defenses using DBAPI/JDBC parameterized queries throughout:

#### Core Connection Manager
[`common/connection_manager.py`](common/connection_manager.py:92-95) provides secure query execution:

```python
# All queries use parameterized execution
def execute(self, query: str, params: Optional[List[Any]] = None):
    cursor = self._connection.cursor()
    if params:
        cursor.execute(query, params)  # Always parameterized
    else:
        cursor.execute(query)
```

#### Vector Operations Security
[`common/db_vector_utils.py`](common/db_vector_utils.py:73) ensures secure vector insertions:

```python
# Secure vector insertion with parameterized queries
def insert_vector(cursor, table_name, vector_column_name, embedding_str, 
                 other_column_names, other_column_values):
    placeholders_list = ["?" for _ in other_column_names] + ["TO_VECTOR(?, FLOAT)"]
    sql_query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
    params = other_column_values + [embedding_str]
    cursor.execute(sql_query, params)  # Parameterized execution
```

#### Pipeline-Level Protection
All RAG pipelines use parameterized queries:

```python
# Example from iris_rag/pipelines/colbert.py
cursor.execute("""
    SELECT doc_id, VECTOR_COSINE(token_embedding, TO_VECTOR(?)) as similarity
    FROM RAG.DocumentTokenEmbeddings 
    WHERE VECTOR_COSINE(token_embedding, TO_VECTOR(?)) > ?
    ORDER BY similarity DESC
""", [query_vector_str, query_vector_str, similarity_threshold])
```

#### Batch Operations Security
```python
# Secure batch insertions using executemany
cursor.executemany(sql_query, batch_params)  # From data/loader_*.py
```

### Vector SQL Limitations Handling
[`common/vector_sql_utils.py`](common/vector_sql_utils.py:22-24) documents IRIS vector operation limitations and provides safe string interpolation when parameterization isn't possible:

```python
# When IRIS vector functions don't support parameterization,
# use validated string interpolation with input sanitization
def validate_vector_input(vector_str):
    # Strict validation before string interpolation
    if not re.match(r'^[\d\.,\-\s\[\]]+$', vector_str):
        raise ValueError("Invalid vector format")
    return vector_str
```

---

## API Key Management

### Environment-Based Key Management
```bash
# Secure API key configuration
export RAG_EMBEDDING__OPENAI__API_KEY="sk-..."
export RAG_EMBEDDING__ANTHROPIC__API_KEY="sk-ant-..."

# Key rotation script
#!/bin/bash
NEW_KEY=$(openssl rand -hex 32)
echo "export RAG_SERVICE_API_KEY=$NEW_KEY" >> .env.new
mv .env.new .env
chmod 600 .env
```

### API Key Validation Middleware
```python
# Secure API key validation pattern
def validate_api_key(request_headers):
    provided_key = request_headers.get('X-API-Key')
    expected_key = os.getenv('RAG_SERVICE_API_KEY')
    return hmac.compare_digest(provided_key or '', expected_key or '')
```

---

## LLM & AI Security

### Prompt Injection Prevention
```python
# Input sanitization for LLM queries
def sanitize_llm_input(user_query):
    # Remove potential prompt injection patterns
    dangerous_patterns = [
        r'ignore\s+previous\s+instructions',
        r'system\s*:',
        r'assistant\s*:',
        r'<\s*script\s*>',
    ]
    
    sanitized = user_query
    for pattern in dangerous_patterns:
        sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
    
    return sanitized[:1000]  # Limit length
```

### LLM Response Validation
```python
# Validate LLM responses before returning to users
def validate_llm_response(response):
    # Check for potential data leakage
    if re.search(r'\b(api[_-]?key|password|secret)\b', response, re.IGNORECASE):
        return "Response filtered for security reasons"
    
    return response
```

### Model Security Configuration
```python
# Secure LLM configuration
llm_config = {
    'temperature': 0.1,  # Reduce randomness for consistent behavior
    'max_tokens': 500,   # Limit response length
    'top_p': 0.9,       # Control response diversity
    'frequency_penalty': 0.1,  # Reduce repetition
}
```

---

## Vector Database Security

### Embedding Security
```python
# Secure embedding generation and storage
def secure_embedding_pipeline(text_content):
    # Sanitize input before embedding
    sanitized_text = re.sub(r'[^\w\s\-\.]', '', text_content)
    
    # Generate embedding with error handling
    try:
        embedding = embedding_function(sanitized_text)
        # Validate embedding dimensions
        if len(embedding) != expected_dimension:
            raise ValueError("Invalid embedding dimension")
        return embedding
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return None
```

### Vector Search Security
```python
# Secure vector similarity search
def secure_vector_search(query_embedding, top_k=10):
    # Validate inputs
    if not isinstance(query_embedding, list) or len(query_embedding) != 768:
        raise ValueError("Invalid query embedding")
    
    if top_k > 100:  # Prevent resource exhaustion
        top_k = 100
    
    # Use parameterized query
    cursor.execute("""
        SELECT TOP ? doc_id, content, 
               VECTOR_COSINE(embedding, TO_VECTOR(?)) as similarity
        FROM RAG.SourceDocuments 
        WHERE VECTOR_COSINE(embedding, TO_VECTOR(?)) > 0.7
        ORDER BY similarity DESC
    """, [top_k, json.dumps(query_embedding), json.dumps(query_embedding)])
```

---

## Network Security

### Firewall Configuration
```bash
# Restrict IRIS database access
ufw allow from 10.0.0.0/8 to any port 1972
ufw allow from 172.16.0.0/12 to any port 1972
ufw allow from 192.168.0.0/16 to any port 1972
ufw deny from any to any port 1972

# API endpoint protection
ufw allow from trusted_subnet to any port 8000
ufw limit ssh
```

### Network Segmentation
```yaml
# Docker network isolation
networks:
  rag_internal:
    internal: true
    driver: bridge
  rag_external:
    driver: bridge

services:
  iris:
    networks:
      - rag_internal
  
  api:
    networks:
      - rag_internal
      - rag_external
```

---

## Data Encryption

### Encryption at Rest
```sql
-- IRIS encryption configuration
SET ^%SYS("Config","Encryption","Enabled")=1
SET ^%SYS("Config","Encryption","Algorithm")="AES256"
```

### Encryption in Transit
```python
# TLS configuration for all connections
import ssl

def create_secure_ssl_context():
    context = ssl.create_default_context()
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    return context
```

### Sensitive Data Handling
```python
# Secure handling of sensitive document content
def process_sensitive_document(content):
    # Redact PII patterns
    pii_patterns = [
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
        r'\b\d{4}[- ]?\d{6}\b',    # Credit card
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email
    ]
    
    processed_content = content
    for pattern in pii_patterns:
        processed_content = re.sub(pattern, '[REDACTED]', processed_content)
    
    return processed_content
```

---

## Input Validation

### Comprehensive Input Sanitization
```python
# Multi-layer input validation
def validate_user_input(user_input):
    # Length validation
    if len(user_input) > 10000:
        raise ValueError("Input too long")
    
    # Character validation
    if not re.match(r'^[\w\s\-\.\,\?\!]+$', user_input):
        raise ValueError("Invalid characters in input")
    
    # SQL injection pattern detection
    sql_patterns = [
        r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)\b)',
        r'(--|#|/\*|\*/)',
        r'(\bUNION\b|\bOR\b.*=.*\bOR\b)',
    ]
    
    for pattern in sql_patterns:
        if re.search(pattern, user_input, re.IGNORECASE):
            raise ValueError("Potentially malicious input detected")
    
    return user_input.strip()
```

### File Upload Security
```python
# Secure file processing
def validate_uploaded_file(file_path):
    # File type validation
    allowed_extensions = {'.txt', '.pdf', '.docx', '.xml'}
    if not any(file_path.endswith(ext) for ext in allowed_extensions):
        raise ValueError("File type not allowed")
    
    # File size validation
    if os.path.getsize(file_path) > 50 * 1024 * 1024:  # 50MB limit
        raise ValueError("File too large")
    
    # Content validation
    with open(file_path, 'rb') as f:
        header = f.read(1024)
        if b'<script' in header.lower():
            raise ValueError("Potentially malicious file content")
```

---

## Dependency Security

### Requirements Management
```bash
# Regular security updates
pip install --upgrade pip
pip-audit --fix
safety check

# Pin dependencies with known good versions
pip freeze > requirements.lock
```

### Vulnerability Scanning
```python
# Automated dependency checking
def check_dependencies():
    import subprocess
    import json
    
    # Run safety check
    result = subprocess.run(['safety', 'check', '--json'], 
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        vulnerabilities = json.loads(result.stdout)
        logger.error(f"Security vulnerabilities found: {vulnerabilities}")
        return False
    
    return True
```

---

## Audit Logging

### Comprehensive Security Logging
```python
# Security event logging
import logging
from datetime import datetime

security_logger = logging.getLogger('security')
security_logger.setLevel(logging.INFO)

handler = logging.FileHandler('/var/log/rag-security.log')
formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] %(message)s [%(filename)s:%(lineno)d]'
)
handler.setFormatter(formatter)
security_logger.addHandler(handler)

def log_security_event(event_type, details, user_id=None, ip_address=None):
    security_logger.info(f"SECURITY_EVENT: {event_type} | "
                        f"User: {user_id} | IP: {ip_address} | "
                        f"Details: {details}")
```

### Database Access Logging
```python
# Log all database operations
def log_database_access(operation, table, user, query_hash):
    security_logger.info(f"DB_ACCESS: {operation} on {table} by {user} "
                        f"(query_hash: {query_hash})")
```

---

## Compliance

### GDPR Compliance
```python
# Data anonymization for GDPR
def anonymize_personal_data(text):
    # Remove personal identifiers
    anonymized = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME]', text)
    anonymized = re.sub(r'\b\d{1,2}/\d{1,2}/\d{4}\b', '[DATE]', anonymized)
    anonymized = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', anonymized)
    return anonymized

# Data retention policy
def enforce_data_retention():
    cutoff_date = datetime.now() - timedelta(days=365)
    cursor.execute("""
        DELETE FROM RAG.AuditLogs 
        WHERE created_date < ?
    """, [cutoff_date])
```

### HIPAA Compliance
```python
# Healthcare data protection
def protect_health_information(content):
    # Remove medical record numbers
    content = re.sub(r'\bMRN\s*:?\s*\d+\b', '[MRN_REDACTED]', content)
    
    # Remove dates of birth
    content = re.sub(r'\bDOB\s*:?\s*\d{1,2}/\d{1,2}/\d{4}\b', 
                    '[DOB_REDACTED]', content)
    
    return content
```

---

## Incident Response

### Automated Threat Detection
```python
# Real-time threat monitoring
def monitor_suspicious_activity():
    # Monitor failed login attempts
    failed_attempts = get_failed_logins_last_hour()
    if failed_attempts > 10:
        alert_security_team("High number of failed logins detected")
    
    # Monitor unusual query patterns
    unusual_queries = detect_unusual_sql_patterns()
    if unusual_queries:
        alert_security_team(f"Unusual SQL patterns detected: {unusual_queries}")

def alert_security_team(message):
    # Send immediate notification
    requests.post(
        os.getenv('SECURITY_WEBHOOK_URL'),
        json={
            'text': f"🚨 SECURITY ALERT: {message}",
            'timestamp': datetime.now().isoformat()
        }
    )
```

### Incident Containment
```python
# Automated incident response
def contain_security_incident(incident_type):
    if incident_type == "sql_injection_attempt":
        # Block suspicious IP
        block_ip_address(get_client_ip())
        
        # Disable affected user account
        disable_user_account(get_current_user())
        
        # Create forensic snapshot
        create_database_snapshot()
    
    elif incident_type == "data_breach":
        # Immediate data access lockdown
        revoke_all_active_sessions()
        
        # Notify compliance team
        notify_compliance_team()
```

---

## Security Testing

### Automated Security Testing
```python
# Security test suite
def test_sql_injection_protection():
    """Test SQL injection prevention"""
    malicious_inputs = [
        "'; DROP TABLE RAG.SourceDocuments; --",
        "1' OR '1'='1",
        "UNION SELECT * FROM RAG.SourceDocuments",
    ]
    
    for malicious_input in malicious_inputs:
        with pytest.raises(ValueError):
            validate_user_input(malicious_input)

def test_parameterized_queries():
    """Verify all database operations use parameterized queries"""
    # Test vector insertion
    cursor = get_test_cursor()
    insert_vector(cursor, "test_table", "embedding", 
                 "[0.1, 0.2, 0.3]", ["doc_id"], ["test_doc"])
    
    # Verify no SQL injection possible
    assert cursor.last_query_used_parameters

def test_api_key_validation():
    """Test API key security"""
    # Test with invalid key
    assert not validate_api_key({'X-API-Key': 'invalid'})
    
    # Test with valid key
    os.environ['RAG_SERVICE_API_KEY'] = 'valid_key'
    assert validate_api_key({'X-API-Key': 'valid_key'})
```

### Penetration Testing Checklist
```bash
# Network security testing
nmap -sV --script=vuln target_host

# Web application testing
sqlmap -u "http://target/api/search" --data="query=test"

# SSL/TLS testing
testssl.sh target_host:443

# Database security testing
iris_security_scanner --host target_iris --port 1972
```

### Security Code Review
```python
# Automated code security scanning
def security_code_review():
    # Check for hardcoded secrets
    secret_patterns = [
        r'password\s*=\s*["\'][^"\']+["\']',
        r'api[_-]?key\s*=\s*["\'][^"\']+["\']',
        r'secret\s*=\s*["\'][^"\']+["\']',
    ]
    
    for file_path in get_python_files():
        with open(file_path, 'r') as f:
            content = f.read()
            for pattern in secret_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    raise SecurityError(f"Hardcoded secret found in {file_path}")
```

---

## Security Monitoring Dashboard

### Key Security Metrics
```python
# Security metrics collection
def collect_security_metrics():
    return {
        'failed_logins_24h': count_failed_logins(hours=24),
        'sql_injection_attempts': count_sql_injection_attempts(),
        'api_key_violations': count_api_key_violations(),
        'unusual_queries': count_unusual_queries(),
        'data_access_violations': count_data_access_violations(),
        'encryption_status': check_encryption_status(),
        'vulnerability_count': count_known_vulnerabilities(),
    }
```

### Automated Security Reports
```python
# Daily security report generation
def generate_security_report():
    metrics = collect_security_metrics()
    
    report = f"""
    RAG Templates Security Report - {datetime.now().strftime('%Y-%m-%d')}
    
    🔒 Authentication Security:
    - Failed logins (24h): {metrics['failed_logins_24h']}
    - API key violations: {metrics['api_key_violations']}
    
    🛡️ Database Security:
    - SQL injection attempts: {metrics['sql_injection_attempts']}
    - Unusual queries detected: {metrics['unusual_queries']}
    - Encryption status: {metrics['encryption_status']}
    
    📊 System Security:
    - Known vulnerabilities: {metrics['vulnerability_count']}
    - Data access violations: {metrics['data_access_violations']}
    """
    
    send_security_report(report)
```

This comprehensive security guide reflects the actual implementation patterns in the RAG templates codebase, focusing on the extensive DBAPI/JDBC parameterized query usage and real security measures implemented throughout the system.