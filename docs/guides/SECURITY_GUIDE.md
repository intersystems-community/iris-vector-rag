# RAG Templates Production Security Guide

## Table of Contents
1. [Authentication & Authorization](#authentication-authorization)
2. [Database Security (IRIS)](#database-security)
3. [API Key Management](#api-key-management)
4. [Network Security](#network-security)
5. [Data Encryption](#data-encryption)
6. [Input Validation](#input-validation)
7. [Audit Logging](#audit-logging)
8. [Compliance](#compliance)
9. [Incident Response](#incident-response)
10. [Security Testing](#security-testing)

---

## Authentication & Authorization

### Best Practices
- Use role-based access control (RBAC) for all services
- Implement multi-factor authentication (MFA) for admin access
- Store credentials in environment variables (never in code)
- Use OAuth2.0 for third-party service integrations

```bash
# Example environment variables
export DB_USER=$(uuidgen | cut -c1-8)
export DB_PASSWORD=$(openssl rand -base64 12)
```

---

## Database Security (InterSystems IRIS)

### IRIS-Specific Recommendations
1. **Secure Connection Strings**
   - Use TLS 1.2+ for all connections
   - Example configuration:
     ```python
     # From common/iris_connector.py
     def create_secure_connection():
         return iris.connect(
             f"{os.getenv('IRIS_HOST')}:1972/USER",
             os.getenv('IRIS_USER'),
             os.getenv('IRIS_PASSWORD'),
             timeout=10,
             ssl=True
         )
     ```

2. **Role Management**
   - Create least-privilege roles for each service
   - Example roles:
     ```sql
     -- From common/db_init_iris_compatible.sql
     CREATE ROLE rag_reader;
     GRANT SELECT ON %ALLIDS TO rag_reader;
     ```

3. **Audit Trail Configuration**
   ```sql
   SET ^%SYS("Audit",1,"Enabled")=1
   SET ^%SYS("Audit",1,"Events","SQL")=1
   ```

---

## API Key Management

### Secure Implementation
- Store keys in environment variables
- Rotate keys quarterly using:
  ```bash
  # Key rotation script pattern
  API_KEY=$(openssl rand -hex 16)
  echo "export OPENAI_API_KEY=$API_KEY" > .env
  chmod 600 .env
  ```

- Use API key middleware for request validation:
  ```python
  # From rag_templates/utils/migration.py
  def validate_api_key(request):
      provided_key = request.headers.get('X-API-Key')
      return provided_key == os.getenv('SERVICE_API_KEY')
  ```

---

## Network Security

### Firewall Configuration
- Restrict access to:
  ```bash
  # Example ufw rules
  ufw allow from 203.0.113.0/24 to any port 1972
  ufw deny from 198.51.100.0/24
  ```

- Use network segmentation:
  ```yaml
  # From docker-compose.iris-only.yml
  networks:
    rag_private:
      internal: true
      driver: bridge
```

---

## Data Encryption

### Implementation Requirements
- **At Rest**: AES-256 encryption for:
  ```sql
  -- From common/db_init_iris_compatible.sql
  SET ^%SYS("Config","Encryption","Enabled")=1
  ```

- **In Transit**: Enforce TLS:
  ```python
  # From common/iris_dbapi_connector.py
  def get_ssl_context():
      context = ssl.create_default_context()
      context.check_hostname = True
      context.verify_mode = ssl.CERT_REQUIRED
      return context
  ```

---

## Input Validation

### SQL Injection Prevention
- Use parameterized queries:
  ```python
  # From common/db_vector_search.py.pre_v2_update
  def safe_search(query):
      cursor.execute(
          "SELECT * FROM documents WHERE MATCH(:query)",
          {'query': query}
      )
  ```

- Sanitize all user inputs:
  ```python
  import re
  def sanitize_input(input_str):
      return re.sub(r'[;`\'"]', '', input_str)
  ```

---

## Audit Logging

### Implementation Strategy
- Enable comprehensive logging:
  ```python
  # From rag_templates/utils/migration.py
  logging.basicConfig(
      filename='security.log',
      level=logging.INFO,
      format='%(asctime)s [%(levelname)s] %(message)s'
  )
  ```

- Monitor for suspicious patterns:
  ```python
  def log_failed_attempt(user, ip):
      logging.warning(f"Failed access attempt by {user} from {ip}")
  ```

---

## Compliance Considerations

### GDPR/HIPAA Requirements
- Data anonymization patterns:
  ```python
  # From data/pmc_processor.py
  def anonymize_text(text):
      return re.sub(r'\b\d{4}[- ]?\d{6}\b', '[REDACTED]', text)
  ```

- Data retention policies:
  ```python
  def purge_old_records():
      cutoff = datetime.now() - timedelta(days=365)
      execute_sql(f"DELETE FROM logs WHERE timestamp < '{cutoff}'")
  ```

---

## Incident Response

### Response Procedure
1. Immediate containment:
   ```bash
   # Network isolation
   sudo iptables -A INPUT -j DROP
   ```

2. Forensic analysis:
   ```python
   # From reports/framework_test_results_*.json
   def analyze_logs(start_time, end_time):
       return [entry for entry in security_log 
               if start_time < entry['timestamp'] < end_time]
   ```

3. Notification workflow:
   ```python
   def notify_security_team(message):
       requests.post(
           os.getenv('SLACK_WEBHOOK_URL'),
           json={'text': message}
       )
   ```

---

## Security Testing

### Testing Procedures
- Penetration testing checklist:
  ```bash
  # Example nmap scan
  nmap -sV --script=vuln 203.0.113.45
  ```

- Code review focus areas:
  ```python
  # From tests/test_core/test_connection.py
  def test_secure_connection():
      with pytest.raises(ssl.SSLError):
          create_insecure_connection()
  ```

- Threat modeling considerations:
  ```mermaid
  graph TD
    A[Attack Surface] --> B[(API Endpoints)]
    A --> C[(Database)]
    A --> D[(File Storage)]