# RAG Templates Configuration System Refactoring Strategy

## Executive Summary

This document defines the concrete strategy for the initial phase of configuration refactoring to address the identified architectural fragmentation, 181 hard-coding instances, and schema inconsistencies in the RAG Templates project.

## 1. Standard Configuration Loader

### 1.1 Recommended Standard: [`iris_rag/config/manager.py`](iris_rag/config/manager.py:10)

**Decision:** The [`ConfigurationManager`](iris_rag/config/manager.py:10) in `iris_rag/config/manager.py` should be adopted as the project-wide standard configuration loader.

**Justification:**
- **Mature Implementation**: Already supports YAML loading, environment variable overrides, and nested key access
- **Environment Variable Integration**: Implements the `RAG_` prefix with `__` delimiter pattern for nested configuration
- **Type Casting**: Includes intelligent type casting for environment variable overrides
- **Extensible Design**: Supports schema validation framework (placeholder implemented)
- **Clean API**: Provides intuitive `get()` method with colon-delimited keys (`"database:iris:host"`)

### 1.2 Configuration File Migration Strategy

**Current State Analysis:**
- [`config/config.yaml`](config/config.yaml:1): Legacy format with flat database keys (`db_host`, `db_port`)
- [`config/default.yaml`](config/default.yaml:1): Modern nested format (`database.iris.host`)
- [`iris_rag/config/default_config.yaml`](iris_rag/config/default_config.yaml:1): Iris-specific configuration

**Migration Plan:**
1. **Phase 1**: Standardize on [`config/default.yaml`](config/default.yaml:1) schema structure
2. **Phase 2**: Migrate [`config/config.yaml`](config/config.yaml:1) to nested format
3. **Phase 3**: Consolidate [`iris_rag/config/default_config.yaml`](iris_rag/config/default_config.yaml:1) into main configuration

## 2. Prioritized Refactoring List

### 2.1 Critical Database Connection Files (Priority 1)

1. **[`check_tables.py`](check_tables.py:13)** - Direct hardcoded connection
   ```python
   # Current: Hardcoded values
   connection = iris.connect(
       hostname="localhost",
       port=1972,
       namespace="USER",
       username="_SYSTEM",
       password="SYS"
   )
   ```

2. **[`common/iris_connector.py`](common/iris_connector.py:43)** - Mixed hardcoded/environment approach
   ```python
   # Current: Environment fallbacks to hardcoded values
   conn_params_dict = {
       "hostname": os.environ.get("IRIS_HOST", "localhost"),
       "port": int(os.environ.get("IRIS_PORT", "1972")),
       # ...
   }
   ```

3. **[`check_columns.py`](check_columns.py:13)** - Identical hardcoded pattern
4. **[`test_basic_rag_retrieval.py`](test_basic_rag_retrieval.py:22)** - Test infrastructure hardcoding
5. **[`eval/test_iris_connect.py`](eval/test_iris_connect.py:11)** - Evaluation hardcoding

### 2.2 Secondary Priority Files (Priority 2)

6. **[`scripts/test_community_schema.py`](scripts/test_community_schema.py:21)** - Script-level hardcoding
7. **[`jdbc_exploration/iris_jdbc_connector.py`](jdbc_exploration/iris_jdbc_connector.py:18)** - JDBC connector defaults
8. **[`eval/config_manager.py`](eval/config_manager.py:31)** - Evaluation-specific config manager
9. **[`common/iris_dbapi_connector.py`](common/iris_dbapi_connector.py:89)** - DBAPI connector
10. **[`scripts/cleanup_doc_ids.py`](scripts/cleanup_doc_ids.py:307)** - Administrative script

## 3. Unified Configuration Schema

### 3.1 Database Connection Schema

```yaml
# Unified Database Configuration Schema
database:
  iris:
    # Connection Parameters
    driver: "intersystems_iris.dbapi._DBAPI"
    host: "localhost"
    port: 1972
    namespace: "USER"
    username: "_SYSTEM"
    password: "SYS"
    
    # Connection Management
    connection_timeout: 30
    max_retries: 3
    retry_delay: 1
    
    # Pool Configuration (future)
    pool_size: 5
    max_overflow: 10
    
  # Common Database Settings
  common:
    timeout: 30
    charset: "utf8"
    autocommit: false
```

### 3.2 Schema Compatibility

The unified schema maintains backward compatibility with existing [`ConfigurationManager`](iris_rag/config/manager.py:10) access patterns:

```python
# Access patterns supported
config_manager.get("database:iris:host")        # "localhost"
config_manager.get("database:iris:port")        # 1972
config_manager.get("database:common:timeout")   # 30
```

## 4. Environment Variable Override Strategy

### 4.1 Naming Convention

**Pattern**: `RAG_<SECTION>__<SUBSECTION>__<KEY>`

**Database Examples**:
```bash
# Database connection overrides
export RAG_DATABASE__IRIS__HOST="production.iris.com"
export RAG_DATABASE__IRIS__PORT="1972"
export RAG_DATABASE__IRIS__NAMESPACE="PROD"
export RAG_DATABASE__IRIS__USERNAME="prod_user"
export RAG_DATABASE__IRIS__PASSWORD="secure_password"

# Common database settings
export RAG_DATABASE__COMMON__TIMEOUT="60"
```

### 4.2 Sensitive Data Handling

**Security Guidelines**:
1. **Never commit passwords** to configuration files
2. **Always use environment variables** for sensitive data
3. **Provide secure defaults** in configuration files
4. **Document required environment variables** in deployment guides

**Implementation**:
```yaml
# config/default.yaml - Safe defaults
database:
  iris:
    host: "localhost"
    port: 1972
    namespace: "USER"
    username: "_SYSTEM"
    password: "CHANGE_ME"  # Clear indicator for required override
```

```bash
# Production environment
export RAG_DATABASE__IRIS__PASSWORD="actual_secure_password"
export RAG_DATABASE__IRIS__HOST="production.iris.internal"
```

### 4.3 Type Casting Support

The [`ConfigurationManager`](iris_rag/config/manager.py:80) already implements intelligent type casting:

```python
# Automatic type conversion
export RAG_DATABASE__IRIS__PORT="1972"        # → int(1972)
export RAG_DATABASE__COMMON__TIMEOUT="30"     # → int(30)
export RAG_MONITORING__ENABLED="true"         # → bool(True)
```

## 5. Implementation Phases

### Phase 1: Core Infrastructure (Week 1)
1. Standardize [`ConfigurationManager`](iris_rag/config/manager.py:10) as project-wide standard
2. Create unified configuration schema
3. Refactor top 3 critical files ([`check_tables.py`](check_tables.py:13), [`common/iris_connector.py`](common/iris_connector.py:43), [`check_columns.py`](check_columns.py:13))

### Phase 2: Test Infrastructure (Week 2)
4. Refactor test files to use [`ConfigurationManager`](iris_rag/config/manager.py:10)
5. Update evaluation scripts
6. Implement configuration validation

### Phase 3: Scripts and Tools (Week 3)
7. Refactor administrative scripts
8. Update JDBC exploration tools
9. Consolidate configuration files

## 6. Success Metrics

### 6.1 Quantitative Goals
- **Reduce hardcoded instances** from 181 to <20
- **Eliminate direct database credentials** in source code
- **Standardize on single configuration manager** across all modules

### 6.2 Quality Indicators
- **All database connections** use [`ConfigurationManager`](iris_rag/config/manager.py:10)
- **Environment variable overrides** work consistently
- **Configuration schema** is documented and validated
- **Sensitive data** is properly externalized

## 7. Risk Mitigation

### 7.1 Backward Compatibility
- Maintain existing environment variable names during transition
- Provide configuration migration utilities
- Document breaking changes clearly

### 7.2 Testing Strategy
- Create comprehensive configuration tests
- Validate environment variable overrides
- Test with multiple deployment scenarios

## 8. Next Steps

1. **Immediate**: Begin Phase 1 implementation with [`check_tables.py`](check_tables.py:13) refactoring
2. **Short-term**: Establish TDD workflow for configuration changes
3. **Medium-term**: Implement configuration validation and schema enforcement
4. **Long-term**: Consider configuration hot-reloading and dynamic updates

---

**Document Version**: 1.0  
**Last Updated**: 2025-06-09  
**Next Review**: After Phase 1 completion