# Technical Environment Information

## Environment Information

| Component | Version/Details |
|-----------|----------------|
| IRIS Version | IRIS for UNIX (Ubuntu Server LTS for ARM64 Containers) 2024.1.2 (Build 398U) Thu Oct 3 2024 14:29:04 EDT |
| Python Version | 3.12.9 |
| Operating System | Darwin 24.3.0 |
| Platform | macOS-15.3.2-arm64-arm-64bit |

## Client Library Versions

| Library | Version |
|---------|--------|
| pyodbc | Not installed |
| sqlalchemy | 2.0.41 |
| sqlalchemy-iris | Unknown |
| langchain-iris | Unknown |
| llama-iris | Unknown |

## SQL Query Test Results

### Direct SQL

**Success**: No

**Query:**
```sql

        SELECT id, VECTOR_COSINE(
            TO_VECTOR(embedding, 'DOUBLE', 5),
            TO_VECTOR('0.1,0.2,0.3,0.4,0.5', 'DOUBLE', 5)
        ) AS score
        FROM TechnicalInfoTest
        
```

**Error:**
```
[SQLCODE: <-1>:<Invalid SQL statement>]
[Location: <Prepare>]
[%msg: < ) expected, : found ^SELECT id , VECTOR_COSINE ( TO_VECTOR ( embedding , :%qpar>]
```

### Parameterized SQL

**Success**: No

**Query:**
```sql

        SELECT id, VECTOR_COSINE(
            TO_VECTOR(embedding, 'DOUBLE', 5),
            TO_VECTOR(?, 'DOUBLE', 5)
        ) AS score
        FROM TechnicalInfoTest
        
```

**Error:**
```
[SQLCODE: <-1>:<Invalid SQL statement>]
[Location: <Prepare>]
[%msg: < ) expected, : found ^SELECT id , VECTOR_COSINE ( TO_VECTOR ( embedding , :%qpar>]
```

### String Interpolation

**Success**: No

**Query:**
```sql

        SELECT id, VECTOR_COSINE(
            TO_VECTOR(embedding, 'DOUBLE', 5),
            TO_VECTOR('0.1,0.2,0.3,0.4,0.5', 'DOUBLE', 5)
        ) AS score
        FROM TechnicalInfoTest
        
```

**Error:**
```
[SQLCODE: <-1>:<Invalid SQL statement>]
[Location: <Prepare>]
[%msg: < ) expected, : found ^SELECT id , VECTOR_COSINE ( TO_VECTOR ( embedding , :%qpar>]
```

