// rag-templates/nodejs/src/vector_sql_utils.js
// SQL utilities for IRIS vector search operations

class VectorSQLUtils {
  static validateVectorString(vectorString) {
    if (!vectorString || typeof vectorString !== 'string') {
      return false;
    }
    
    // Remove whitespace and check basic format
    const cleaned = vectorString.trim();
    if (!cleaned) return false;
    
    // Check for SQL injection patterns first
    const sqlInjectionPatterns = [
      /['";]/,                    // Quotes and semicolons
      /\b(DROP|DELETE|INSERT|UPDATE|UNION|SELECT|CREATE|ALTER|EXEC|EXECUTE)\b/i,
      /--/,                       // SQL comments
      /\/\*/,                     // SQL block comments
      /\bOR\s+\d+\s*=\s*\d+/i,   // OR 1=1 patterns
      /\bAND\s+\d+\s*=\s*\d+/i   // AND 1=1 patterns
    ];
    
    for (const pattern of sqlInjectionPatterns) {
      if (pattern.test(vectorString)) {
        return false;
      }
    }
    
    // Split by comma and validate each number
    const parts = cleaned.split(',');
    if (parts.length === 0) return false;
    
    for (const part of parts) {
      const trimmed = part.trim();
      if (!trimmed) return false;
      
      // Check if the entire trimmed part is a valid number
      // Use a strict regex to ensure only numbers (no extra characters)
      const numberRegex = /^-?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/;
      if (!numberRegex.test(trimmed)) {
        return false;
      }
      
      // Additional check with parseFloat
      const num = parseFloat(trimmed);
      if (isNaN(num) || !isFinite(num)) return false;
    }
    
    return true;
  }

  static validateTopK(topK) {
    return Number.isInteger(topK) && topK > 0 && topK <= Number.MAX_SAFE_INTEGER;
  }

  static validateTableName(tableName) {
    if (!tableName || typeof tableName !== 'string') {
      return false;
    }
    
    // Basic SQL identifier validation
    const sqlIdentifierRegex = /^[a-zA-Z_][a-zA-Z0-9_]*$/;
    return sqlIdentifierRegex.test(tableName);
  }

  static validateColumnName(columnName) {
    if (!columnName || typeof columnName !== 'string') {
      return false;
    }
    
    // Basic SQL identifier validation
    const sqlIdentifierRegex = /^[a-zA-Z_][a-zA-Z0-9_]*$/;
    return sqlIdentifierRegex.test(columnName);
  }

  static formatVectorSearchSQL(options) {
    const {
      vectorString,
      topK = 10,
      tableName = 'SourceDocuments',
      vectorColumn = 'embedding',
      contentColumn = 'text_content',
      idColumn = 'doc_id',
      additionalColumns = [],
      additionalWhere = null,
      minSimilarity = null,
      embeddingDim = 384
    } = options;

    if (!vectorString) {
      throw new Error('Vector string is required');
    }
    if (!this.validateVectorString(vectorString)) {
      throw new Error(`Invalid vector string: ${vectorString}`);
    }
    if (!this.validateTopK(topK)) {
      throw new Error(`Invalid top_k value: ${topK}`);
    }
    if (!this.validateTableName(tableName)) {
      throw new Error(`Invalid table name: ${tableName}`);
    }
    if (!this.validateColumnName(vectorColumn)) {
      throw new Error(`Invalid vector column name: ${vectorColumn}`);
    }

    // Build SELECT clause
    let selectColumns = [idColumn];
    if (contentColumn) {
      selectColumns.push(contentColumn);
    }
    selectColumns.push('source_file', 'page_number', 'chunk_index');
    selectColumns = selectColumns.concat(additionalColumns);
    
    // Add similarity score
    selectColumns.push(`VECTOR_COSINE(${vectorColumn}, TO_VECTOR('${vectorString}', 'FLOAT', ${embeddingDim})) AS score`);

    let sql = `SELECT TOP ${topK} ${selectColumns.join(', ')} FROM ${tableName}`;
    
    // Build WHERE clause
    const whereConditions = [`${vectorColumn} IS NOT NULL`];
    
    if (minSimilarity !== null) {
      whereConditions.push(`VECTOR_COSINE(${vectorColumn}, TO_VECTOR('${vectorString}', 'FLOAT', ${embeddingDim})) >= ${minSimilarity}`);
    }
    
    if (additionalWhere) {
      // Basic SQL injection protection - reject dangerous patterns
      const dangerousPatterns = [
        /;\s*(drop|delete|update|insert|create|alter)\s+/i,
        /union\s+select/i,
        /--/,
        /\/\*/,
        /\*\//
      ];
      
      for (const pattern of dangerousPatterns) {
        if (pattern.test(additionalWhere)) {
          throw new Error('Potentially dangerous SQL detected in additionalWhere clause');
        }
      }
      
      whereConditions.push(additionalWhere);
    }
    
    if (whereConditions.length > 0) {
      sql += ` WHERE ${whereConditions.join(' AND ')}`;
    }
    
    sql += ` ORDER BY score DESC`;
    
    return sql;
  }

  static formatVectorForIris(vector) {
    if (!Array.isArray(vector)) {
      throw new Error('Vector must be an array');
    }
    
    return vector.map(num => {
      if (typeof num !== 'number' || !isFinite(num)) {
        throw new Error('All vector elements must be finite numbers');
      }
      
      // Handle very small numbers in scientific notation
      if (Math.abs(num) < 1e-6 && num !== 0) {
        return num.toFixed(9);
      }
      
      return num.toString();
    }).join(',');
  }

  static formatInsertSQL(options) {
    const {
      tableName = 'SourceDocuments',
      docId,
      textContent,
      vectorString,
      sourceFile,
      pageNumber,
      chunkIndex,
      additionalFields = {}
    } = options;

    if (!docId || !textContent || !vectorString) {
      throw new Error('docId, textContent, and vectorString are required');
    }

    if (!this.validateTableName(tableName)) {
      throw new Error(`Invalid table name: ${tableName}`);
    }

    if (!this.validateVectorString(vectorString)) {
      throw new Error(`Invalid vector string: ${vectorString}`);
    }

    const columns = ['doc_id', 'text_content', 'embedding_vector', 'source_file', 'page_number', 'chunk_index'];
    const values = ['?', '?', `TO_VECTOR('${vectorString}')`, '?', '?', '?'];
    const params = [docId, textContent, sourceFile, pageNumber, chunkIndex];

    // Add additional fields
    for (const [key, value] of Object.entries(additionalFields)) {
      if (!this.validateColumnName(key)) {
        throw new Error(`Invalid column name: ${key}`);
      }
      columns.push(key);
      values.push('?');
      params.push(value);
    }

    const sql = `INSERT INTO ${tableName} (${columns.join(', ')}) VALUES (${values.join(', ')})`;
    
    return { sql, params };
  }

  static formatUpdateSQL(options) {
    const {
      tableName = 'SourceDocuments',
      docId,
      textContent,
      vectorString,
      additionalFields = {}
    } = options;

    if (!docId) {
      throw new Error('docId is required for update');
    }

    if (!this.validateTableName(tableName)) {
      throw new Error(`Invalid table name: ${tableName}`);
    }

    const setClauses = [];
    const params = [];

    if (textContent) {
      setClauses.push('text_content = ?');
      params.push(textContent);
    }

    if (vectorString) {
      if (!this.validateVectorString(vectorString)) {
        throw new Error(`Invalid vector string: ${vectorString}`);
      }
      setClauses.push(`embedding_vector = TO_VECTOR('${vectorString}')`);
    }

    // Add additional fields
    for (const [key, value] of Object.entries(additionalFields)) {
      if (!this.validateColumnName(key)) {
        throw new Error(`Invalid column name: ${key}`);
      }
      setClauses.push(`${key} = ?`);
      params.push(value);
    }

    if (setClauses.length === 0) {
      throw new Error('At least one field must be provided for update');
    }

    params.push(docId);
    const sql = `UPDATE ${tableName} SET ${setClauses.join(', ')} WHERE doc_id = ?`;
    
    return { sql, params };
  }

  static formatDeleteSQL(options) {
    const {
      tableName = 'SourceDocuments',
      docId
    } = options;

    if (!docId) {
      throw new Error('docId is required for delete');
    }

    if (!this.validateTableName(tableName)) {
      throw new Error(`Invalid table name: ${tableName}`);
    }

    const sql = `DELETE FROM ${tableName} WHERE doc_id = ?`;
    const params = [docId];
    
    return { sql, params };
  }
}

module.exports = VectorSQLUtils;