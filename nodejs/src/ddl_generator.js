// rag-templates/nodejs/src/ddl_generator.js
// DDL Generator for creating table and index SQL statements

/**
 * Configuration object for table creation
 * @typedef {Object} TableConfig
 * @property {string} tableName - Name of the table to create
 * @property {Array<Object>} columns - Array of column definitions
 * @property {Array<string>} primaryKey - Array of column names forming the primary key
 * @property {Array<Object>} indexes - Array of index definitions
 * @property {Object} options - Additional table options
 */

/**
 * DDL Generator class for creating SQL DDL statements
 */
class DDLGenerator {
  /**
   * Create a new DDL Generator instance
   * @param {string} dialect - Database dialect (default: 'iris')
   */
  constructor(dialect = 'iris') {
    this.dialect = dialect.toLowerCase();
    this.supportedDialects = ['iris'];
    
    if (!this.supportedDialects.includes(this.dialect)) {
      throw new Error(`Unsupported dialect: ${dialect}. Supported dialects: ${this.supportedDialects.join(', ')}`);
    }
  }

  /**
   * Generate CREATE TABLE SQL statement
   * @param {TableConfig|Object} config - Table configuration object
   * @returns {string} CREATE TABLE SQL statement
   */
  generateCreateTableSQL(config) {
    if (!config) {
      throw new Error('Table configuration is required');
    }

    // Use default configuration if not provided
    const tableConfig = this._getDefaultTableConfig(config);
    
    // Validate table configuration
    this._validateTableConfig(tableConfig);

    const { tableName, columns, primaryKey, options = {} } = tableConfig;

    // Build column definitions
    const columnDefs = columns.map(col => this._formatColumnDefinition(col)).join(',\n  ');
    
    // Build primary key constraint
    const primaryKeyDef = primaryKey && primaryKey.length > 0 
      ? `,\n  PRIMARY KEY (${primaryKey.join(', ')})`
      : '';

    // Build table options
    const tableOptions = this._formatTableOptions(options);

    const sql = `CREATE TABLE ${this._sanitizeIdentifier(tableName)} (
  ${columnDefs}${primaryKeyDef}
)${tableOptions}`;

    return sql;
  }

  /**
   * Generate CREATE INDEX SQL statement
   * @param {string} tableName - Name of the table
   * @param {Object} indexConfig - Index configuration
   * @returns {string} CREATE INDEX SQL statement
   */
  generateCreateIndexSQL(tableName, indexConfig) {
    if (!tableName) {
      throw new Error('Table name is required');
    }
    if (!indexConfig) {
      throw new Error('Index configuration is required');
    }

    this._validateIndexConfig(indexConfig);

    const { name, columns, type = 'BTREE', unique = false } = indexConfig;
    
    const indexName = name || `idx_${tableName}_${columns.join('_')}`;
    const uniqueKeyword = unique ? 'UNIQUE ' : '';
    
    // Handle IRIS vector index types
    let typeClause = '';
    if (type === 'VECTOR_HNSW') {
      typeClause = ' USING VECTOR_HNSW';
    } else if (type !== 'BTREE') {
      typeClause = ` USING ${type}`;
    }
    
    const columnList = columns.map(col => {
      if (typeof col === 'string') {
        return this._sanitizeIdentifier(col);
      } else if (typeof col === 'object' && col.name) {
        const direction = col.direction ? ` ${col.direction.toUpperCase()}` : '';
        return `${this._sanitizeIdentifier(col.name)}${direction}`;
      }
      throw new Error('Invalid column specification in index');
    }).join(', ');

    return `CREATE ${uniqueKeyword}INDEX ${this._sanitizeIdentifier(indexName)} ON ${this._sanitizeIdentifier(tableName)}${typeClause} (${columnList})`;
  }

  /**
   * Get default table configuration for SourceDocuments
   * @param {Object} config - Partial configuration to merge with defaults
   * @returns {TableConfig} Complete table configuration
   */
  _getDefaultTableConfig(config) {
    const defaultConfig = {
      tableName: 'SourceDocuments',
      columns: [
        { name: 'doc_id', type: 'VARCHAR(255)', nullable: false },
        { name: 'title', type: 'VARCHAR(500)', nullable: false },
        { name: 'text_content', type: 'LONGTEXT', nullable: false },
        { name: 'source_file', type: 'VARCHAR(255)', nullable: true },
        { name: 'page_number', type: 'INTEGER', nullable: true },
        { name: 'chunk_index', type: 'INTEGER', nullable: true },
        { name: 'embedding', type: 'VECTOR(FLOAT, 384)', nullable: true },
        { name: 'created_at', type: 'TIMESTAMP', nullable: true, default: 'CURRENT_TIMESTAMP' },
        { name: 'updated_at', type: 'TIMESTAMP', nullable: true }
      ],
      primaryKey: ['doc_id'],
      indexes: [
        { name: 'idx_source_file', columns: ['source_file'] },
        { name: 'idx_page_chunk', columns: ['page_number', 'chunk_index'] },
        { name: 'idx_created_at', columns: ['created_at'] },
        { name: 'idx_embedding_hnsw', columns: ['embedding'], type: 'VECTOR_HNSW' }
      ],
      options: {}
    };

    // Merge with provided config
    return {
      ...defaultConfig,
      ...config,
      columns: config.columns || defaultConfig.columns,
      primaryKey: config.primaryKey || defaultConfig.primaryKey,
      indexes: config.indexes || defaultConfig.indexes,
      options: { ...defaultConfig.options, ...(config.options || {}) }
    };
  }

  /**
   * Validate table configuration
   * @param {TableConfig} config - Table configuration to validate
   * @throws {Error} If configuration is invalid
   */
  _validateTableConfig(config) {
    if (!config.tableName || typeof config.tableName !== 'string') {
      throw new Error('Table name must be a non-empty string');
    }

    if (!this._isValidIdentifier(config.tableName)) {
      throw new Error('Table name contains invalid characters');
    }

    if (!Array.isArray(config.columns) || config.columns.length === 0) {
      throw new Error('Columns must be a non-empty array');
    }

    // Validate each column
    config.columns.forEach((col, index) => {
      if (!col.name || typeof col.name !== 'string') {
        throw new Error(`Column ${index} must have a valid name`);
      }
      if (!this._isValidIdentifier(col.name)) {
        throw new Error(`Column name '${col.name}' contains invalid characters`);
      }
      if (!col.type || typeof col.type !== 'string') {
        throw new Error(`Column '${col.name}' must have a valid type`);
      }
    });

    // Validate primary key
    if (config.primaryKey && Array.isArray(config.primaryKey)) {
      config.primaryKey.forEach(keyCol => {
        if (!config.columns.find(col => col.name === keyCol)) {
          throw new Error(`Primary key column '${keyCol}' not found in table columns`);
        }
      });
    }
  }

  /**
   * Validate index configuration
   * @param {Object} indexConfig - Index configuration to validate
   * @throws {Error} If configuration is invalid
   */
  _validateIndexConfig(indexConfig) {
    if (!Array.isArray(indexConfig.columns) || indexConfig.columns.length === 0) {
      throw new Error('Index must have at least one column');
    }

    if (indexConfig.name && !this._isValidIdentifier(indexConfig.name)) {
      throw new Error('Index name contains invalid characters');
    }
  }

  /**
   * Format a column definition for SQL
   * @param {Object} column - Column configuration
   * @returns {string} Formatted column definition
   */
  _formatColumnDefinition(column) {
    const { name, type, nullable = true, default: defaultValue } = column;
    
    let definition = `${this._sanitizeIdentifier(name)} ${type}`;
    
    if (!nullable) {
      definition += ' NOT NULL';
    }
    
    if (defaultValue !== undefined) {
      definition += ` DEFAULT ${defaultValue}`;
    }
    
    return definition;
  }

  /**
   * Format table options for SQL
   * @param {Object} options - Table options
   * @returns {string} Formatted table options
   */
  _formatTableOptions(options) {
    if (!options || Object.keys(options).length === 0) {
      return '';
    }

    const optionPairs = Object.entries(options).map(([key, value]) => {
      return `${key.toUpperCase()}=${value}`;
    });

    return ` ${optionPairs.join(' ')}`;
  }

  /**
   * Sanitize SQL identifier (table/column names)
   * @param {string} identifier - Identifier to sanitize
   * @returns {string} Sanitized identifier
   */
  _sanitizeIdentifier(identifier) {
    if (!this._isValidIdentifier(identifier)) {
      throw new Error(`Invalid identifier: ${identifier}`);
    }
    return identifier;
  }

  /**
   * Check if identifier is valid (alphanumeric and underscores only)
   * @param {string} identifier - Identifier to validate
   * @returns {boolean} True if valid
   */
  _isValidIdentifier(identifier) {
    if (!identifier || typeof identifier !== 'string') {
      return false;
    }
    // Allow alphanumeric characters, underscores, and must start with letter or underscore
    return /^[a-zA-Z_][a-zA-Z0-9_]*$/.test(identifier);
  }
}

module.exports = DDLGenerator;