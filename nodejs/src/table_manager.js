// rag-templates/nodejs/src/table_manager.js
// Table Manager for database schema management

const DDLGenerator = require('./ddl_generator');

/**
 * Table Manager class for managing database table schemas
 */
class TableManager {
  /**
   * Create a new Table Manager instance
   * @param {Object} connectionManager - Database connection manager
   * @param {Object} options - Configuration options
   * @param {string} options.dialect - Database dialect (default: 'iris')
   * @param {string} options.schema - Database schema name (optional)
   */
  constructor(connectionManager, options = {}) {
    if (!connectionManager) {
      throw new Error('Connection manager is required');
    }

    this.connectionManager = connectionManager;
    this.options = {
      dialect: 'iris',
      schema: null,
      ...options
    };
    
    this.ddlGenerator = new DDLGenerator(this.options.dialect);
  }

  /**
   * Check if a table exists in the database
   * @param {string} tableName - Name of the table to check (default: 'SourceDocuments')
   * @returns {Promise<boolean>} True if table exists, false otherwise
   */
  async tableExists(tableName = 'SourceDocuments') {
    if (!tableName || typeof tableName !== 'string') {
      throw new Error('Table name must be a non-empty string');
    }

    // Build schema condition
    const schemaCondition = this.options.schema 
      ? `TABLE_SCHEMA = '${this.options.schema}'`
      : '1=1';

    const sql = `
      SELECT COUNT(*) as table_count
      FROM INFORMATION_SCHEMA.TABLES 
      WHERE TABLE_NAME = ? AND ${schemaCondition}
    `;

    try {
      const connection = await this.connectionManager.connect();
      const result = await connection.query(sql, [tableName]);
      
      // Handle different result formats
      const count = Array.isArray(result) && result.length > 0 
        ? (Array.isArray(result[0]) ? result[0][0] : result[0].table_count)
        : 0;
        
      return parseInt(count) > 0;
    } catch (error) {
      throw new Error(`Failed to check table existence: ${error.message}`);
    }
  }

  /**
   * Create a table using the provided configuration
   * @param {Object} config - Table configuration object
   * @returns {Promise<void>}
   */
  async createTable(config) {
    if (!config) {
      throw new Error('Table configuration is required');
    }

    try {
      // Generate CREATE TABLE SQL
      const createTableSQL = this.ddlGenerator.generateCreateTableSQL(config);
      
      // Execute table creation
      const connection = await this.connectionManager.connect();
      await connection.query(createTableSQL);

      // Create indexes if specified
      if (config.indexes && Array.isArray(config.indexes)) {
        for (const indexConfig of config.indexes) {
          try {
            const createIndexSQL = this.ddlGenerator.generateCreateIndexSQL(
              config.tableName || 'SourceDocuments', 
              indexConfig
            );
            await connection.query(createIndexSQL);
          } catch (indexError) {
            // Log warning for index creation failures but don't throw
            console.warn(`Warning: Failed to create index '${indexConfig.name}': ${indexError.message}`);
          }
        }
      }
    } catch (error) {
      throw new Error(`Table creation failed: ${error.message}`);
    }
  }

  /**
   * Ensure a table exists, creating it if necessary
   * @param {Object} config - Table configuration object
   * @returns {Promise<Object>} Result object with creation status
   */
  async ensureTable(config) {
    if (!config) {
      throw new Error('Table configuration is required');
    }

    const tableName = config.tableName || 'SourceDocuments';
    
    // Validate table name
    if (!this._isValidTableName(tableName)) {
      throw new Error(`Invalid table name: "${tableName}". Table names cannot contain dots, spaces, or special characters except underscores.`);
    }
    
    try {
      const exists = await this.tableExists(tableName);
      
      if (exists) {
        return {
          created: false,
          existed: true,
          tableName
        };
      }

      // Table doesn't exist, create it
      await this.createTable(config);
      
      return {
        created: true,
        existed: false,
        tableName
      };
    } catch (error) {
      throw new Error(`Failed to ensure table exists: ${error.message}`);
    }
  }

  /**
   * Get the schema of an existing table
   * @param {string} tableName - Name of the table
   * @returns {Promise<Object>} Table schema information
   */
  async getTableSchema(tableName) {
    if (!tableName || typeof tableName !== 'string') {
      throw new Error('Table name must be a non-empty string');
    }

    // Build schema condition
    const schemaCondition = this.options.schema 
      ? `TABLE_SCHEMA = '${this.options.schema}'`
      : '1=1';

    const sql = `
      SELECT 
        COLUMN_NAME,
        DATA_TYPE,
        IS_NULLABLE,
        COLUMN_DEFAULT,
        ORDINAL_POSITION
      FROM INFORMATION_SCHEMA.COLUMNS 
      WHERE TABLE_NAME = ? AND ${schemaCondition}
      ORDER BY ORDINAL_POSITION
    `;

    try {
      const connection = await this.connectionManager.connect();
      const result = await connection.query(sql, [tableName]);
      
      if (!Array.isArray(result) || result.length === 0) {
        throw new Error(`Table '${tableName}' not found or has no columns`);
      }

      // Format schema information
      const columns = result.map(row => {
        const columnData = Array.isArray(row) ? {
          name: row[0],
          dataType: row[1],
          nullable: row[2],
          defaultValue: row[3],
          position: row[4]
        } : {
          name: row.COLUMN_NAME,
          dataType: row.DATA_TYPE,
          nullable: row.IS_NULLABLE,
          defaultValue: row.COLUMN_DEFAULT,
          position: row.ORDINAL_POSITION
        };

        return {
          name: columnData.name,
          dataType: columnData.dataType,
          nullable: columnData.nullable === 'YES' || columnData.nullable === true,
          defaultValue: columnData.defaultValue,
          position: parseInt(columnData.position)
        };
      });

      // Get primary key information
      const primaryKey = await this._getPrimaryKeyColumns(tableName);

      return {
        tableName,
        columns,
        primaryKey,
        columnCount: columns.length
      };
    } catch (error) {
      throw new Error(`Failed to get table schema: ${error.message}`);
    }
  }

  /**
   * Validate table schema against expected configuration
   * @param {string} tableName - Name of the table to validate
   * @param {Object} expectedSchemaConfig - Expected schema configuration
   * @returns {Promise<Object>} Validation result
   */
  async validateSchema(tableName, expectedSchemaConfig) {
    if (!tableName || typeof tableName !== 'string') {
      throw new Error('Table name must be a non-empty string');
    }
    if (!expectedSchemaConfig) {
      throw new Error('Expected schema configuration is required');
    }

    try {
      const actualSchema = await this.getTableSchema(tableName);
      const differences = [];

      // Check if table exists
      if (!actualSchema) {
        return {
          valid: false,
          differences: ['Table does not exist']
        };
      }

      // Validate expected columns exist
      const expectedColumns = expectedSchemaConfig.columns || [];
      const actualColumnNames = actualSchema.columns.map(col => col.name.toLowerCase());

      for (const expectedCol of expectedColumns) {
        const colName = expectedCol.name.toLowerCase();
        const actualCol = actualSchema.columns.find(col => col.name.toLowerCase() === colName);

        if (!actualCol) {
          differences.push(`Missing column: ${expectedCol.name}`);
          continue;
        }

        // Check data type compatibility (basic check)
        if (!this._isDataTypeCompatible(actualCol.dataType, expectedCol.type)) {
          differences.push(`Column '${expectedCol.name}' type mismatch: expected ${expectedCol.type}, got ${actualCol.dataType}`);
        }
      }

      // Check for embedding column (critical for vector operations)
      const hasEmbeddingColumn = actualSchema.columns.some(col => 
        col.name.toLowerCase() === 'embedding' && 
        col.dataType.toUpperCase().includes('VECTOR')
      );

      if (!hasEmbeddingColumn) {
        differences.push('Missing or invalid embedding column of type VECTOR');
      }

      // Check primary key
      const expectedPrimaryKey = expectedSchemaConfig.primaryKey || [];
      if (expectedPrimaryKey.length > 0) {
        const missingPkColumns = expectedPrimaryKey.filter(pkCol => 
          !actualSchema.primaryKey.includes(pkCol)
        );
        if (missingPkColumns.length > 0) {
          differences.push(`Primary key missing columns: ${missingPkColumns.join(', ')}`);
        }
      }

      return {
        valid: differences.length === 0,
        differences,
        actualSchema,
        expectedColumns: expectedColumns.length,
        actualColumns: actualSchema.columns.length
      };
    } catch (error) {
      throw new Error(`Schema validation failed: ${error.message}`);
    }
  }

  /**
   * Get primary key columns for a table
   * @param {string} tableName - Name of the table
   * @returns {Promise<Array<string>>} Array of primary key column names
   * @private
   */
  async _getPrimaryKeyColumns(tableName) {
    const schemaCondition = this.options.schema 
      ? `TABLE_SCHEMA = '${this.options.schema}'`
      : '1=1';

    const sql = `
      SELECT COLUMN_NAME
      FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
      WHERE TABLE_NAME = ? AND ${schemaCondition} AND CONSTRAINT_NAME = 'PRIMARY'
      ORDER BY ORDINAL_POSITION
    `;

    try {
      const connection = await this.connectionManager.connect();
      const result = await connection.query(sql, [tableName]);
      
      return Array.isArray(result) 
        ? result.map(row => Array.isArray(row) ? row[0] : row.COLUMN_NAME)
        : [];
    } catch (error) {
      // Primary key information might not be available in all IRIS configurations
      console.warn(`Could not retrieve primary key information: ${error.message}`);
      return [];
    }
  }

  /**
   * Check if data types are compatible
   * @param {string} actualType - Actual column data type
   * @param {string} expectedType - Expected column data type
   * @returns {boolean} True if types are compatible
   * @private
   */
  _isDataTypeCompatible(actualType, expectedType) {
    if (!actualType || !expectedType) {
      return false;
    }

    const actual = actualType.toUpperCase();
    const expected = expectedType.toUpperCase();

    // Exact match
    if (actual === expected) {
      return true;
    }

    // Handle common type variations
    const typeMapping = {
      'VARCHAR': ['VARCHAR', 'TEXT', 'STRING'],
      'TEXT': ['TEXT', 'LONGTEXT', 'VARCHAR'],
      'LONGTEXT': ['LONGTEXT', 'TEXT', 'VARCHAR'],
      'INTEGER': ['INTEGER', 'INT', 'BIGINT'],
      'INT': ['INT', 'INTEGER', 'BIGINT'],
      'TIMESTAMP': ['TIMESTAMP', 'DATETIME'],
      'VECTOR': ['VECTOR']
    };

    // Check if actual type starts with expected base type
    for (const [baseType, variants] of Object.entries(typeMapping)) {
      if (expected.startsWith(baseType) && variants.some(variant => actual.startsWith(variant))) {
        return true;
      }
    }

    return false;
  }

  /**
   * Validate table name format
   * @param {string} tableName - Table name to validate
   * @returns {boolean} True if valid, false otherwise
   */
  _isValidTableName(tableName) {
    if (!tableName || typeof tableName !== 'string') {
      return false;
    }
    
    // Table names should only contain letters, numbers, and underscores
    // No dots, spaces, or special characters
    const validTableNameRegex = /^[a-zA-Z][a-zA-Z0-9_]*$/;
    return validTableNameRegex.test(tableName);
  }
}

module.exports = TableManager;
