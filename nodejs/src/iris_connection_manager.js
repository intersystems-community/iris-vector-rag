// rag-templates/nodejs/src/iris_connection_manager.js
// Mock implementation for InterSystems IRIS connection management

class IrisConnectionManager {
  constructor(config = {}) {
    this.config = {
      host: config.host || 'localhost',
      port: config.port || 1972,
      namespace: config.namespace || 'USER',
      username: config.username || '_SYSTEM',
      password: config.password || 'SYS',
      ...config
    };
    this.connection = null;
    this.isConnected = false;
  }

  async connect() {
    if (this.isConnected) {
      return this.connection;
    }

    try {
      // Mock connection object
      this.connection = {
        host: this.config.host,
        port: this.config.port,
        namespace: this.config.namespace,
        isConnected: true,
        queryHistory: [],
        mockResults: new Map(),
        errorMode: false,
        errorMessage: '',

        // Mock query method
        query: async (sql, params = []) => {
          this.connection.queryHistory.push({ sql, params });

          if (this.connection.errorMode) {
            throw new Error(this.connection.errorMessage);
          }

          // Return mock results based on query type
          if (sql.includes('VECTOR_COSINE')) {
            return this.connection.mockResults.get('vector_search') || [];
          } else if (sql.includes('COUNT(*)')) {
            return this.connection.mockResults.get('count_documents') || [[0]];
          } else if (sql.includes('INSERT')) {
            return { affectedRows: 1 };
          } else if (sql.includes('UPDATE')) {
            return { affectedRows: 1 };
          } else if (sql.includes('DELETE')) {
            return { affectedRows: 1 };
          }

          return [];
        },

        // Mock helper methods
        setMockResult: (queryType, result) => {
          this.connection.mockResults.set(queryType, result);
        },

        setErrorMode: (enabled, message = 'Mock error') => {
          this.connection.errorMode = enabled;
          this.connection.errorMessage = message;
        },

        getQueryHistory: () => {
          return [...this.connection.queryHistory];
        },

        clearHistory: () => {
          this.connection.queryHistory = [];
        }
      };

      this.isConnected = true;
      return this.connection;
    } catch (error) {
      throw new Error(`Failed to connect to IRIS: ${error.message}`);
    }
  }

  async disconnect() {
    if (this.connection) {
      this.connection.isConnected = false;
      this.connection = null;
    }
    this.isConnected = false;
  }

  async close() {
    await this.disconnect();
  }

  /**
   * Execute a query (alias for connection.query for compatibility)
   * @param {string} sql - SQL query to execute
   * @param {Array} params - Query parameters
   * @returns {Promise<any>} Query result
   */
  async executeQuery(sql, params = []) {
    const connection = await this.connect();
    return await connection.query(sql, params);
  }

  getConnectionInfo() {
    return {
      host: this.config.host,
      port: this.config.port,
      namespace: this.config.namespace,
      isConnected: this.isConnected
    };
  }

  async testConnection() {
    try {
      const conn = await this.connect();
      const result = await conn.query('SELECT 1 as test');
      return { success: true, result };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }
}

export default IrisConnectionManager;