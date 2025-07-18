// rag-templates/nodejs/src/real_iris_connection_manager.js
// Real implementation for InterSystems IRIS connection management

const { IRIS } = require('intersystems-iris');

/**
 * Real IRIS Connection Manager for production use
 */
class RealIrisConnectionManager {
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
    if (this.isConnected && this.connection) {
      return this.connection;
    }

    try {
      // Create IRIS connection using the correct API
      // Constructor: new IRIS(host, port, namespace, username, password)
      this.connection = new IRIS(
        this.config.host,
        this.config.port,
        this.config.namespace,
        this.config.username,
        this.config.password
      );

      this.isConnected = true;
      
      return {
        // Wrapper for query method to match expected interface
        query: async (sql, params = []) => {
          try {
            const result = await this.connection.sql(sql, params);
            // Return rows if available, otherwise return the full result
            return result.rows || result;
          } catch (error) {
            throw new Error(`Query failed: ${error.message}`);
          }
        }
      };
    } catch (error) {
      this.isConnected = false;
      throw new Error(`Failed to connect to IRIS: ${error.message}`);
    }
  }

  async disconnect() {
    if (this.connection) {
      try {
        await this.connection.close();
      } catch (error) {
        console.warn('Error during disconnect:', error.message);
      }
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

module.exports = RealIrisConnectionManager;