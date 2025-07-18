// rag-templates/nodejs/tests/unit/iris_connection_manager.test.js
// Unit tests for IrisConnectionManager

const IrisConnectionManager = require('../../src/iris_connection_manager');
const { TestConfiguration } = require('../conftest');

describe('IrisConnectionManager', () => {
  let testConfig;
  let manager;

  beforeEach(() => {
    testConfig = new TestConfiguration();
    manager = new IrisConnectionManager(testConfig.irisConfig);
  });

  afterEach(async () => {
    if (manager) {
      await manager.disconnect();
    }
  });

  describe('constructor', () => {
    it('should create instance with default configuration', () => {
      const defaultManager = new IrisConnectionManager();
      const info = defaultManager.getConnectionInfo();
      
      expect(info.host).toBe('localhost');
      expect(info.port).toBe(1972);
      expect(info.namespace).toBe('USER');
      expect(info.isConnected).toBe(false);
    });

    it('should create instance with custom configuration', () => {
      const customConfig = {
        host: 'custom-host',
        port: 9999,
        namespace: 'CUSTOM',
        username: 'testuser',
        password: 'testpass'
      };
      
      const customManager = new IrisConnectionManager(customConfig);
      const info = customManager.getConnectionInfo();
      
      expect(info.host).toBe('custom-host');
      expect(info.port).toBe(9999);
      expect(info.namespace).toBe('CUSTOM');
      expect(info.isConnected).toBe(false);
    });

    it('should merge custom config with defaults', () => {
      const partialConfig = { host: 'partial-host' };
      const partialManager = new IrisConnectionManager(partialConfig);
      const info = partialManager.getConnectionInfo();
      
      expect(info.host).toBe('partial-host');
      expect(info.port).toBe(1972); // default
      expect(info.namespace).toBe('USER'); // default
    });
  });

  describe('connect', () => {
    it('should establish connection successfully', async () => {
      const connection = await manager.connect();
      
      expect(connection).toBeDefined();
      expect(connection.isConnected).toBe(true);
      expect(connection.host).toBe(testConfig.irisConfig.host);
      expect(manager.isConnected).toBe(true);
    });

    it('should return existing connection if already connected', async () => {
      const connection1 = await manager.connect();
      const connection2 = await manager.connect();
      
      expect(connection1).toBe(connection2);
      expect(manager.isConnected).toBe(true);
    });

    it('should provide connection with query method', async () => {
      const connection = await manager.connect();
      
      expect(connection.query).toBeDefined();
      expect(typeof connection.query).toBe('function');
    });

    it('should provide connection with mock helper methods', async () => {
      const connection = await manager.connect();
      
      expect(connection.setMockResult).toBeDefined();
      expect(connection.setErrorMode).toBeDefined();
      expect(connection.getQueryHistory).toBeDefined();
      expect(connection.clearHistory).toBeDefined();
    });
  });

  describe('disconnect', () => {
    it('should disconnect successfully', async () => {
      await manager.connect();
      expect(manager.isConnected).toBe(true);
      
      await manager.disconnect();
      
      expect(manager.isConnected).toBe(false);
      expect(manager.connection).toBeNull();
    });

    it('should handle disconnect when not connected', async () => {
      expect(manager.isConnected).toBe(false);
      
      await expect(manager.disconnect()).resolves.not.toThrow();
      expect(manager.isConnected).toBe(false);
    });

    it('should mark connection as disconnected', async () => {
      const connection = await manager.connect();
      expect(connection.isConnected).toBe(true);
      
      await manager.disconnect();
      
      expect(connection.isConnected).toBe(false);
    });
  });

  describe('close', () => {
    it('should close connection successfully', async () => {
      await manager.connect();
      expect(manager.isConnected).toBe(true);
      
      await manager.close();
      
      expect(manager.isConnected).toBe(false);
      expect(manager.connection).toBeNull();
    });

    it('should handle close when not connected', async () => {
      expect(manager.isConnected).toBe(false);
      
      await expect(manager.close()).resolves.not.toThrow();
      expect(manager.isConnected).toBe(false);
    });
  });

  describe('getConnectionInfo', () => {
    it('should return connection information when not connected', () => {
      const info = manager.getConnectionInfo();
      
      expect(info).toEqual({
        host: testConfig.irisConfig.host,
        port: testConfig.irisConfig.port,
        namespace: testConfig.irisConfig.namespace,
        isConnected: false
      });
    });

    it('should return connection information when connected', async () => {
      await manager.connect();
      const info = manager.getConnectionInfo();
      
      expect(info).toEqual({
        host: testConfig.irisConfig.host,
        port: testConfig.irisConfig.port,
        namespace: testConfig.irisConfig.namespace,
        isConnected: true
      });
    });
  });

  describe('testConnection', () => {
    it('should test connection successfully', async () => {
      const result = await manager.testConnection();
      
      expect(result.success).toBe(true);
      expect(result.result).toBeDefined();
      expect(manager.isConnected).toBe(true);
    });

    it('should handle connection test failures', async () => {
      // Connect first to get a connection object
      await manager.connect();
      const connection = manager.connection;
      
      // Set error mode on the connection that will be reused
      connection.setErrorMode(true, 'Test connection failure');
      
      // Test connection while error mode is active
      const result = await manager.testConnection();
      
      expect(result.success).toBe(false);
      expect(result.error).toBeDefined();
      expect(typeof result.error).toBe('string');
      expect(result.error).toContain('Test connection failure');
    });
  });

  describe('connection query functionality', () => {
    let connection;

    beforeEach(async () => {
      connection = await manager.connect();
    });

    it('should execute queries and track history', async () => {
      const sql = 'SELECT * FROM test_table';
      const params = ['param1', 'param2'];
      
      await connection.query(sql, params);
      
      const history = connection.getQueryHistory();
      expect(history).toHaveLength(1);
      expect(history[0]).toEqual({ sql, params });
    });

    it('should handle vector search queries', async () => {
      connection.setMockResult('vector_search', [
        { id: 1, content: 'test', similarity: 0.95 }
      ]);
      
      const result = await connection.query('SELECT * FROM docs WHERE VECTOR_COSINE(embedding, ?) > 0.8');
      
      expect(result).toHaveLength(1);
      expect(result[0].similarity).toBe(0.95);
    });

    it('should handle count queries', async () => {
      connection.setMockResult('count_documents', [[42]]);
      
      const result = await connection.query('SELECT COUNT(*) FROM docs');
      
      expect(result).toEqual([[42]]);
    });

    it('should handle INSERT queries', async () => {
      const result = await connection.query('INSERT INTO docs (content) VALUES (?)', ['test']);
      
      expect(result.affectedRows).toBe(1);
    });

    it('should handle UPDATE queries', async () => {
      const result = await connection.query('UPDATE docs SET content = ? WHERE id = ?', ['new content', 1]);
      
      expect(result.affectedRows).toBe(1);
    });

    it('should handle DELETE queries', async () => {
      const result = await connection.query('DELETE FROM docs WHERE id = ?', [1]);
      
      expect(result.affectedRows).toBe(1);
    });

    it('should handle unknown query types', async () => {
      const result = await connection.query('SHOW TABLES');
      
      expect(result).toEqual([]);
    });

    it('should handle query errors when error mode enabled', async () => {
      connection.setErrorMode(true, 'Database connection lost');
      
      await expect(connection.query('SELECT 1')).rejects.toThrow('Database connection lost');
    });

    it('should clear query history', async () => {
      await connection.query('SELECT 1');
      await connection.query('SELECT 2');
      
      expect(connection.getQueryHistory()).toHaveLength(2);
      
      connection.clearHistory();
      
      expect(connection.getQueryHistory()).toHaveLength(0);
    });

    it('should set and retrieve mock results', async () => {
      const mockData = [{ id: 1, name: 'test' }];
      connection.setMockResult('custom_query', mockData);
      
      expect(connection.mockResults.get('custom_query')).toEqual(mockData);
    });

    it('should handle error mode toggle', async () => {
      // Initially no error
      await expect(connection.query('SELECT 1')).resolves.not.toThrow();
      
      // Enable error mode
      connection.setErrorMode(true, 'Test error');
      await expect(connection.query('SELECT 1')).rejects.toThrow('Test error');
      
      // Disable error mode
      connection.setErrorMode(false);
      await expect(connection.query('SELECT 1')).resolves.not.toThrow();
    });
  });

  describe('error handling and edge cases', () => {
    it('should handle multiple connect/disconnect cycles', async () => {
      for (let i = 0; i < 3; i++) {
        await manager.connect();
        expect(manager.isConnected).toBe(true);
        
        await manager.disconnect();
        expect(manager.isConnected).toBe(false);
      }
    });

    it('should maintain separate query histories for different connections', async () => {
      const manager2 = new IrisConnectionManager(testConfig.irisConfig);
      
      const conn1 = await manager.connect();
      const conn2 = await manager2.connect();
      
      await conn1.query('SELECT 1');
      await conn2.query('SELECT 2');
      
      expect(conn1.getQueryHistory()).toHaveLength(1);
      expect(conn2.getQueryHistory()).toHaveLength(1);
      expect(conn1.getQueryHistory()[0].sql).toBe('SELECT 1');
      expect(conn2.getQueryHistory()[0].sql).toBe('SELECT 2');
      
      await manager2.disconnect();
    });

    it('should handle configuration with extra properties', () => {
      const configWithExtras = {
        host: 'test-host',
        port: 1972,
        namespace: 'TEST',
        extraProperty: 'should be preserved',
        anotherExtra: 123
      };
      
      const managerWithExtras = new IrisConnectionManager(configWithExtras);
      
      expect(managerWithExtras.config.extraProperty).toBe('should be preserved');
      expect(managerWithExtras.config.anotherExtra).toBe(123);
    });

    it('should handle empty configuration object', () => {
      const emptyManager = new IrisConnectionManager({});
      const info = emptyManager.getConnectionInfo();
      
      expect(info.host).toBe('localhost');
      expect(info.port).toBe(1972);
      expect(info.namespace).toBe('USER');
    });
  });

  describe('integration scenarios', () => {
    it('should support typical RAG workflow queries', async () => {
      const connection = await manager.connect();
      
      // Set up mock data for vector search
      connection.setMockResult('vector_search', [
        { id: 1, content: 'Document 1', source_file: 'doc1.pdf', similarity: 0.95 },
        { id: 2, content: 'Document 2', source_file: 'doc2.pdf', similarity: 0.87 }
      ]);
      
      // Test vector search
      const searchResults = await connection.query(
        'SELECT TOP 5 id, content, source_file, VECTOR_COSINE(embedding, TO_VECTOR(?, \'FLOAT\', 384)) as similarity FROM documents WHERE embedding IS NOT NULL ORDER BY similarity DESC',
        ['0.1,0.2,0.3']
      );
      
      expect(searchResults).toHaveLength(2);
      expect(searchResults[0].similarity).toBe(0.95);
      
      // Test document insertion
      const insertResult = await connection.query(
        'INSERT INTO documents (content, source_file, embedding) VALUES (?, ?, TO_VECTOR(?, \'FLOAT\', 384))',
        ['New document', 'new.pdf', '0.4,0.5,0.6']
      );
      
      expect(insertResult.affectedRows).toBe(1);
      
      // Verify query history
      const history = connection.getQueryHistory();
      expect(history).toHaveLength(2);
      expect(history[0].sql).toContain('VECTOR_COSINE');
      expect(history[1].sql).toContain('INSERT INTO');
    });

    it('should handle connection recovery scenarios', async () => {
      // Initial connection
      await manager.connect();
      expect(manager.isConnected).toBe(true);
      
      // Simulate connection loss
      await manager.disconnect();
      expect(manager.isConnected).toBe(false);
      
      // Reconnect
      const newConnection = await manager.connect();
      expect(manager.isConnected).toBe(true);
      expect(newConnection.isConnected).toBe(true);
      
      // Verify new connection works
      const result = await newConnection.query('SELECT 1');
      expect(result).toBeDefined();
    });
  });
});