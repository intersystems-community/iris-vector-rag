// rag-templates/nodejs/tests/conftest.js
// Global test configuration and fixtures following rag-templates patterns

const path = require('path');

/**
 * Test configuration manager
 * Loads configuration from environment variables with secure defaults
 */
class TestConfiguration {
  constructor() {
    this.testEnvironment = process.env.NODE_ENV || 'test';
    this.irisConfig = this.loadIrisConfig();
    this.embeddingConfig = this.loadEmbeddingConfig();
    this.performanceConfig = this.loadPerformanceConfig();
    
    this.validateConfiguration();
    this.setupGlobalTestVariables();
  }

  loadIrisConfig() {
    return {
      host: process.env.TEST_IRIS_HOST || 'localhost',
      port: parseInt(process.env.TEST_IRIS_PORT) || 1972,
      namespace: process.env.TEST_IRIS_NAMESPACE || 'USER',
      username: process.env.TEST_IRIS_USERNAME || 'SuperUser',
      password: process.env.TEST_IRIS_PASSWORD || 'SYS',
      useReal: process.env.USE_REAL_IRIS === 'true',
      timeout: parseInt(process.env.TEST_IRIS_TIMEOUT) || 30000
    };
  }

  loadEmbeddingConfig() {
    return {
      model: process.env.TEST_EMBEDDING_MODEL || 'Xenova/all-MiniLM-L6-v2',
      useReal: process.env.USE_REAL_EMBEDDINGS === 'true',
      dimension: parseInt(process.env.TEST_EMBEDDING_DIM) || 384,
      batchSize: parseInt(process.env.TEST_EMBEDDING_BATCH_SIZE) || 10
    };
  }

  loadPerformanceConfig() {
    return {
      enableMetrics: process.env.COLLECT_PERFORMANCE_METRICS === 'true',
      documentCount: parseInt(process.env.TEST_DOCUMENT_COUNT) || 10,
      timeoutMs: parseInt(process.env.TEST_TIMEOUT_MS) || 30000,
      maxMemoryMB: parseInt(process.env.TEST_MAX_MEMORY_MB) || 512
    };
  }

  validateConfiguration() {
    // Validate IRIS config
    if (this.irisConfig.port < 1 || this.irisConfig.port > 65535) {
      throw new Error(`Invalid IRIS port: ${this.irisConfig.port}`);
    }

    // Validate embedding config
    if (this.embeddingConfig.dimension < 1) {
      throw new Error(`Invalid embedding dimension: ${this.embeddingConfig.dimension}`);
    }

    // Validate performance config
    if (this.performanceConfig.documentCount < 1) {
      throw new Error(`Invalid document count: ${this.performanceConfig.documentCount}`);
    }
  }

  setupGlobalTestVariables() {
    global.TEST_CONFIG = {
      iris: this.irisConfig,
      embedding: this.embeddingConfig,
      performance: this.performanceConfig,
      environment: this.testEnvironment
    };

    global.TEST_DOCUMENTS = this.createTestDocuments();
    global.TEST_VECTORS = this.createTestVectors();
  }

  createTestDocuments() {
    return [
      {
        docId: 'test_doc_001',
        title: 'InterSystems IRIS Vector Search',
        content: 'InterSystems IRIS provides native vector search capabilities for similarity matching and retrieval-augmented generation workflows.',
        sourceFile: 'iris_documentation.pdf',
        pageNumber: 1,
        chunkIndex: 0
      },
      {
        docId: 'test_doc_002', 
        title: 'Machine Learning with IRIS',
        content: 'IRIS supports machine learning workflows with built-in Python integration and AutoML capabilities for advanced analytics.',
        sourceFile: 'iris_ml_guide.pdf',
        pageNumber: 2,
        chunkIndex: 0
      },
      {
        docId: 'test_doc_003',
        title: 'Vector Database Operations',
        content: 'Vector databases enable semantic similarity search using cosine distance and other vector operations for AI applications.',
        sourceFile: 'vector_db_manual.pdf',
        pageNumber: 1,
        chunkIndex: 1
      }
    ];
  }

  createTestVectors() {
    const dimension = this.embeddingConfig.dimension;
    return {
      query_vector: Array(dimension).fill(0).map((_, i) => Math.sin(i * 0.1)),
      doc_vectors: [
        Array(dimension).fill(0).map((_, i) => Math.cos(i * 0.1)),
        Array(dimension).fill(0).map((_, i) => Math.sin(i * 0.2)),
        Array(dimension).fill(0).map((_, i) => Math.cos(i * 0.2))
      ]
    };
  }
}

/**
 * Mock IRIS Connection
 * Simulates IRIS database operations for testing
 */
class MockIrisConnection {
  constructor() {
    this.mockResults = new Map();
    this.queryHistory = [];
    this.isConnected = false;
    this.shouldThrowError = false;
    this.errorMessage = 'Mock database error';
  }

  setMockResult(operation, result) {
    this.mockResults.set(operation, result);
  }

  setErrorMode(shouldThrow, message = 'Mock database error') {
    this.shouldThrowError = shouldThrow;
    this.errorMessage = message;
  }

  async connect() {
    if (this.shouldThrowError) {
      throw new Error(this.errorMessage);
    }
    this.isConnected = true;
    return this;
  }

  async executeQuery(sql, params = []) {
    this.queryHistory.push({ sql, params, timestamp: Date.now() });

    if (this.shouldThrowError) {
      throw new Error(this.errorMessage);
    }

    // Pattern matching for different query types
    if (sql.includes('VECTOR_COSINE')) {
      return this.mockResults.get('vector_search') || [];
    }
    
    if (sql.includes('INSERT INTO SourceDocuments')) {
      return this.mockResults.get('insert_document') || { rowsAffected: 1 };
    }

    if (sql.includes('UPDATE SourceDocuments')) {
      return this.mockResults.get('update_document') || { rowsAffected: 1 };
    }

    if (sql.includes('DELETE FROM SourceDocuments')) {
      return this.mockResults.get('delete_document') || { rowsAffected: 1 };
    }

    if (sql.includes('COUNT(*)')) {
      return this.mockResults.get('count_documents') || [[10]];
    }

    return [];
  }

  async close() {
    this.isConnected = false;
  }

  getQueryHistory() {
    return this.queryHistory;
  }

  clearHistory() {
    this.queryHistory = [];
  }
}

/**
 * Mock Embedding Utils
 * Simulates embedding generation for testing
 */
class MockEmbeddingUtils {
  constructor(modelName = 'mock-model') {
    this.modelName = modelName;
    this.embeddingDimension = 384;
    this.isInitialized = false;
    this.deterministicMode = true;
    this.generationHistory = [];
  }

  setEmbeddingDimension(dimension) {
    this.embeddingDimension = dimension;
  }

  setDeterministicMode(enabled) {
    this.deterministicMode = enabled;
  }

  async initialize() {
    this.isInitialized = true;
  }

  async generateEmbedding(text) {
    this.generationHistory.push({ text, timestamp: Date.now() });

    if (!this.isInitialized) {
      throw new Error('Embedding utils not initialized');
    }

    if (!text || typeof text !== 'string') {
      throw new Error('Text input is required and must be a string');
    }

    // Generate deterministic embeddings based on text hash
    if (this.deterministicMode) {
      const hash = this.simpleHash(text);
      return Array(this.embeddingDimension).fill(0).map((_, i) => 
        Math.sin((hash + i) * 0.1) * 0.5
      );
    }

    // Generate random embeddings
    return Array(this.embeddingDimension).fill(0).map(() => 
      (Math.random() - 0.5) * 2
    );
  }

  async generateBatchEmbeddings(texts, batchSize = 10) {
    const embeddings = [];
    for (const text of texts) {
      embeddings.push(await this.generateEmbedding(text));
    }
    return embeddings;
  }

  simpleHash(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash);
  }

  getEmbeddingDimension() {
    return this.embeddingDimension;
  }

  isReady() {
    return this.isInitialized;
  }

  getModelInfo() {
    return {
      modelName: this.modelName,
      isInitialized: this.isInitialized,
      embeddingDimension: this.embeddingDimension
    };
  }

  getGenerationHistory() {
    return this.generationHistory;
  }

  clearHistory() {
    this.generationHistory = [];
  }
}

// Initialize test configuration
const testConfig = new TestConfiguration();

// Export factory functions for fixtures
global.createMockIrisConnection = () => {
  const mock = new MockIrisConnection();
  
  // Setup default mock responses
  mock.setMockResult('vector_search', [
    ['test_doc_001', 'Sample content 1', 'test.pdf', 1, 0, 0.95],
    ['test_doc_002', 'Sample content 2', 'test.pdf', 1, 1, 0.87],
    ['test_doc_003', 'Sample content 3', 'test.pdf', 2, 0, 0.82]
  ]);
  
  mock.setMockResult('insert_document', { rowsAffected: 1, success: true });
  mock.setMockResult('update_document', { rowsAffected: 1, success: true });
  mock.setMockResult('delete_document', { rowsAffected: 1, success: true });
  mock.setMockResult('count_documents', [[25]]);
  
  return mock;
};

global.createMockEmbeddingUtils = () => {
  const mock = new MockEmbeddingUtils();
  mock.setEmbeddingDimension(TEST_CONFIG.embedding.dimension);
  mock.setDeterministicMode(true);
  return mock;
};

// Export classes for direct use
module.exports = {
  TestConfiguration,
  MockIrisConnection,
  MockEmbeddingUtils,
  createMockIrisConnection: global.createMockIrisConnection,
  createMockEmbeddingUtils: global.createMockEmbeddingUtils
};