// rag-templates/nodejs/tests/integration/vector_search_pipeline.test.js
// Integration tests for the complete vector search pipeline

const { createVectorSearchPipeline } = require('../../src/index');
const { createMockIrisConnection, createMockEmbeddingUtils } = require('../conftest');

describe('Vector Search Pipeline Integration', () => {
  let pipeline;
  let mockConnection;
  let mockEmbedding;

  beforeEach(() => {
    mockConnection = createMockIrisConnection();
    mockEmbedding = createMockEmbeddingUtils();
    
    // Create pipeline with mocked dependencies
    pipeline = createVectorSearchPipeline({
      connection: {
        host: 'localhost',
        port: 1972,
        namespace: 'USER',
        username: 'test_user',
        password: 'test_pass'
      },
      embeddingModel: 'test-model'
    });

    // Replace real components with mocks
    pipeline.connectionManager.connect = jest.fn().mockResolvedValue(mockConnection);
    pipeline.embeddingUtils = mockEmbedding;
  });

  afterEach(async () => {
    if (pipeline) {
      await pipeline.close();
    }
  });

  describe('pipeline creation', () => {
    test('should create pipeline with default configuration', () => {
      const defaultPipeline = createVectorSearchPipeline();
      
      expect(defaultPipeline).toBeDefined();
      expect(defaultPipeline.connectionManager).toBeDefined();
      expect(defaultPipeline.vectorSearchClient).toBeDefined();
      expect(defaultPipeline.embeddingUtils).toBeDefined();
    });

    test('should create pipeline with custom configuration', () => {
      const customPipeline = createVectorSearchPipeline({
        connection: {
          host: 'custom-host',
          port: 9999,
          namespace: 'CUSTOM'
        },
        embeddingModel: 'custom-model'
      });
      
      expect(customPipeline.connectionManager.config.host).toBe('custom-host');
      expect(customPipeline.connectionManager.config.port).toBe(9999);
      expect(customPipeline.connectionManager.config.namespace).toBe('CUSTOM');
    });

    test('should provide all expected methods', () => {
      expect(typeof pipeline.search).toBe('function');
      expect(typeof pipeline.indexDocument).toBe('function');
      expect(typeof pipeline.indexDocuments).toBe('function');
      expect(typeof pipeline.processAndIndexDocument).toBe('function');
      expect(typeof pipeline.updateDocument).toBe('function');
      expect(typeof pipeline.deleteDocument).toBe('function');
      expect(typeof pipeline.getStats).toBe('function');
      expect(typeof pipeline.close).toBe('function');
    });
  });

  describe('document indexing workflow', () => {
    beforeEach(async () => {
      await mockEmbedding.initialize();
    });

    test('should index a single document successfully', async () => {
      const docId = 'test_doc_001';
      const title = 'Test Document';
      const content = 'This is a test document about InterSystems IRIS vector search capabilities.';
      const sourceFile = 'test.pdf';
      const pageNumber = 1;
      const chunkIndex = 0;

      await pipeline.indexDocument(docId, title, content, sourceFile, pageNumber, chunkIndex);

      // Verify embedding was generated
      expect(mockEmbedding.getGenerationHistory()).toHaveLength(1);
      expect(mockEmbedding.getGenerationHistory()[0].text).toBe(content);

      // Verify database insertion was called
      const queryHistory = mockConnection.getQueryHistory();
      expect(queryHistory).toHaveLength(1);
      expect(queryHistory[0].sql).toContain('INSERT INTO SourceDocuments');
    });

    test('should handle batch document indexing', async () => {
      const documents = [
        {
          docId: 'batch_doc_001',
          title: 'Batch Document 1',
          content: 'First document in batch about IRIS data platform.',
          sourceFile: 'batch.pdf',
          pageNumber: 1,
          chunkIndex: 0
        },
        {
          docId: 'batch_doc_002',
          title: 'Batch Document 2', 
          content: 'Second document in batch about vector search.',
          sourceFile: 'batch.pdf',
          pageNumber: 1,
          chunkIndex: 1
        },
        {
          docId: 'batch_doc_003',
          title: 'Batch Document 3',
          content: 'Third document in batch about machine learning.',
          sourceFile: 'batch.pdf',
          pageNumber: 2,
          chunkIndex: 0
        }
      ];

      await pipeline.indexDocuments(documents, { batchSize: 2 });

      // Verify embeddings were generated for all documents
      expect(mockEmbedding.getGenerationHistory()).toHaveLength(3);

      // Verify database insertions were called
      const queryHistory = mockConnection.getQueryHistory();
      expect(queryHistory).toHaveLength(3);
      queryHistory.forEach(query => {
        expect(query.sql).toContain('INSERT INTO SourceDocuments');
      });
    });

    test('should process and index long document with chunking', async () => {
      const longContent = `
        InterSystems IRIS is a complete data platform that makes it easier to build high-performance, 
        machine learning-enabled applications. IRIS provides native vector search capabilities for 
        similarity matching and retrieval-augmented generation workflows.
        
        The platform includes comprehensive data management with SQL and NoSQL capabilities. 
        Vector search in IRIS enables semantic similarity matching using cosine distance and 
        other vector operations for AI applications.
        
        IRIS supports machine learning workflows with built-in Python integration and AutoML 
        capabilities for advanced analytics. The vector index uses HNSW algorithm for efficient 
        approximate nearest neighbor search.
      `;

      const chunkIds = await pipeline.processAndIndexDocument(
        'long_doc_001',
        'Long IRIS Document',
        longContent,
        'iris_guide.pdf',
        1,
        {
          chunking: {
            chunkSize: 200,
            overlap: 50,
            splitOnSentences: true
          }
        }
      );

      expect(Array.isArray(chunkIds)).toBe(true);
      expect(chunkIds.length).toBeGreaterThan(1);
      
      // Verify multiple embeddings were generated
      expect(mockEmbedding.getGenerationHistory().length).toBeGreaterThan(1);
      
      // Verify multiple database insertions
      const queryHistory = mockConnection.getQueryHistory();
      expect(queryHistory.length).toBeGreaterThan(1);
    });

    test('should handle indexing errors gracefully', async () => {
      mockConnection.setErrorMode(true, 'Database connection failed');

      await expect(pipeline.indexDocument(
        'error_doc',
        'Error Document',
        'This should fail',
        'error.pdf',
        1,
        0
      )).rejects.toThrow('Document indexing failed');
    });
  });

  describe('search workflow', () => {
    beforeEach(async () => {
      await mockEmbedding.initialize();
      
      // Setup mock search results
      mockConnection.setMockResult('vector_search', [
        ['doc_001', 'IRIS provides vector search capabilities', 'iris.pdf', 1, 0, 0.95],
        ['doc_002', 'Machine learning with IRIS platform', 'ml.pdf', 2, 1, 0.87],
        ['doc_003', 'Vector databases enable semantic search', 'vector.pdf', 1, 2, 0.82]
      ]);
    });

    test('should perform complete search workflow', async () => {
      const query = 'vector search capabilities in IRIS';
      const results = await pipeline.search(query, { topK: 3 });

      // Verify embedding was generated for query
      expect(mockEmbedding.getGenerationHistory()).toHaveLength(1);
      expect(mockEmbedding.getGenerationHistory()[0].text).toBe(query);

      // Verify search was executed
      const queryHistory = mockConnection.getQueryHistory();
      expect(queryHistory).toHaveLength(1);
      expect(queryHistory[0].sql).toContain('VECTOR_COSINE');

      // Verify results format
      expect(Array.isArray(results)).toBe(true);
      expect(results).toHaveLength(3);
      
      results.forEach(result => {
        expect(result).toHaveProperty('docId');
        expect(result).toHaveProperty('textContent');
        expect(result).toHaveProperty('sourceFile');
        expect(result).toHaveProperty('pageNumber');
        expect(result).toHaveProperty('chunkIndex');
        expect(result).toHaveProperty('score');
        expect(typeof result.score).toBe('number');
      });
    });

    test('should handle search with filters', async () => {
      const query = 'machine learning';
      const results = await pipeline.search(query, {
        topK: 5,
        additionalWhere: "source_file = 'ml.pdf'",
        minSimilarity: 0.8
      });

      const queryHistory = mockConnection.getQueryHistory();
      expect(queryHistory[0].sql).toContain("source_file = 'ml.pdf'");
      expect(queryHistory[0].sql).toContain('score >= 0.8');
    });

    test('should handle empty search results', async () => {
      mockConnection.setMockResult('vector_search', []);

      const results = await pipeline.search('nonexistent query');
      
      expect(Array.isArray(results)).toBe(true);
      expect(results).toHaveLength(0);
    });

    test('should handle search errors gracefully', async () => {
      mockConnection.setErrorMode(true, 'Search query failed');

      await expect(pipeline.search('test query')).rejects.toThrow('Search failed');
    });
  });

  describe('document management workflow', () => {
    beforeEach(async () => {
      await mockEmbedding.initialize();
    });

    test('should update existing document', async () => {
      const docId = 'update_doc_001';
      const newContent = 'Updated content about IRIS vector capabilities';

      await pipeline.updateDocument(docId, newContent);

      // Verify new embedding was generated
      expect(mockEmbedding.getGenerationHistory()).toHaveLength(1);
      expect(mockEmbedding.getGenerationHistory()[0].text).toBe(newContent);

      // Verify update query was executed
      const queryHistory = mockConnection.getQueryHistory();
      expect(queryHistory).toHaveLength(1);
      expect(queryHistory[0].sql).toContain('UPDATE SourceDocuments');
    });

    test('should delete document', async () => {
      const docId = 'delete_doc_001';

      await pipeline.deleteDocument(docId);

      // Verify delete query was executed
      const queryHistory = mockConnection.getQueryHistory();
      expect(queryHistory).toHaveLength(1);
      expect(queryHistory[0].sql).toContain('DELETE FROM SourceDocuments');
      expect(queryHistory[0].params).toContain(docId);
    });

    test('should get collection statistics', async () => {
      mockConnection.setMockResult('count_documents', [[42]]);

      const stats = await pipeline.getStats();

      expect(stats).toHaveProperty('totalDocuments');
      expect(stats).toHaveProperty('totalFiles');
      expect(stats).toHaveProperty('oldestDocument');
      expect(stats).toHaveProperty('newestDocument');
    });

    test('should check document existence', async () => {
      mockConnection.setMockResult('count_documents', [[1]]);

      const exists = await pipeline.documentExists('test_doc_001');
      
      expect(typeof exists).toBe('boolean');
      expect(exists).toBe(true);
    });

    test('should get documents by source file', async () => {
      mockConnection.setMockResult('vector_search', [
        ['doc_001', 'Content 1', 'test.pdf', 1, 0, null],
        ['doc_002', 'Content 2', 'test.pdf', 1, 1, null]
      ]);

      const docs = await pipeline.getDocumentsBySource('test.pdf', 10);

      expect(Array.isArray(docs)).toBe(true);
      expect(docs).toHaveLength(2);
    });
  });

  describe('error handling and edge cases', () => {
    test('should handle connection failures', async () => {
      pipeline.connectionManager.connect = jest.fn().mockRejectedValue(new Error('Connection failed'));

      await expect(pipeline.search('test query')).rejects.toThrow();
    });

    test('should handle embedding generation failures', async () => {
      mockEmbedding.generateEmbedding = jest.fn().mockRejectedValue(new Error('Embedding failed'));

      await expect(pipeline.search('test query')).rejects.toThrow('Search failed');
    });

    test('should validate required parameters', async () => {
      await expect(pipeline.indexDocument()).rejects.toThrow('Document ID and content are required');
      await expect(pipeline.search()).rejects.toThrow('Query must be a non-empty string');
    });

    test('should handle resource cleanup', async () => {
      const closeSpy = jest.spyOn(pipeline.connectionManager, 'close');
      
      await pipeline.close();
      
      expect(closeSpy).toHaveBeenCalled();
    });
  });

  describe('performance considerations', () => {
    test('should handle large batch operations efficiently', async () => {
      const largeBatch = Array(50).fill(0).map((_, i) => ({
        docId: `perf_doc_${i}`,
        title: `Performance Document ${i}`,
        content: `This is performance test document number ${i} for testing batch operations.`,
        sourceFile: 'performance.pdf',
        pageNumber: Math.floor(i / 10) + 1,
        chunkIndex: i % 10
      }));

      const startTime = Date.now();
      await pipeline.indexDocuments(largeBatch, { batchSize: 10 });
      const endTime = Date.now();

      // Should complete within reasonable time (with mocks)
      expect(endTime - startTime).toBeLessThan(5000);
      
      // Verify all documents were processed
      expect(mockEmbedding.getGenerationHistory()).toHaveLength(50);
      expect(mockConnection.getQueryHistory()).toHaveLength(50);
    });

    test('should handle concurrent operations', async () => {
      const operations = [
        pipeline.search('query 1'),
        pipeline.search('query 2'),
        pipeline.search('query 3')
      ];

      const results = await Promise.all(operations);
      
      expect(results).toHaveLength(3);
      results.forEach(result => {
        expect(Array.isArray(result)).toBe(true);
      });
    });
  });
});