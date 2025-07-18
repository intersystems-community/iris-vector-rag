// rag-templates/nodejs/tests/unit/db_vector_search.test.js
// Unit tests for VectorSearchClient

const { VectorSearchClient } = require('../../src/db_vector_search');
const VectorSQLUtils = require('../../src/vector_sql_utils');

describe('VectorSearchClient', () => {
  let vectorSearchClient;
  let mockConnectionManager;

  beforeEach(() => {
    // Create mock connection manager
    mockConnectionManager = {
      executeQuery: jest.fn()
    };
    
    vectorSearchClient = new VectorSearchClient(mockConnectionManager);
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('constructor', () => {
    test('should create instance with connection manager', () => {
      expect(vectorSearchClient.connectionManager).toBe(mockConnectionManager);
    });

    test('should throw error if no connection manager provided', () => {
      expect(() => new VectorSearchClient()).toThrow('Connection manager is required');
    });
  });

  describe('searchSourceDocuments', () => {
    const mockQueryVector = [0.1, 0.2, 0.3, 0.4];
    const mockResults = [
      ['doc1', 'Sample content 1', 'file1.pdf', 1, 0, 0.95],
      ['doc2', 'Sample content 2', 'file2.pdf', 2, 1, 0.87]
    ];

    beforeEach(() => {
      mockConnectionManager.executeQuery.mockResolvedValue(mockResults);
    });

    test('should perform vector search with default parameters', async () => {
      const results = await vectorSearchClient.searchSourceDocuments(mockQueryVector);

      expect(mockConnectionManager.executeQuery).toHaveBeenCalledTimes(1);
      expect(results).toHaveLength(2);
      expect(results[0]).toEqual({
        docId: 'doc1',
        textContent: 'Sample content 1',
        sourceFile: 'file1.pdf',
        pageNumber: 1,
        chunkIndex: 0,
        score: 0.95
      });
    });

    test('should perform vector search with custom topK', async () => {
      await vectorSearchClient.searchSourceDocuments(mockQueryVector, 5);

      const sqlCall = mockConnectionManager.executeQuery.mock.calls[0][0];
      expect(sqlCall).toContain('SELECT TOP 5');
    });

    test('should perform vector search with additional WHERE clause', async () => {
      const additionalWhere = "source_file = 'specific.pdf'";
      await vectorSearchClient.searchSourceDocuments(mockQueryVector, 10, additionalWhere);

      const sqlCall = mockConnectionManager.executeQuery.mock.calls[0][0];
      expect(sqlCall).toContain(additionalWhere);
    });

    test('should handle empty results', async () => {
      mockConnectionManager.executeQuery.mockResolvedValue([]);
      
      const results = await vectorSearchClient.searchSourceDocuments(mockQueryVector);
      expect(results).toEqual([]);
    });

    test('should handle database errors', async () => {
      mockConnectionManager.executeQuery.mockRejectedValue(new Error('Database connection failed'));

      await expect(vectorSearchClient.searchSourceDocuments(mockQueryVector))
        .rejects.toThrow('Vector search failed: Database connection failed');
    });

    test('should validate query vector format', async () => {
      await expect(vectorSearchClient.searchSourceDocuments(null))
        .rejects.toThrow('Query vector is required');

      await expect(vectorSearchClient.searchSourceDocuments([]))
        .rejects.toThrow('Query vector cannot be empty');

      await expect(vectorSearchClient.searchSourceDocuments(['invalid']))
        .rejects.toThrow('Query vector must contain only numbers');
    });

    test('should validate topK parameter', async () => {
      await expect(vectorSearchClient.searchSourceDocuments(mockQueryVector, 0))
        .rejects.toThrow('TopK must be a positive integer');

      await expect(vectorSearchClient.searchSourceDocuments(mockQueryVector, -1))
        .rejects.toThrow('TopK must be a positive integer');

      await expect(vectorSearchClient.searchSourceDocuments(mockQueryVector, 'invalid'))
        .rejects.toThrow('TopK must be a positive integer');
    });
  });

  describe('insertDocument', () => {
    const mockEmbedding = Array(384).fill(0).map((_, i) => i / 384);

    test('should insert document with all parameters', async () => {
      mockConnectionManager.executeQuery.mockResolvedValue([]);

      await vectorSearchClient.insertDocument(
        'doc1', 'Test Title', 'Test content', 'test.pdf', 1, 0, mockEmbedding
      );

      expect(mockConnectionManager.executeQuery).toHaveBeenCalledTimes(1);
      const [sql, params] = mockConnectionManager.executeQuery.mock.calls[0];
      
      expect(sql).toContain('INSERT INTO SourceDocuments');
      expect(sql).toContain('TO_VECTOR(?, \'FLOAT\', 384)');
      expect(params).toHaveLength(7);
      expect(params[0]).toBe('doc1');
      expect(params[1]).toBe('Test Title');
    });

    test('should handle database insertion errors', async () => {
      mockConnectionManager.executeQuery.mockRejectedValue(new Error('Duplicate key'));

      await expect(vectorSearchClient.insertDocument(
        'doc1', 'Test Title', 'Test content', 'test.pdf', 1, 0, mockEmbedding
      )).rejects.toThrow('Document insertion failed: Duplicate key');
    });

    test('should validate required parameters', async () => {
      await expect(vectorSearchClient.insertDocument())
        .rejects.toThrow('Document ID is required');

      await expect(vectorSearchClient.insertDocument('doc1'))
        .rejects.toThrow('Title is required');

      await expect(vectorSearchClient.insertDocument('doc1', 'title'))
        .rejects.toThrow('Text content is required');

      await expect(vectorSearchClient.insertDocument('doc1', 'title', 'content'))
        .rejects.toThrow('Source file is required');

      await expect(vectorSearchClient.insertDocument('doc1', 'title', 'content', 'file.pdf'))
        .rejects.toThrow('Page number is required');

      await expect(vectorSearchClient.insertDocument('doc1', 'title', 'content', 'file.pdf', 1))
        .rejects.toThrow('Chunk index is required');

      await expect(vectorSearchClient.insertDocument('doc1', 'title', 'content', 'file.pdf', 1, 0))
        .rejects.toThrow('Embedding is required');
    });

    test('should validate embedding format', async () => {
      await expect(vectorSearchClient.insertDocument(
        'doc1', 'title', 'content', 'file.pdf', 1, 0, []
      )).rejects.toThrow('Embedding cannot be empty');

      await expect(vectorSearchClient.insertDocument(
        'doc1', 'title', 'content', 'file.pdf', 1, 0, ['invalid']
      )).rejects.toThrow('Embedding must contain only numbers');
    });
  });

  describe('updateDocument', () => {
    const mockEmbedding = Array(384).fill(0).map((_, i) => i / 384);

    test('should update document content and embedding', async () => {
      mockConnectionManager.executeQuery.mockResolvedValue([]);

      await vectorSearchClient.updateDocument('doc1', 'Updated content', mockEmbedding);

      expect(mockConnectionManager.executeQuery).toHaveBeenCalledTimes(1);
      const [sql, params] = mockConnectionManager.executeQuery.mock.calls[0];
      
      expect(sql).toContain('UPDATE SourceDocuments');
      expect(sql).toContain('SET text_content = ?, embedding = TO_VECTOR(?, \'FLOAT\', 384)');
      expect(params).toEqual(['Updated content', expect.any(String), 'doc1']);
    });

    test('should handle update errors', async () => {
      mockConnectionManager.executeQuery.mockRejectedValue(new Error('Document not found'));

      await expect(vectorSearchClient.updateDocument('doc1', 'content', mockEmbedding))
        .rejects.toThrow('Document update failed: Document not found');
    });

    test('should validate update parameters', async () => {
      await expect(vectorSearchClient.updateDocument())
        .rejects.toThrow('Document ID is required');

      await expect(vectorSearchClient.updateDocument('doc1'))
        .rejects.toThrow('Text content is required');

      await expect(vectorSearchClient.updateDocument('doc1', 'content'))
        .rejects.toThrow('Embedding is required');
    });
  });

  describe('deleteDocument', () => {
    test('should delete document by ID', async () => {
      mockConnectionManager.executeQuery.mockResolvedValue([]);

      await vectorSearchClient.deleteDocument('doc1');

      expect(mockConnectionManager.executeQuery).toHaveBeenCalledWith(
        'DELETE FROM SourceDocuments WHERE doc_id = ?',
        ['doc1']
      );
    });

    test('should handle deletion errors', async () => {
      mockConnectionManager.executeQuery.mockRejectedValue(new Error('Database error'));

      await expect(vectorSearchClient.deleteDocument('doc1'))
        .rejects.toThrow('Document deletion failed: Database error');
    });

    test('should validate document ID', async () => {
      await expect(vectorSearchClient.deleteDocument())
        .rejects.toThrow('Document ID is required');

      await expect(vectorSearchClient.deleteDocument(''))
        .rejects.toThrow('Document ID is required');
    });
  });

  describe('getDocumentStats', () => {
    test('should return document statistics', async () => {
      const mockStatsResult = [[100, 25, '2024-01-01', '2024-12-31']];
      mockConnectionManager.executeQuery.mockResolvedValue(mockStatsResult);

      const stats = await vectorSearchClient.getDocumentStats();

      expect(stats).toEqual({
        totalDocuments: 100,
        totalFiles: 25,
        oldestDocument: '2024-01-01',
        newestDocument: '2024-12-31'
      });
    });

    test('should handle stats query errors', async () => {
      mockConnectionManager.executeQuery.mockRejectedValue(new Error('Query failed'));

      await expect(vectorSearchClient.getDocumentStats())
        .rejects.toThrow('Failed to get document stats: Query failed');
    });

    test('should handle empty database', async () => {
      const mockEmptyResult = [[0, 0, null, null]];
      mockConnectionManager.executeQuery.mockResolvedValue(mockEmptyResult);

      const stats = await vectorSearchClient.getDocumentStats();

      expect(stats).toEqual({
        totalDocuments: 0,
        totalFiles: 0,
        oldestDocument: null,
        newestDocument: null
      });
    });
  });

  describe('getDocumentsBySource', () => {
    const mockDocuments = [
      ['doc1', 'Title 1', 'Content 1', 1, 0, '2024-01-01'],
      ['doc2', 'Title 2', 'Content 2', 1, 1, '2024-01-02']
    ];

    test('should get documents by source file with default limit', async () => {
      mockConnectionManager.executeQuery.mockResolvedValue(mockDocuments);

      const results = await vectorSearchClient.getDocumentsBySource('test.pdf');

      expect(mockConnectionManager.executeQuery).toHaveBeenCalledWith(
        expect.stringContaining('WHERE source_file = ?'),
        ['test.pdf', 100]
      );
      expect(results).toHaveLength(2);
      expect(results[0]).toEqual({
        docId: 'doc1',
        title: 'Title 1',
        textContent: 'Content 1',
        pageNumber: 1,
        chunkIndex: 0,
        createdAt: '2024-01-01'
      });
    });

    test('should get documents by source file with custom limit', async () => {
      mockConnectionManager.executeQuery.mockResolvedValue(mockDocuments);

      await vectorSearchClient.getDocumentsBySource('test.pdf', 50);

      expect(mockConnectionManager.executeQuery).toHaveBeenCalledWith(
        expect.stringContaining('LIMIT ?'),
        ['test.pdf', 50]
      );
    });

    test('should handle source query errors', async () => {
      mockConnectionManager.executeQuery.mockRejectedValue(new Error('Query failed'));

      await expect(vectorSearchClient.getDocumentsBySource('test.pdf'))
        .rejects.toThrow('Failed to get documents by source: Query failed');
    });

    test('should validate source file parameter', async () => {
      await expect(vectorSearchClient.getDocumentsBySource())
        .rejects.toThrow('Source file is required');

      await expect(vectorSearchClient.getDocumentsBySource(''))
        .rejects.toThrow('Source file is required');
    });
  });

  describe('documentExists', () => {
    test('should return true if document exists', async () => {
      mockConnectionManager.executeQuery.mockResolvedValue([[1]]);

      const exists = await vectorSearchClient.documentExists('doc1');

      expect(exists).toBe(true);
      expect(mockConnectionManager.executeQuery).toHaveBeenCalledWith(
        'SELECT COUNT(*) FROM SourceDocuments WHERE doc_id = ?',
        ['doc1']
      );
    });

    test('should return false if document does not exist', async () => {
      mockConnectionManager.executeQuery.mockResolvedValue([[0]]);

      const exists = await vectorSearchClient.documentExists('nonexistent');

      expect(exists).toBe(false);
    });

    test('should handle existence check errors', async () => {
      mockConnectionManager.executeQuery.mockRejectedValue(new Error('Query failed'));

      await expect(vectorSearchClient.documentExists('doc1'))
        .rejects.toThrow('Failed to check document existence: Query failed');
    });

    test('should validate document ID parameter', async () => {
      await expect(vectorSearchClient.documentExists())
        .rejects.toThrow('Document ID is required');

      await expect(vectorSearchClient.documentExists(''))
        .rejects.toThrow('Document ID is required');
    });
  });

  describe('batchInsertDocuments', () => {
    const mockDocuments = [
      {
        docId: 'doc1',
        title: 'Title 1',
        textContent: 'Content 1',
        sourceFile: 'file1.pdf',
        pageNumber: 1,
        chunkIndex: 0,
        embedding: Array(384).fill(0.1)
      },
      {
        docId: 'doc2',
        title: 'Title 2',
        textContent: 'Content 2',
        sourceFile: 'file2.pdf',
        pageNumber: 1,
        chunkIndex: 0,
        embedding: Array(384).fill(0.2)
      }
    ];

    test('should batch insert multiple documents', async () => {
      mockConnectionManager.executeQuery.mockResolvedValue([]);

      await vectorSearchClient.batchInsertDocuments(mockDocuments);

      expect(mockConnectionManager.executeQuery).toHaveBeenCalledTimes(2);
      
      const firstCall = mockConnectionManager.executeQuery.mock.calls[0];
      expect(firstCall[0]).toContain('INSERT INTO SourceDocuments');
      expect(firstCall[1][0]).toBe('doc1');
    });

    test('should handle batch insertion errors', async () => {
      mockConnectionManager.executeQuery.mockRejectedValue(new Error('Batch failed'));

      await expect(vectorSearchClient.batchInsertDocuments(mockDocuments))
        .rejects.toThrow('Batch document insertion failed: Batch failed');
    });

    test('should validate documents array', async () => {
      await expect(vectorSearchClient.batchInsertDocuments())
        .rejects.toThrow('Documents array is required and must not be empty');

      await expect(vectorSearchClient.batchInsertDocuments([]))
        .rejects.toThrow('Documents array is required and must not be empty');

      await expect(vectorSearchClient.batchInsertDocuments('not-array'))
        .rejects.toThrow('Documents array is required and must not be empty');
    });

    test('should validate individual document structure', async () => {
      const invalidDocs = [{ docId: 'doc1' }]; // Missing required fields

      await expect(vectorSearchClient.batchInsertDocuments(invalidDocs))
        .rejects.toThrow('Invalid document structure');
    });

    test('should handle empty batch gracefully', async () => {
      await expect(vectorSearchClient.batchInsertDocuments([]))
        .rejects.toThrow('Documents array is required and must not be empty');
    });
  });

  describe('integration with VectorSQLUtils', () => {
    test('should use VectorSQLUtils for vector formatting', async () => {
      const mockVector = [0.1, 0.2, 0.3];
      const formatSpy = jest.spyOn(VectorSQLUtils, 'formatVectorForIris');
      mockConnectionManager.executeQuery.mockResolvedValue([]);

      await vectorSearchClient.insertDocument(
        'doc1', 'title', 'content', 'file.pdf', 1, 0, mockVector
      );

      expect(formatSpy).toHaveBeenCalledWith(mockVector);
    });

    test('should use VectorSQLUtils for SQL generation', async () => {
      const mockVector = [0.1, 0.2, 0.3];
      const sqlSpy = jest.spyOn(VectorSQLUtils, 'formatVectorSearchSQL');
      mockConnectionManager.executeQuery.mockResolvedValue([]);

      await vectorSearchClient.searchSourceDocuments(mockVector);

      expect(sqlSpy).toHaveBeenCalledWith({
        tableName: 'SourceDocuments',
        vectorColumn: 'embedding',
        vectorString: expect.any(String),
        embeddingDim: 384,
        topK: 10,
        idColumn: 'doc_id',
        contentColumn: 'text_content',
        additionalWhere: null
      });
    });
  });

  describe('error handling and edge cases', () => {
    test('should handle malformed database responses', async () => {
      mockConnectionManager.executeQuery.mockResolvedValue([['incomplete']]);

      await expect(vectorSearchClient.searchSourceDocuments([0.1, 0.2]))
        .rejects.toThrow('Malformed database response');
    });

    test('should handle connection timeouts', async () => {
      mockConnectionManager.executeQuery.mockRejectedValue(new Error('Connection timeout'));

      await expect(vectorSearchClient.getDocumentStats())
        .rejects.toThrow('Failed to get document stats: Connection timeout');
    });

    test('should handle very large vectors', async () => {
      const largeVector = Array(1536).fill(0.1); // Large embedding dimension
      mockConnectionManager.executeQuery.mockResolvedValue([]);

      await expect(vectorSearchClient.insertDocument(
        'doc1', 'title', 'content', 'file.pdf', 1, 0, largeVector
      )).resolves.not.toThrow();
    });

    test('should handle special characters in content', async () => {
      const specialContent = "Content with 'quotes' and \"double quotes\" and \n newlines";
      mockConnectionManager.executeQuery.mockResolvedValue([]);

      await expect(vectorSearchClient.insertDocument(
        'doc1', 'title', specialContent, 'file.pdf', 1, 0, Array(384).fill(0.1)
      )).resolves.not.toThrow();
    });
  });
});