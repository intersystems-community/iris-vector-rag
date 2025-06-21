// rag-templates/nodejs/tests/unit/index.test.js
// Unit tests for main index module

const {
  IrisConnectionManager,
  VectorSearchClient,
  VectorSQLUtils,
  EmbeddingUtils,
  createVectorSearchPipeline,
  createRAGPipeline
} = require('../../src/index');
const { TestConfiguration } = require('../conftest');

describe('Index Module', () => {
  let testConfig;

  beforeEach(() => {
    testConfig = new TestConfiguration();
  });

  describe('exports', () => {
    it('should export all individual components', () => {
      expect(IrisConnectionManager).toBeDefined();
      expect(VectorSearchClient).toBeDefined();
      expect(VectorSQLUtils).toBeDefined();
      expect(EmbeddingUtils).toBeDefined();
    });

    it('should export factory functions', () => {
      expect(createVectorSearchPipeline).toBeDefined();
      expect(createRAGPipeline).toBeDefined();
      expect(typeof createVectorSearchPipeline).toBe('function');
      expect(typeof createRAGPipeline).toBe('function');
    });

    it('should have createRAGPipeline as alias for createVectorSearchPipeline', () => {
      expect(createRAGPipeline).toBe(createVectorSearchPipeline);
    });

    it('should export constructors that can be instantiated', () => {
      expect(() => new IrisConnectionManager()).not.toThrow();
      expect(() => new EmbeddingUtils()).not.toThrow();
      expect(() => new VectorSQLUtils()).not.toThrow();
    });
  });

  describe('createVectorSearchPipeline', () => {
    let pipeline;

    afterEach(async () => {
      if (pipeline) {
        await pipeline.close();
      }
    });

    it('should create pipeline with default configuration', () => {
      pipeline = createVectorSearchPipeline();
      
      expect(pipeline).toBeDefined();
      expect(pipeline.connectionManager).toBeInstanceOf(IrisConnectionManager);
      expect(pipeline.vectorSearchClient).toBeInstanceOf(VectorSearchClient);
      expect(pipeline.embeddingUtils).toBeInstanceOf(EmbeddingUtils);
    });

    it('should create pipeline with custom configuration', () => {
      const config = {
        connection: testConfig.irisConfig,
        embeddingModel: 'custom-model'
      };
      
      pipeline = createVectorSearchPipeline(config);
      
      expect(pipeline.connectionManager.config.host).toBe(testConfig.irisConfig.host);
      expect(pipeline.embeddingUtils.modelName).toBe('custom-model');
    });

    it('should provide all expected methods', () => {
      pipeline = createVectorSearchPipeline();
      
      expect(typeof pipeline.search).toBe('function');
      expect(typeof pipeline.indexDocument).toBe('function');
      expect(typeof pipeline.indexDocuments).toBe('function');
      expect(typeof pipeline.processAndIndexDocument).toBe('function');
      expect(typeof pipeline.updateDocument).toBe('function');
      expect(typeof pipeline.deleteDocument).toBe('function');
      expect(typeof pipeline.getStats).toBe('function');
      expect(typeof pipeline.getDocumentsBySource).toBe('function');
      expect(typeof pipeline.documentExists).toBe('function');
      expect(typeof pipeline.initialize).toBe('function');
      expect(typeof pipeline.getInfo).toBe('function');
      expect(typeof pipeline.close).toBe('function');
    });

    describe('search method', () => {
      beforeEach(() => {
        pipeline = createVectorSearchPipeline({ connection: testConfig.irisConfig });
      });

      it('should perform basic search', async () => {
        const query = 'test search query';
        const mockResults = [
          { docId: 'doc1', content: 'result 1', similarity: 0.95 }
        ];

        // Mock the embedding generation and search
        jest.spyOn(pipeline.embeddingUtils, 'generateEmbedding')
          .mockResolvedValue([0.1, 0.2, 0.3]);
        jest.spyOn(pipeline.vectorSearchClient, 'searchSourceDocuments')
          .mockResolvedValue(mockResults);

        const results = await pipeline.search(query);

        expect(pipeline.embeddingUtils.generateEmbedding).toHaveBeenCalledWith(query);
        expect(pipeline.vectorSearchClient.searchSourceDocuments).toHaveBeenCalledWith(
          [0.1, 0.2, 0.3], 10, undefined
        );
        expect(results).toEqual(mockResults);
      });

      it('should perform search with custom options', async () => {
        const query = 'test search query';
        const options = {
          topK: 5,
          additionalWhere: "source_file = 'test.pdf'",
          minSimilarity: 0.8
        };

        jest.spyOn(pipeline.embeddingUtils, 'generateEmbedding')
          .mockResolvedValue([0.1, 0.2, 0.3]);
        jest.spyOn(pipeline.vectorSearchClient, 'searchSourceDocuments')
          .mockResolvedValue([]);

        await pipeline.search(query, options);

        expect(pipeline.vectorSearchClient.searchSourceDocuments).toHaveBeenCalledWith(
          [0.1, 0.2, 0.3], 
          5, 
          "source_file = 'test.pdf' AND score >= 0.8"
        );
      });

      it('should handle minimum similarity without additional WHERE', async () => {
        const query = 'test search query';
        const options = { minSimilarity: 0.7 };

        jest.spyOn(pipeline.embeddingUtils, 'generateEmbedding')
          .mockResolvedValue([0.1, 0.2, 0.3]);
        jest.spyOn(pipeline.vectorSearchClient, 'searchSourceDocuments')
          .mockResolvedValue([]);

        await pipeline.search(query, options);

        expect(pipeline.vectorSearchClient.searchSourceDocuments).toHaveBeenCalledWith(
          [0.1, 0.2, 0.3], 10, 'score >= 0.7'
        );
      });

      it('should validate query parameter', async () => {
        await expect(pipeline.search('')).rejects.toThrow('Query must be a non-empty string');
        await expect(pipeline.search(null)).rejects.toThrow('Query must be a non-empty string');
        await expect(pipeline.search(123)).rejects.toThrow('Query must be a non-empty string');
      });

      it('should handle search errors', async () => {
        const query = 'test search query';
        
        jest.spyOn(pipeline.embeddingUtils, 'generateEmbedding')
          .mockRejectedValue(new Error('Embedding generation failed'));

        await expect(pipeline.search(query)).rejects.toThrow('Search failed: Embedding generation failed');
      });
    });

    describe('indexDocument method', () => {
      beforeEach(() => {
        pipeline = createVectorSearchPipeline({ connection: testConfig.irisConfig });
      });

      it('should index a single document', async () => {
        const docId = 'test_doc_001';
        const title = 'Test Document';
        const content = 'This is test content';
        const sourceFile = 'test.pdf';
        const pageNumber = 1;
        const chunkIndex = 0;

        jest.spyOn(pipeline.embeddingUtils, 'generateEmbedding')
          .mockResolvedValue([0.1, 0.2, 0.3]);
        jest.spyOn(pipeline.vectorSearchClient, 'insertDocument')
          .mockResolvedValue({ success: true });

        const result = await pipeline.indexDocument(
          docId, title, content, sourceFile, pageNumber, chunkIndex
        );

        expect(pipeline.embeddingUtils.generateEmbedding).toHaveBeenCalledWith(content);
        expect(pipeline.vectorSearchClient.insertDocument).toHaveBeenCalledWith(
          docId, title, content, sourceFile, pageNumber, chunkIndex, [0.1, 0.2, 0.3]
        );
        expect(result).toEqual({ success: true });
      });

      it('should validate required parameters', async () => {
        await expect(pipeline.indexDocument('', 'title', 'content', 'file', 1, 0))
          .rejects.toThrow('Document ID and content are required');
        await expect(pipeline.indexDocument('id', 'title', '', 'file', 1, 0))
          .rejects.toThrow('Document ID and content are required');
        await expect(pipeline.indexDocument(null, 'title', 'content', 'file', 1, 0))
          .rejects.toThrow('Document ID and content are required');
      });

      it('should handle indexing errors', async () => {
        jest.spyOn(pipeline.embeddingUtils, 'generateEmbedding')
          .mockRejectedValue(new Error('Embedding failed'));

        await expect(pipeline.indexDocument('id', 'title', 'content', 'file', 1, 0))
          .rejects.toThrow('Document indexing failed: Embedding failed');
      });
    });

    describe('indexDocuments method', () => {
      beforeEach(() => {
        pipeline = createVectorSearchPipeline({ connection: testConfig.irisConfig });
      });

      it('should index multiple documents in batch', async () => {
        const documents = [
          { docId: 'doc1', title: 'Title 1', content: 'Content 1', sourceFile: 'file1.pdf', pageNumber: 1, chunkIndex: 0 },
          { docId: 'doc2', title: 'Title 2', content: 'Content 2', sourceFile: 'file2.pdf', pageNumber: 1, chunkIndex: 0 }
        ];

        const mockEmbeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];

        jest.spyOn(pipeline.embeddingUtils, 'generateBatchEmbeddings')
          .mockResolvedValue(mockEmbeddings);
        jest.spyOn(pipeline.vectorSearchClient, 'batchInsertDocuments')
          .mockResolvedValue({ success: true });

        const result = await pipeline.indexDocuments(documents);

        expect(pipeline.embeddingUtils.generateBatchEmbeddings).toHaveBeenCalledWith(
          ['Content 1', 'Content 2'], 10
        );
        expect(pipeline.vectorSearchClient.batchInsertDocuments).toHaveBeenCalledWith([
          {
            docId: 'doc1',
            title: 'Title 1',
            textContent: 'Content 1',
            sourceFile: 'file1.pdf',
            pageNumber: 1,
            chunkIndex: 0,
            embedding: [0.1, 0.2, 0.3]
          },
          {
            docId: 'doc2',
            title: 'Title 2',
            textContent: 'Content 2',
            sourceFile: 'file2.pdf',
            pageNumber: 1,
            chunkIndex: 0,
            embedding: [0.4, 0.5, 0.6]
          }
        ]);
        expect(result).toEqual({ success: true });
      });

      it('should use custom batch size', async () => {
        const documents = [
          { docId: 'doc1', content: 'Content 1' }
        ];

        jest.spyOn(pipeline.embeddingUtils, 'generateBatchEmbeddings')
          .mockResolvedValue([[0.1, 0.2, 0.3]]);
        jest.spyOn(pipeline.vectorSearchClient, 'batchInsertDocuments')
          .mockResolvedValue({ success: true });

        await pipeline.indexDocuments(documents, { batchSize: 5 });

        expect(pipeline.embeddingUtils.generateBatchEmbeddings).toHaveBeenCalledWith(
          ['Content 1'], 5
        );
      });

      it('should validate documents parameter', async () => {
        await expect(pipeline.indexDocuments(null))
          .rejects.toThrow('Documents array is required and must not be empty');
        await expect(pipeline.indexDocuments([]))
          .rejects.toThrow('Documents array is required and must not be empty');
        await expect(pipeline.indexDocuments('not an array'))
          .rejects.toThrow('Documents array is required and must not be empty');
      });

      it('should handle batch indexing errors', async () => {
        const documents = [{ docId: 'doc1', content: 'Content 1' }];
        
        jest.spyOn(pipeline.embeddingUtils, 'generateBatchEmbeddings')
          .mockRejectedValue(new Error('Batch embedding failed'));

        await expect(pipeline.indexDocuments(documents))
          .rejects.toThrow('Batch document indexing failed: Batch embedding failed');
      });
    });

    describe('processAndIndexDocument method', () => {
      beforeEach(() => {
        pipeline = createVectorSearchPipeline({ connection: testConfig.irisConfig });
      });

      it('should process and index document with chunking', async () => {
        const docId = 'test_doc';
        const title = 'Test Document';
        const text = 'This is a long document that will be chunked into smaller pieces for processing.';
        const sourceFile = 'test.pdf';
        const pageNumber = 1;

        // Mock static methods
        jest.spyOn(EmbeddingUtils, 'preprocessText')
          .mockReturnValue('preprocessed text');
        jest.spyOn(EmbeddingUtils, 'chunkText')
          .mockReturnValue(['chunk 1', 'chunk 2']);

        // Mock instance method
        jest.spyOn(pipeline, 'indexDocuments')
          .mockResolvedValue({ success: true });

        const result = await pipeline.processAndIndexDocument(
          docId, title, text, sourceFile, pageNumber
        );

        expect(EmbeddingUtils.preprocessText).toHaveBeenCalledWith(text, undefined);
        expect(EmbeddingUtils.chunkText).toHaveBeenCalledWith('preprocessed text', undefined);
        expect(pipeline.indexDocuments).toHaveBeenCalledWith([
          {
            docId: 'test_doc_chunk_0',
            title: 'Test Document (Chunk 1)',
            content: 'chunk 1',
            sourceFile: 'test.pdf',
            pageNumber: 1,
            chunkIndex: 0
          },
          {
            docId: 'test_doc_chunk_1',
            title: 'Test Document (Chunk 2)',
            content: 'chunk 2',
            sourceFile: 'test.pdf',
            pageNumber: 1,
            chunkIndex: 1
          }
        ], {});
        expect(result).toEqual(['test_doc_chunk_0', 'test_doc_chunk_1']);
      });

      it('should validate required parameters', async () => {
        await expect(pipeline.processAndIndexDocument('', 'title', 'text', 'file', 1))
          .rejects.toThrow('Document ID and text are required');
        await expect(pipeline.processAndIndexDocument('id', 'title', '', 'file', 1))
          .rejects.toThrow('Document ID and text are required');
      });

      it('should handle processing errors', async () => {
        jest.spyOn(EmbeddingUtils, 'preprocessText')
          .mockImplementation(() => { throw new Error('Preprocessing failed'); });

        await expect(pipeline.processAndIndexDocument('id', 'title', 'text', 'file', 1))
          .rejects.toThrow('Document processing and indexing failed: Preprocessing failed');
      });
    });

    describe('updateDocument method', () => {
      beforeEach(() => {
        pipeline = createVectorSearchPipeline({ connection: testConfig.irisConfig });
      });

      it('should update document with new content', async () => {
        const docId = 'test_doc';
        const newContent = 'Updated content';

        jest.spyOn(pipeline.embeddingUtils, 'generateEmbedding')
          .mockResolvedValue([0.1, 0.2, 0.3]);
        jest.spyOn(pipeline.vectorSearchClient, 'updateDocument')
          .mockResolvedValue({ success: true });

        const result = await pipeline.updateDocument(docId, newContent);

        expect(pipeline.embeddingUtils.generateEmbedding).toHaveBeenCalledWith(newContent);
        expect(pipeline.vectorSearchClient.updateDocument).toHaveBeenCalledWith(
          docId, newContent, [0.1, 0.2, 0.3]
        );
        expect(result).toEqual({ success: true });
      });

      it('should validate required parameters', async () => {
        await expect(pipeline.updateDocument('', 'content'))
          .rejects.toThrow('Document ID and new content are required');
        await expect(pipeline.updateDocument('id', ''))
          .rejects.toThrow('Document ID and new content are required');
      });

      it('should handle update errors', async () => {
        jest.spyOn(pipeline.embeddingUtils, 'generateEmbedding')
          .mockRejectedValue(new Error('Embedding failed'));

        await expect(pipeline.updateDocument('id', 'content'))
          .rejects.toThrow('Document update failed: Embedding failed');
      });
    });

    describe('utility methods', () => {
      beforeEach(() => {
        pipeline = createVectorSearchPipeline({ connection: testConfig.irisConfig });
      });

      it('should delete document', async () => {
        jest.spyOn(pipeline.vectorSearchClient, 'deleteDocument')
          .mockResolvedValue({ success: true });

        const result = await pipeline.deleteDocument('test_doc');

        expect(pipeline.vectorSearchClient.deleteDocument).toHaveBeenCalledWith('test_doc');
        expect(result).toEqual({ success: true });
      });

      it('should get statistics', async () => {
        const mockStats = { totalDocuments: 100, avgSimilarity: 0.85 };
        jest.spyOn(pipeline.vectorSearchClient, 'getDocumentStats')
          .mockResolvedValue(mockStats);

        const result = await pipeline.getStats();

        expect(result).toEqual(mockStats);
      });

      it('should get documents by source', async () => {
        const mockDocs = [{ docId: 'doc1', content: 'content1' }];
        jest.spyOn(pipeline.vectorSearchClient, 'getDocumentsBySource')
          .mockResolvedValue(mockDocs);

        const result = await pipeline.getDocumentsBySource('test.pdf', 50);

        expect(pipeline.vectorSearchClient.getDocumentsBySource).toHaveBeenCalledWith('test.pdf', 50);
        expect(result).toEqual(mockDocs);
      });

      it('should check document existence', async () => {
        jest.spyOn(pipeline.vectorSearchClient, 'documentExists')
          .mockResolvedValue(true);

        const result = await pipeline.documentExists('test_doc');

        expect(pipeline.vectorSearchClient.documentExists).toHaveBeenCalledWith('test_doc');
        expect(result).toBe(true);
      });

      it('should initialize embedding utils', async () => {
        jest.spyOn(pipeline.embeddingUtils, 'initialize')
          .mockResolvedValue();

        await pipeline.initialize();

        expect(pipeline.embeddingUtils.initialize).toHaveBeenCalled();
      });

      it('should get pipeline info', () => {
        const mockConnectionConfig = { host: 'localhost', port: 1972 };
        const mockEmbeddingInfo = { modelName: 'test-model', isInitialized: true };

        pipeline.connectionManager.config = mockConnectionConfig;
        jest.spyOn(pipeline.embeddingUtils, 'getModelInfo')
          .mockReturnValue(mockEmbeddingInfo);
        jest.spyOn(pipeline.embeddingUtils, 'isReady')
          .mockReturnValue(true);

        const info = pipeline.getInfo();

        expect(info).toEqual({
          connection: mockConnectionConfig,
          embedding: mockEmbeddingInfo,
          isReady: true
        });
      });

      it('should close pipeline', async () => {
        jest.spyOn(pipeline.connectionManager, 'close')
          .mockResolvedValue();

        await pipeline.close();

        expect(pipeline.connectionManager.close).toHaveBeenCalled();
      });
    });
  });

  describe('integration scenarios', () => {
    let pipeline;

    beforeEach(() => {
      pipeline = createVectorSearchPipeline({ connection: testConfig.irisConfig });
    });

    afterEach(async () => {
      if (pipeline) {
        await pipeline.close();
      }
    });

    it('should support complete RAG workflow', async () => {
      // Mock all dependencies
      jest.spyOn(pipeline.embeddingUtils, 'generateEmbedding')
        .mockResolvedValue([0.1, 0.2, 0.3]);
      jest.spyOn(pipeline.vectorSearchClient, 'insertDocument')
        .mockResolvedValue({ success: true });
      jest.spyOn(pipeline.vectorSearchClient, 'searchSourceDocuments')
        .mockResolvedValue([
          { docId: 'doc1', content: 'relevant content', similarity: 0.95 }
        ]);

      // Index a document
      await pipeline.indexDocument('doc1', 'Test Doc', 'test content', 'test.pdf', 1, 0);

      // Search for similar content
      const results = await pipeline.search('test query');

      expect(results).toHaveLength(1);
      expect(results[0].similarity).toBe(0.95);
    });

    it('should handle errors gracefully in workflow', async () => {
      // Test that errors are properly wrapped and propagated
      jest.spyOn(pipeline.embeddingUtils, 'generateEmbedding')
        .mockRejectedValue(new Error('Model not loaded'));

      await expect(pipeline.search('test query'))
        .rejects.toThrow('Search failed: Model not loaded');
      
      await expect(pipeline.indexDocument('doc1', 'title', 'content', 'file', 1, 0))
        .rejects.toThrow('Document indexing failed: Model not loaded');
    });
  });
});