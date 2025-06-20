// rag-templates/nodejs/tests/unit/embedding_utils.test.js
// Unit tests for EmbeddingUtils following TDD and rag-templates patterns

const EmbeddingUtils = require('../../src/embedding_utils');

describe('EmbeddingUtils', () => {
  let embeddingUtils;

  beforeEach(() => {
    embeddingUtils = new EmbeddingUtils('Xenova/all-MiniLM-L6-v2');
  });

  afterEach(() => {
    // Clean up any resources
    if (embeddingUtils && embeddingUtils.model) {
      embeddingUtils.model = null;
    }
  });

  describe('constructor', () => {
    test('should create instance with default model', () => {
      const utils = new EmbeddingUtils();
      expect(utils.modelName).toBe('Xenova/all-MiniLM-L6-v2');
      expect(utils.embeddingDimension).toBe(384);
      expect(utils.isInitialized).toBe(false);
    });

    test('should create instance with custom model', () => {
      const utils = new EmbeddingUtils('custom-model');
      expect(utils.modelName).toBe('custom-model');
      expect(utils.isInitialized).toBe(false);
    });

    test('should set correct embedding dimension for known models', () => {
      const utils = new EmbeddingUtils('Xenova/all-MiniLM-L6-v2');
      expect(utils.embeddingDimension).toBe(384);
    });
  });

  describe('initialization', () => {
    test('should initialize successfully with valid model', async () => {
      // Mock the pipeline function to avoid actual model loading in tests
      const mockPipeline = jest.fn().mockResolvedValue({
        model: 'mock-model',
        tokenizer: 'mock-tokenizer'
      });
      
      // Mock the transformers module
      jest.doMock('@xenova/transformers', () => ({
        pipeline: mockPipeline
      }));

      await embeddingUtils.initialize();
      
      expect(embeddingUtils.isInitialized).toBe(true);
      expect(embeddingUtils.model).toBeDefined();
    });

    test('should not reinitialize if already initialized', async () => {
      embeddingUtils.isInitialized = true;
      embeddingUtils.model = 'existing-model';
      
      const originalModel = embeddingUtils.model;
      await embeddingUtils.initialize();
      
      expect(embeddingUtils.model).toBe(originalModel);
    });

    test('should handle initialization errors gracefully', async () => {
      const embeddingUtils = new EmbeddingUtils('invalid-model');
      embeddingUtils._shouldFailInitialization = true;

      await expect(embeddingUtils.initialize()).rejects.toThrow('Failed to load embedding model');
    });
  });

  describe('generateEmbedding', () => {
    beforeEach(() => {
      // Mock successful initialization
      embeddingUtils.isInitialized = true;
      embeddingUtils.model = jest.fn().mockResolvedValue({
        data: new Float32Array(384).fill(0.1)
      });
    });

    test('should generate embedding for valid text', async () => {
      const text = 'This is a test sentence.';
      const embedding = await embeddingUtils.generateEmbedding(text);
      
      expect(Array.isArray(embedding)).toBe(true);
      expect(embedding.length).toBe(384);
      expect(embedding.every(val => typeof val === 'number')).toBe(true);
    });

    test('should generate embedding for InterSystems documentation text', async () => {
      const irisText = 'InterSystems IRIS is a complete data platform that makes it easier to build high-performance, machine learning-enabled applications.';
      const embedding = await embeddingUtils.generateEmbedding(irisText);
      
      expect(Array.isArray(embedding)).toBe(true);
      expect(embedding.length).toBe(384);
      expect(embedding.every(val => typeof val === 'number')).toBe(true);
    });

    test('should generate different embeddings for different texts', async () => {
      embeddingUtils.model = jest.fn()
        .mockResolvedValueOnce({ data: new Float32Array(384).fill(0.1) })
        .mockResolvedValueOnce({ data: new Float32Array(384).fill(0.2) });

      const text1 = 'Machine learning algorithms';
      const text2 = 'Database optimization techniques';
      
      const embedding1 = await embeddingUtils.generateEmbedding(text1);
      const embedding2 = await embeddingUtils.generateEmbedding(text2);
      
      expect(embedding1).not.toEqual(embedding2);
    });

    test('should handle empty text input', async () => {
      await expect(embeddingUtils.generateEmbedding('')).rejects.toThrow('Text input is required');
    });

    test('should handle null/undefined text input', async () => {
      await expect(embeddingUtils.generateEmbedding(null)).rejects.toThrow('Text input is required');
      await expect(embeddingUtils.generateEmbedding(undefined)).rejects.toThrow('Text input is required');
    });

    test('should handle non-string input', async () => {
      await expect(embeddingUtils.generateEmbedding(123)).rejects.toThrow('Text input is required');
      await expect(embeddingUtils.generateEmbedding({})).rejects.toThrow('Text input is required');
    });

    test('should auto-initialize if not initialized', async () => {
      embeddingUtils.isInitialized = false;
      embeddingUtils.initialize = jest.fn().mockResolvedValue();
      embeddingUtils.model = jest.fn().mockResolvedValue({
        data: new Float32Array(384).fill(0.1)
      });

      await embeddingUtils.generateEmbedding('test text');
      
      expect(embeddingUtils.initialize).toHaveBeenCalled();
    });
  });

  describe('generateBatchEmbeddings', () => {
    beforeEach(() => {
      embeddingUtils.isInitialized = true;
      embeddingUtils.generateEmbedding = jest.fn().mockImplementation((text) => 
        Promise.resolve(new Array(384).fill(text.length * 0.1))
      );
    });

    test('should generate embeddings for multiple texts', async () => {
      const texts = [
        'First document about machine learning',
        'Second document about database systems',
        'Third document about vector search'
      ];
      
      const embeddings = await embeddingUtils.generateBatchEmbeddings(texts, 2);
      
      expect(Array.isArray(embeddings)).toBe(true);
      expect(embeddings.length).toBe(3);
      embeddings.forEach(embedding => {
        expect(Array.isArray(embedding)).toBe(true);
        expect(embedding.length).toBe(384);
      });
    });

    test('should handle InterSystems documentation excerpts', async () => {
      const irisTexts = [
        'IRIS Data Platform provides comprehensive data management capabilities.',
        'Vector search in IRIS enables semantic similarity matching.',
        'InterSystems IRIS supports machine learning workflows.',
        'The IRIS vector index uses HNSW algorithm.'
      ];
      
      const embeddings = await embeddingUtils.generateBatchEmbeddings(irisTexts, 2);
      
      expect(embeddings.length).toBe(4);
      embeddings.forEach(embedding => {
        expect(embedding.length).toBe(384);
        expect(embedding.every(val => typeof val === 'number')).toBe(true);
      });
    });

    test('should handle empty array', async () => {
      await expect(embeddingUtils.generateBatchEmbeddings([])).rejects.toThrow('Texts array is required');
    });

    test('should handle non-array input', async () => {
      await expect(embeddingUtils.generateBatchEmbeddings('not an array')).rejects.toThrow('Texts array is required');
    });

    test('should process large batches efficiently', async () => {
      const texts = Array(10).fill(0).map((_, i) => `Document ${i} about various topics`);
      
      const startTime = Date.now();
      const embeddings = await embeddingUtils.generateBatchEmbeddings(texts, 3);
      const endTime = Date.now();
      
      expect(embeddings.length).toBe(10);
      expect(endTime - startTime).toBeLessThan(1000); // Should be fast with mocks
    });

    test('should handle batch processing errors', async () => {
      embeddingUtils.generateEmbedding = jest.fn()
        .mockResolvedValueOnce(new Array(384).fill(0.1))
        .mockRejectedValueOnce(new Error('Embedding failed'))
        .mockResolvedValueOnce(new Array(384).fill(0.3));

      const texts = ['text1', 'text2', 'text3'];
      
      await expect(embeddingUtils.generateBatchEmbeddings(texts)).rejects.toThrow('Batch embedding generation failed');
    });
  });

  describe('calculateCosineSimilarity', () => {
    test('should calculate similarity between identical vectors', () => {
      const vector = [1, 2, 3, 4];
      const similarity = EmbeddingUtils.calculateCosineSimilarity(vector, vector);
      expect(similarity).toBeCloseTo(1.0, 5);
    });

    test('should calculate similarity between orthogonal vectors', () => {
      const vector1 = [1, 0, 0];
      const vector2 = [0, 1, 0];
      const similarity = EmbeddingUtils.calculateCosineSimilarity(vector1, vector2);
      expect(similarity).toBeCloseTo(0.0, 5);
    });

    test('should calculate similarity between opposite vectors', () => {
      const vector1 = [1, 1, 1];
      const vector2 = [-1, -1, -1];
      const similarity = EmbeddingUtils.calculateCosineSimilarity(vector1, vector2);
      expect(similarity).toBeCloseTo(-1.0, 5);
    });

    test('should handle zero vectors', () => {
      const vector1 = [0, 0, 0];
      const vector2 = [1, 2, 3];
      const similarity = EmbeddingUtils.calculateCosineSimilarity(vector1, vector2);
      expect(similarity).toBe(0);
    });

    test('should validate input arrays', () => {
      expect(() => {
        EmbeddingUtils.calculateCosineSimilarity('not array', [1, 2, 3]);
      }).toThrow('Both inputs must be arrays');

      expect(() => {
        EmbeddingUtils.calculateCosineSimilarity([1, 2], [1, 2, 3]);
      }).toThrow('Vectors must have the same length');
    });

    test('should handle high-dimensional vectors', () => {
      const vector1 = Array(384).fill(0).map((_, i) => Math.sin(i * 0.1));
      const vector2 = Array(384).fill(0).map((_, i) => Math.cos(i * 0.1));
      
      const similarity = EmbeddingUtils.calculateCosineSimilarity(vector1, vector2);
      expect(typeof similarity).toBe('number');
      expect(similarity).toBeGreaterThanOrEqual(-1);
      expect(similarity).toBeLessThanOrEqual(1);
    });
  });

  describe('preprocessText', () => {
    test('should remove extra whitespace by default', () => {
      const text = 'This   has    extra   spaces\n\nand\t\ttabs';
      const processed = EmbeddingUtils.preprocessText(text);
      expect(processed).toBe('This has extra spaces and tabs');
    });

    test('should convert to lowercase when specified', () => {
      const text = 'This Is Mixed Case Text';
      const processed = EmbeddingUtils.preprocessText(text, { toLowerCase: true });
      expect(processed).toBe('this is mixed case text');
    });

    test('should remove special characters when specified', () => {
      const text = 'Text with @#$% special chars!';
      const processed = EmbeddingUtils.preprocessText(text, { removeSpecialChars: true });
      expect(processed).toBe('Text with  special chars!');
    });

    test('should truncate to max length', () => {
      const text = 'This is a very long text that should be truncated';
      const processed = EmbeddingUtils.preprocessText(text, { maxLength: 20 });
      expect(processed.length).toBeLessThanOrEqual(20);
      expect(processed).toBe('This is a very long');
    });

    test('should handle empty or null text', () => {
      expect(EmbeddingUtils.preprocessText('')).toBe('');
      expect(EmbeddingUtils.preprocessText(null)).toBe('');
      expect(EmbeddingUtils.preprocessText(undefined)).toBe('');
    });

    test('should process InterSystems documentation text', () => {
      const irisText = 'InterSystems IRIS™   provides   comprehensive\n\ndata management capabilities.';
      const processed = EmbeddingUtils.preprocessText(irisText, {
        removeExtraWhitespace: true,
        removeSpecialChars: true
      });
      expect(processed).toBe('InterSystems IRIS provides comprehensive data management capabilities.');
    });
  });

  describe('chunkText', () => {
    test('should chunk text by character count', () => {
      const text = 'This is a long text that needs to be chunked into smaller pieces for processing.';
      const chunks = EmbeddingUtils.chunkText(text, {
        chunkSize: 30,
        overlap: 5,
        splitOnSentences: false
      });
      
      expect(Array.isArray(chunks)).toBe(true);
      expect(chunks.length).toBeGreaterThan(1);
      chunks.forEach(chunk => {
        expect(chunk.length).toBeLessThanOrEqual(30);
      });
    });

    test('should chunk text by sentences', () => {
      const text = 'First sentence. Second sentence. Third sentence. Fourth sentence.';
      const chunks = EmbeddingUtils.chunkText(text, {
        chunkSize: 40,
        splitOnSentences: true
      });
      
      expect(Array.isArray(chunks)).toBe(true);
      chunks.forEach(chunk => {
        expect(chunk.trim()).toMatch(/\.$/); // Should end with period
      });
    });

    test('should handle InterSystems documentation text', () => {
      const irisText = `
        InterSystems IRIS is a complete data platform. It provides SQL and NoSQL capabilities. 
        The platform includes vector search functionality. This enables semantic similarity matching. 
        IRIS supports machine learning workflows. Python integration is built-in.
      `;
      
      const chunks = EmbeddingUtils.chunkText(irisText, {
        chunkSize: 100,
        overlap: 20,
        splitOnSentences: true
      });
      
      expect(chunks.length).toBeGreaterThan(1);
      chunks.forEach(chunk => {
        expect(chunk.trim().length).toBeGreaterThan(0);
      });
    });

    test('should handle empty text', () => {
      const chunks = EmbeddingUtils.chunkText('');
      expect(chunks).toEqual([]);
    });

    test('should handle text shorter than chunk size', () => {
      const text = 'Short text.';
      const chunks = EmbeddingUtils.chunkText(text, { chunkSize: 100 });
      expect(chunks).toEqual(['Short text.']);
    });
  });

  describe('utility methods', () => {
    test('should return correct embedding dimension', () => {
      expect(embeddingUtils.getEmbeddingDimension()).toBe(384);
    });

    test('should report ready status correctly', () => {
      embeddingUtils.isInitialized = true;
      expect(embeddingUtils.isReady()).toBe(true);
      
      embeddingUtils.isInitialized = false;
      expect(embeddingUtils.isReady()).toBe(false);
    });

    test('should return model information', () => {
      embeddingUtils.isInitialized = true;
      const info = embeddingUtils.getModelInfo();
      
      expect(info).toHaveProperty('modelName', 'Xenova/all-MiniLM-L6-v2');
      expect(info).toHaveProperty('isInitialized', true);
      expect(info).toHaveProperty('embeddingDimension', 384);
    });
  });

  describe('error handling', () => {
    test('should handle model loading failures gracefully', async () => {
      embeddingUtils.isInitialized = false;
      embeddingUtils.initialize = jest.fn().mockRejectedValue(new Error('Model load failed'));
      
      await expect(embeddingUtils.generateEmbedding('test')).rejects.toThrow();
    });

    test('should handle embedding generation failures', async () => {
      embeddingUtils.isInitialized = true;
      embeddingUtils.model = jest.fn().mockRejectedValue(new Error('Generation failed'));
      
      await expect(embeddingUtils.generateEmbedding('test')).rejects.toThrow('Embedding generation failed');
    });
  });
});