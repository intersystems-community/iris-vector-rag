// rag-templates/nodejs/src/index.js
// Main export module for IRIS RAG Node.js library

// Use HTTP IRIS connection manager in production, mock in tests
const IrisConnectionManager = process.env.NODE_ENV === 'test'
  ? require('./iris_connection_manager')  // Mock for tests
  : require('../../../support-tools-mcp/src/http_iris_connection_manager');  // HTTP-based for production
const { VectorSearchClient } = require('./db_vector_search');
const VectorSQLUtils = require('./vector_sql_utils');
const EmbeddingUtils = require('./embedding_utils');
const TableManager = require('./table_manager');
const DDLGenerator = require('./ddl_generator');

/**
 * Main factory function for creating a complete vector search pipeline
 * @param {Object} config - Configuration object
 * @param {Object} config.connection - IRIS connection configuration
 * @param {string} config.embeddingModel - Embedding model name (optional)
 * @returns {Object} Complete RAG pipeline with all components
 */
function createVectorSearchPipeline(config = {}) {
  const connectionManager = new IrisConnectionManager(config.connection);
  const vectorSearchClient = new VectorSearchClient(connectionManager);
  const embeddingUtils = new EmbeddingUtils(config.embeddingModel);
  
  // Initialize TableManager with relevant configuration
  const tableManagerOptions = {
    dialect: config.dialect || 'iris',
    schema: config.schema || null
  };
  const tableManager = new TableManager(connectionManager, tableManagerOptions);

  return {
    connectionManager,
    vectorSearchClient,
    embeddingUtils,
    tableManager,
    
    /**
     * Perform a complete search: generate embedding and search for similar documents
     * @param {string} query - Search query text
     * @param {Object} options - Search options
     * @param {number} options.topK - Number of results to return (default: 10)
     * @param {string} options.additionalWhere - Additional SQL WHERE conditions
     * @param {number} options.minSimilarity - Minimum similarity threshold
     * @returns {Promise<Array>} Search results with similarity scores
     */
    async search(query, options = {}) {
      if (!query || typeof query !== 'string') {
        throw new Error('Query must be a non-empty string');
      }

      try {
        const embedding = await embeddingUtils.generateEmbedding(query);
        let additionalWhere = options.additionalWhere;

        // Add minimum similarity filter if specified
        if (options.minSimilarity && typeof options.minSimilarity === 'number') {
          const similarityCondition = `score >= ${options.minSimilarity}`;
          additionalWhere = additionalWhere 
            ? `${additionalWhere} AND ${similarityCondition}`
            : similarityCondition;
        }

        return await vectorSearchClient.searchSourceDocuments(
          embedding, 
          options.topK || 10, 
          additionalWhere
        );
      } catch (error) {
        throw new Error(`Search failed: ${error.message}`);
      }
    },

    /**
     * Index a single document: generate embedding and store in database
     * @param {string} docId - Unique document identifier
     * @param {string} title - Document title
     * @param {string} content - Document text content
     * @param {string} sourceFile - Source file name
     * @param {number} pageNumber - Page number in source
     * @param {number} chunkIndex - Chunk index within page
     * @returns {Promise<void>}
     */
    async indexDocument(docId, title, content, sourceFile, pageNumber, chunkIndex) {
      if (!docId || !content) {
        throw new Error('Document ID and content are required');
      }

      try {
        const embedding = await embeddingUtils.generateEmbedding(content);
        return await vectorSearchClient.insertDocument(
          docId, title, content, sourceFile, pageNumber, chunkIndex, embedding
        );
      } catch (error) {
        throw new Error(`Document indexing failed: ${error.message}`);
      }
    },

    /**
     * Index multiple documents in batch
     * @param {Array} documents - Array of document objects
     * @param {Object} options - Batch processing options
     * @param {number} options.batchSize - Embedding batch size (default: 10)
     * @returns {Promise<void>}
     */
    async indexDocuments(documents, options = {}) {
      if (!Array.isArray(documents) || documents.length === 0) {
        throw new Error('Documents array is required and must not be empty');
      }

      try {
        // Extract text content for batch embedding generation
        const texts = documents.map(doc => doc.content);
        const embeddings = await embeddingUtils.generateBatchEmbeddings(
          texts, 
          options.batchSize || 10
        );

        // Prepare documents with embeddings for batch insertion
        const documentsWithEmbeddings = documents.map((doc, index) => ({
          docId: doc.docId,
          title: doc.title,
          textContent: doc.content,
          sourceFile: doc.sourceFile,
          pageNumber: doc.pageNumber,
          chunkIndex: doc.chunkIndex,
          embedding: embeddings[index]
        }));

        return await vectorSearchClient.batchInsertDocuments(documentsWithEmbeddings);
      } catch (error) {
        throw new Error(`Batch document indexing failed: ${error.message}`);
      }
    },

    /**
     * Process and index a text document by chunking it
     * @param {string} docId - Base document identifier
     * @param {string} title - Document title
     * @param {string} text - Full document text
     * @param {string} sourceFile - Source file name
     * @param {number} pageNumber - Page number in source
     * @param {Object} options - Processing options
     * @returns {Promise<Array>} Array of created chunk IDs
     */
    async processAndIndexDocument(docId, title, text, sourceFile, pageNumber, options = {}) {
      if (!docId || !text) {
        throw new Error('Document ID and text are required');
      }

      try {
        // Preprocess and chunk the text
        const preprocessedText = EmbeddingUtils.preprocessText(text, options.preprocessing);
        const chunks = EmbeddingUtils.chunkText(preprocessedText, options.chunking);

        // Prepare document chunks for indexing
        const documents = chunks.map((chunk, index) => ({
          docId: `${docId}_chunk_${index}`,
          title: `${title} (Chunk ${index + 1})`,
          content: chunk,
          sourceFile,
          pageNumber,
          chunkIndex: index
        }));

        // Index all chunks
        await this.indexDocuments(documents, options);

        return documents.map(doc => doc.docId);
      } catch (error) {
        throw new Error(`Document processing and indexing failed: ${error.message}`);
      }
    },

    /**
     * Update an existing document's content and embedding
     * @param {string} docId - Document identifier
     * @param {string} newContent - New document content
     * @returns {Promise<void>}
     */
    async updateDocument(docId, newContent) {
      if (!docId || !newContent) {
        throw new Error('Document ID and new content are required');
      }

      try {
        const embedding = await embeddingUtils.generateEmbedding(newContent);
        return await vectorSearchClient.updateDocument(docId, newContent, embedding);
      } catch (error) {
        throw new Error(`Document update failed: ${error.message}`);
      }
    },

    /**
     * Delete a document from the index
     * @param {string} docId - Document identifier to delete
     * @returns {Promise<void>}
     */
    async deleteDocument(docId) {
      return await vectorSearchClient.deleteDocument(docId);
    },

    /**
     * Get statistics about the document collection
     * @returns {Promise<Object>} Collection statistics
     */
    async getStats() {
      return await vectorSearchClient.getDocumentStats();
    },

    /**
     * Get documents from a specific source file
     * @param {string} sourceFile - Source file name
     * @param {number} limit - Maximum number of results
     * @returns {Promise<Array>} Documents from the source file
     */
    async getDocumentsBySource(sourceFile, limit = 100) {
      return await vectorSearchClient.getDocumentsBySource(sourceFile, limit);
    },

    /**
     * Check if a document exists in the index
     * @param {string} docId - Document identifier to check
     * @returns {Promise<boolean>} True if document exists
     */
    async documentExists(docId) {
      return await vectorSearchClient.documentExists(docId);
    },

    /**
     * Initialize the pipeline with optional table schema management
     * @param {Object} initOptions - Initialization options
     * @param {boolean} initOptions.ensureTableSchema - Whether to ensure table schema exists
     * @param {Object} initOptions.tableConfig - Table configuration for schema creation
     * @returns {Promise<void>}
     */
    async initialize(initOptions = {}) {
      // Handle table schema initialization if requested
      if (initOptions.ensureTableSchema === true) {
        const tableConfig = initOptions.tableConfig || {};
        
        try {
          // Ensure the table exists
          const ensureResult = await this.tableManager.ensureTable(tableConfig);
          
          // Validate schema if table was created or already existed
          if (ensureResult.created || ensureResult.existed) {
            const validationResult = await this.tableManager.validateSchema(
              ensureResult.tableName,
              tableConfig
            );
            
            if (!validationResult.valid) {
              const errorMessage = `Table schema validation failed for '${ensureResult.tableName}': ${validationResult.differences.join(', ')}`;
              throw new Error(errorMessage);
            }
          }
        } catch (error) {
          throw new Error(`Table schema initialization failed: ${error.message}`);
        }
      }
      
      // Initialize embedding utilities
      await embeddingUtils.initialize();
    },

    /**
     * Get information about the current configuration
     * @returns {Object} Configuration and status information
     */
    getInfo() {
      return {
        connection: connectionManager.config,
        embedding: embeddingUtils.getModelInfo(),
        isReady: embeddingUtils.isReady()
      };
    },

    /**
     * Close all connections and clean up resources
     * @returns {Promise<void>}
     */
    async close() {
      await connectionManager.close();
    }
  };
}

// Export all individual components and the factory function
module.exports = {
  // Individual components
  IrisConnectionManager,
  VectorSearchClient,
  VectorSQLUtils,
  EmbeddingUtils,
  TableManager,
  DDLGenerator,
  
  // Main factory function
  createVectorSearchPipeline,
  
  // Convenience alias
  createRAGPipeline: createVectorSearchPipeline
};