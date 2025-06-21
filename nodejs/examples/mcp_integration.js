// rag-templates/nodejs/examples/mcp_integration.js
// Example demonstrating MCP server integration with the RAG library

const { createVectorSearchPipeline } = require('../src/index');

/**
 * MCP-style tool implementation for IRIS vector search
 * This demonstrates how the RAG library can be integrated into an MCP server
 */
class IrisVectorSearchTool {
  constructor(config = {}) {
    this.pipeline = null;
    this.config = {
      connection: {
        host: config.host || process.env.IRIS_HOST || 'localhost',
        port: config.port || parseInt(process.env.IRIS_PORT) || 1972,
        namespace: config.namespace || process.env.IRIS_NAMESPACE || 'USER',
        username: config.username || process.env.IRIS_USERNAME || 'SuperUser',
        password: config.password || process.env.IRIS_PASSWORD || 'SYS'
      },
      embeddingModel: config.embeddingModel || 'Xenova/all-MiniLM-L6-v2'
    };
  }

  /**
   * Initialize the RAG pipeline (called once during MCP server startup)
   */
  async initialize() {
    if (this.pipeline) return;

    try {
      console.log('🔧 Initializing IRIS Vector Search Tool...');
      this.pipeline = createVectorSearchPipeline(this.config);
      await this.pipeline.initialize();
      console.log('✅ IRIS Vector Search Tool ready');
    } catch (error) {
      throw new Error(`Failed to initialize IRIS Vector Search Tool: ${error.message}`);
    }
  }

  /**
   * MCP tool: Search for similar documents
   * @param {Object} params - Search parameters
   * @param {string} params.query - Search query text
   * @param {number} params.maxResults - Maximum number of results (default: 5)
   * @param {string} params.sourceFile - Optional source file filter
   * @param {number} params.minSimilarity - Minimum similarity threshold
   * @returns {Promise<Object>} Search results
   */
  async searchDocuments(params) {
    if (!this.pipeline) {
      await this.initialize();
    }

    const { query, maxResults = 5, sourceFile, minSimilarity } = params;

    if (!query || typeof query !== 'string') {
      throw new Error('Query parameter is required and must be a string');
    }

    try {
      let additionalWhere = null;
      
      // Add source file filter if specified
      if (sourceFile) {
        additionalWhere = `source_file = '${sourceFile}'`;
      }

      const results = await this.pipeline.search(query, {
        topK: Math.min(maxResults, 20), // Limit to reasonable maximum
        additionalWhere,
        minSimilarity
      });

      return {
        success: true,
        query,
        resultsCount: results.length,
        results: results.map(result => ({
          documentId: result.docId,
          content: result.textContent,
          sourceFile: result.sourceFile,
          pageNumber: result.pageNumber,
          chunkIndex: result.chunkIndex,
          similarityScore: Math.round(result.score * 10000) / 10000 // Round to 4 decimal places
        }))
      };
    } catch (error) {
      return {
        success: false,
        error: `Search failed: ${error.message}`,
        query,
        resultsCount: 0,
        results: []
      };
    }
  }

  /**
   * MCP tool: Add a document to the vector index
   * @param {Object} params - Document parameters
   * @param {string} params.docId - Unique document identifier
   * @param {string} params.title - Document title
   * @param {string} params.content - Document content
   * @param {string} params.sourceFile - Source file name
   * @param {number} params.pageNumber - Page number (default: 1)
   * @param {number} params.chunkIndex - Chunk index (default: 0)
   * @returns {Promise<Object>} Operation result
   */
  async addDocument(params) {
    if (!this.pipeline) {
      await this.initialize();
    }

    const { 
      docId, 
      title, 
      content, 
      sourceFile, 
      pageNumber = 1, 
      chunkIndex = 0 
    } = params;

    if (!docId || !content || !sourceFile) {
      throw new Error('docId, content, and sourceFile parameters are required');
    }

    try {
      await this.pipeline.indexDocument(
        docId,
        title || 'Untitled Document',
        content,
        sourceFile,
        pageNumber,
        chunkIndex
      );

      return {
        success: true,
        message: `Document ${docId} successfully added to vector index`,
        documentId: docId
      };
    } catch (error) {
      return {
        success: false,
        error: `Failed to add document: ${error.message}`,
        documentId: docId
      };
    }
  }

  /**
   * MCP tool: Get collection statistics
   * @returns {Promise<Object>} Collection statistics
   */
  async getCollectionStats() {
    if (!this.pipeline) {
      await this.initialize();
    }

    try {
      const stats = await this.pipeline.getStats();
      
      return {
        success: true,
        statistics: {
          totalDocuments: stats.totalDocuments,
          totalSourceFiles: stats.totalFiles,
          oldestDocument: stats.oldestDocument,
          newestDocument: stats.newestDocument
        }
      };
    } catch (error) {
      return {
        success: false,
        error: `Failed to get statistics: ${error.message}`,
        statistics: null
      };
    }
  }

  /**
   * MCP tool: Process and index a long document with automatic chunking
   * @param {Object} params - Processing parameters
   * @param {string} params.docId - Base document identifier
   * @param {string} params.title - Document title
   * @param {string} params.content - Full document content
   * @param {string} params.sourceFile - Source file name
   * @param {number} params.pageNumber - Page number (default: 1)
   * @param {number} params.chunkSize - Chunk size in characters (default: 500)
   * @param {number} params.overlap - Overlap between chunks (default: 50)
   * @returns {Promise<Object>} Processing result
   */
  async processDocument(params) {
    if (!this.pipeline) {
      await this.initialize();
    }

    const { 
      docId, 
      title, 
      content, 
      sourceFile, 
      pageNumber = 1,
      chunkSize = 500,
      overlap = 50
    } = params;

    if (!docId || !content || !sourceFile) {
      throw new Error('docId, content, and sourceFile parameters are required');
    }

    try {
      const chunkIds = await this.pipeline.processAndIndexDocument(
        docId,
        title || 'Untitled Document',
        content,
        sourceFile,
        pageNumber,
        {
          chunking: {
            chunkSize,
            overlap,
            splitOnSentences: true
          },
          preprocessing: {
            removeExtraWhitespace: true,
            maxLength: chunkSize
          }
        }
      );

      return {
        success: true,
        message: `Document ${docId} processed and indexed into ${chunkIds.length} chunks`,
        documentId: docId,
        chunksCreated: chunkIds.length,
        chunkIds
      };
    } catch (error) {
      return {
        success: false,
        error: `Failed to process document: ${error.message}`,
        documentId: docId,
        chunksCreated: 0,
        chunkIds: []
      };
    }
  }

  /**
   * Clean up resources
   */
  async cleanup() {
    if (this.pipeline) {
      await this.pipeline.close();
      this.pipeline = null;
    }
  }
}

/**
 * Example MCP server integration demonstration
 */
async function mcpIntegrationExample() {
  console.log('🚀 Starting MCP Integration Example');

  const vectorTool = new IrisVectorSearchTool();

  try {
    // Initialize the tool
    await vectorTool.initialize();

    // Example 1: Add a document
    console.log('\n📝 Adding a sample document...');
    const addResult = await vectorTool.addDocument({
      docId: 'mcp_example_001',
      title: 'MCP Integration Guide',
      content: 'Model Context Protocol (MCP) enables seamless integration between AI assistants and external tools. This example demonstrates how to integrate IRIS vector search capabilities into an MCP server, providing powerful semantic search functionality.',
      sourceFile: 'mcp_guide.pdf',
      pageNumber: 1,
      chunkIndex: 0
    });
    console.log('Add result:', addResult);

    // Example 2: Process a longer document
    console.log('\n📄 Processing a longer document...');
    const longContent = `
      Vector databases are specialized database systems designed to store, index, and query high-dimensional vector data efficiently. 
      They are essential components in modern AI applications, particularly for similarity search, recommendation systems, and retrieval-augmented generation.
      
      IRIS provides native vector capabilities with support for various distance metrics including cosine similarity, Euclidean distance, and dot product.
      The vector indexing in IRIS uses advanced algorithms like HNSW (Hierarchical Navigable Small World) for fast approximate nearest neighbor search.
      
      Integration with MCP allows developers to expose vector search capabilities as tools that can be used by AI assistants and other applications.
      This creates powerful workflows where AI systems can dynamically search and retrieve relevant information from large document collections.
    `;

    const processResult = await vectorTool.processDocument({
      docId: 'vector_db_guide',
      title: 'Vector Database Guide',
      content: longContent,
      sourceFile: 'vector_db_comprehensive.pdf',
      pageNumber: 1,
      chunkSize: 300,
      overlap: 50
    });
    console.log('Process result:', processResult);

    // Example 3: Search for documents
    console.log('\n🔍 Searching for documents...');
    const searchResult = await vectorTool.searchDocuments({
      query: 'vector database indexing algorithms',
      maxResults: 3,
      minSimilarity: 0.1
    });
    console.log('Search result:', JSON.stringify(searchResult, null, 2));

    // Example 4: Get collection statistics
    console.log('\n📊 Getting collection statistics...');
    const statsResult = await vectorTool.getCollectionStats();
    console.log('Statistics result:', statsResult);

    // Example 5: Search with source file filter
    console.log('\n🎯 Searching with source file filter...');
    const filteredSearchResult = await vectorTool.searchDocuments({
      query: 'MCP integration',
      maxResults: 5,
      sourceFile: 'mcp_guide.pdf'
    });
    console.log('Filtered search result:', JSON.stringify(filteredSearchResult, null, 2));

  } catch (error) {
    console.error('❌ MCP Integration example failed:', error.message);
  } finally {
    // Clean up resources
    console.log('\n🧹 Cleaning up...');
    await vectorTool.cleanup();
    console.log('✅ MCP Integration example completed');
  }
}

// Run the example if this file is executed directly
if (require.main === module) {
  mcpIntegrationExample().catch(console.error);
}

module.exports = { IrisVectorSearchTool, mcpIntegrationExample };