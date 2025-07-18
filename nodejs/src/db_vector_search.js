// rag-templates/nodejs/src/db_vector_search.js
// Vector search client for IRIS database operations

const VectorSQLUtils = require('./vector_sql_utils');

class VectorSearchClient {
  constructor(connectionManager) {
    if (!connectionManager) {
      throw new Error('Connection manager is required');
    }
    this.connectionManager = connectionManager;
  }

  /**
   * Search for similar documents using vector similarity
   * @param {number[]} queryVector - The query vector for similarity search
   * @param {number} topK - Number of top results to return (default: 10)
   * @param {string|null} additionalWhere - Additional WHERE clause conditions
   * @returns {Promise<Array>} Array of search results with similarity scores
   */
  async searchSourceDocuments(queryVector, topK = 10, additionalWhere = null) {
    // Validate query vector
    if (!queryVector) {
      throw new Error('Query vector is required');
    }
    if (!Array.isArray(queryVector)) {
      throw new Error('Query vector must be an array');
    }
    if (queryVector.length === 0) {
      throw new Error('Query vector cannot be empty');
    }
    if (!queryVector.every(val => typeof val === 'number')) {
      throw new Error('Query vector must contain only numbers');
    }

    // Validate topK
    if (!Number.isInteger(topK) || topK <= 0) {
      throw new Error('TopK must be a positive integer');
    }

    const vectorString = VectorSQLUtils.formatVectorForIris(queryVector);
    
    const sql = VectorSQLUtils.formatVectorSearchSQL({
      tableName: 'SourceDocuments',
      vectorColumn: 'embedding',
      vectorString,
      embeddingDim: 384,
      topK,
      idColumn: 'doc_id',
      contentColumn: 'text_content',
      additionalWhere
    });

    try {
      const results = await this.connectionManager.executeQuery(sql);
      
      // Validate database response format
      return results.map(row => {
        if (!row || row.length < 6) {
          throw new Error('Malformed database response');
        }
        return {
          docId: row[0],
          textContent: row[1],
          sourceFile: row[2],
          pageNumber: row[3],
          chunkIndex: row[4],
          score: parseFloat(row[5])
        };
      });
    } catch (error) {
      throw new Error(`Vector search failed: ${error.message}`);
    }
  }

  /**
   * Insert a new document with its vector embedding
   * @param {string} docId - Unique document identifier
   * @param {string} title - Document title
   * @param {string} textContent - Document text content
   * @param {string} sourceFile - Source file name
   * @param {number} pageNumber - Page number in source
   * @param {number} chunkIndex - Chunk index within page
   * @param {number[]} embedding - Vector embedding for the document
   * @returns {Promise<void>}
   */
  async insertDocument(docId, title, textContent, sourceFile, pageNumber, chunkIndex, embedding) {
    // Validate required parameters
    if (!docId) {
      throw new Error('Document ID is required');
    }
    if (!title) {
      throw new Error('Title is required');
    }
    if (!textContent) {
      throw new Error('Text content is required');
    }
    if (!sourceFile) {
      throw new Error('Source file is required');
    }
    if (pageNumber === undefined || pageNumber === null) {
      throw new Error('Page number is required');
    }
    if (chunkIndex === undefined || chunkIndex === null) {
      throw new Error('Chunk index is required');
    }
    if (!embedding) {
      throw new Error('Embedding is required');
    }

    // Validate embedding format
    if (!Array.isArray(embedding)) {
      throw new Error('Embedding must be an array');
    }
    if (embedding.length === 0) {
      throw new Error('Embedding cannot be empty');
    }
    if (!embedding.every(val => typeof val === 'number')) {
      throw new Error('Embedding must contain only numbers');
    }

    const vectorString = VectorSQLUtils.formatVectorForIris(embedding);
    
    const sql = `
      INSERT INTO SourceDocuments 
      (doc_id, title, text_content, source_file, page_number, chunk_index, embedding, created_at)
      VALUES (?, ?, ?, ?, ?, ?, TO_VECTOR(?, 'FLOAT', 384), CURRENT_TIMESTAMP)
    `;

    try {
      await this.connectionManager.executeQuery(sql, [
        docId, title, textContent, sourceFile, pageNumber, chunkIndex, vectorString
      ]);
    } catch (error) {
      throw new Error(`Document insertion failed: ${error.message}`);
    }
  }

  /**
   * Update an existing document's content and embedding
   * @param {string} docId - Document identifier to update
   * @param {string} textContent - New text content
   * @param {number[]} embedding - New vector embedding
   * @returns {Promise<void>}
   */
  async updateDocument(docId, textContent, embedding) {
    // Validate required parameters
    if (!docId) {
      throw new Error('Document ID is required');
    }
    if (!textContent) {
      throw new Error('Text content is required');
    }
    if (!embedding) {
      throw new Error('Embedding is required');
    }

    const vectorString = VectorSQLUtils.formatVectorForIris(embedding);
    
    const sql = `
      UPDATE SourceDocuments 
      SET text_content = ?, embedding = TO_VECTOR(?, 'FLOAT', 384), updated_at = CURRENT_TIMESTAMP
      WHERE doc_id = ?
    `;

    try {
      await this.connectionManager.executeQuery(sql, [textContent, vectorString, docId]);
    } catch (error) {
      throw new Error(`Document update failed: ${error.message}`);
    }
  }

  /**
   * Delete a document by its ID
   * @param {string} docId - Document identifier to delete
   * @returns {Promise<void>}
   */
  async deleteDocument(docId) {
    // Validate document ID
    if (!docId) {
      throw new Error('Document ID is required');
    }

    const sql = `DELETE FROM SourceDocuments WHERE doc_id = ?`;

    try {
      await this.connectionManager.executeQuery(sql, [docId]);
    } catch (error) {
      throw new Error(`Document deletion failed: ${error.message}`);
    }
  }

  /**
   * Get statistics about the document collection
   * @returns {Promise<Object>} Statistics object with document counts and dates
   */
  async getDocumentStats() {
    const sql = `
      SELECT 
        COUNT(*) as total_documents,
        COUNT(DISTINCT source_file) as total_files,
        MIN(created_at) as oldest_document,
        MAX(created_at) as newest_document
      FROM SourceDocuments
    `;

    try {
      const result = await this.connectionManager.executeQuery(sql);
      return {
        totalDocuments: parseInt(result[0][0]),
        totalFiles: parseInt(result[0][1]),
        oldestDocument: result[0][2],
        newestDocument: result[0][3]
      };
    } catch (error) {
      throw new Error(`Failed to get document stats: ${error.message}`);
    }
  }

  /**
   * Get documents by source file
   * @param {string} sourceFile - Source file name to filter by
   * @param {number} limit - Maximum number of results (default: 100)
   * @returns {Promise<Array>} Array of documents from the specified source
   */
  async getDocumentsBySource(sourceFile, limit = 100) {
    // Validate source file
    if (!sourceFile) {
      throw new Error('Source file is required');
    }

    const sql = `
      SELECT doc_id, title, text_content, page_number, chunk_index, created_at
      FROM SourceDocuments 
      WHERE source_file = ?
      ORDER BY page_number, chunk_index
      LIMIT ?
    `;

    try {
      const results = await this.connectionManager.executeQuery(sql, [sourceFile, limit]);
      return results.map(row => ({
        docId: row[0],
        title: row[1],
        textContent: row[2],
        pageNumber: row[3],
        chunkIndex: row[4],
        createdAt: row[5]
      }));
    } catch (error) {
      throw new Error(`Failed to get documents by source: ${error.message}`);
    }
  }

  /**
   * Check if a document exists
   * @param {string} docId - Document identifier to check
   * @returns {Promise<boolean>} True if document exists, false otherwise
   */
  async documentExists(docId) {
    // Validate document ID
    if (!docId) {
      throw new Error('Document ID is required');
    }

    const sql = `SELECT COUNT(*) FROM SourceDocuments WHERE doc_id = ?`;

    try {
      const result = await this.connectionManager.executeQuery(sql, [docId]);
      return parseInt(result[0][0]) > 0;
    } catch (error) {
      throw new Error(`Failed to check document existence: ${error.message}`);
    }
  }

  /**
   * Batch insert multiple documents
   * @param {Array} documents - Array of document objects to insert
   * @returns {Promise<void>}
   */
  async batchInsertDocuments(documents) {
    if (!Array.isArray(documents) || documents.length === 0) {
      throw new Error('Documents array is required and must not be empty');
    }

    const sql = `
      INSERT INTO SourceDocuments 
      (doc_id, title, text_content, source_file, page_number, chunk_index, embedding, created_at)
      VALUES (?, ?, ?, ?, ?, ?, TO_VECTOR(?, 'FLOAT', 384), CURRENT_TIMESTAMP)
    `;

    try {
      for (const doc of documents) {
        // Validate individual document structure
        if (!doc.docId || !doc.title || !doc.textContent || !doc.sourceFile ||
            doc.pageNumber === undefined || doc.chunkIndex === undefined || !doc.embedding) {
          throw new Error('Invalid document structure');
        }

        const vectorString = VectorSQLUtils.formatVectorForIris(doc.embedding);
        await this.connectionManager.executeQuery(sql, [
          doc.docId, doc.title, doc.textContent, doc.sourceFile,
          doc.pageNumber, doc.chunkIndex, vectorString
        ]);
      }
    } catch (error) {
      throw new Error(`Batch document insertion failed: ${error.message}`);
    }
  }
}

module.exports = { VectorSearchClient };