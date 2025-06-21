/**
 * Simple API for RAG Templates Library Consumption Framework - JavaScript Implementation.
 * 
 * This module provides a zero-configuration Simple API that enables immediate RAG usage
 * with sensible defaults. The RAG class implements lazy initialization and provides
 * a clean, simple interface for document addition and querying.
 * 
 * Mirrors the Python Simple API functionality exactly.
 */

const { createVectorSearchPipeline } = require('./index');
const { ConfigManager } = require('./config-manager');

/**
 * Zero-configuration Simple API for RAG operations.
 * 
 * This class provides immediate RAG functionality with sensible defaults,
 * implementing lazy initialization to defer expensive operations until needed.
 * 
 * Example usage:
 *   // Zero-config initialization
 *   const rag = new RAG();
 *   
 *   // Add documents
 *   await rag.addDocuments(["Document 1 text", "Document 2 text"]);
 *   
 *   // Query for answers
 *   const answer = await rag.query("What is machine learning?");
 */
class RAG {
    /**
     * Initialize the Simple RAG API.
     * 
     * @param {string} configPath - Optional path to configuration file
     * @param {Object} options - Additional configuration overrides
     */
    constructor(configPath = null, options = {}) {
        this._configManager = null;
        this._pipeline = null;
        this._initialized = false;
        this._configPath = configPath;
        this._configOverrides = options;
        
        // Initialize configuration manager immediately (lightweight)
        try {
            this._configManager = new ConfigManager(configPath);
            
            // Apply any configuration overrides
            for (const [key, value] of Object.entries(options)) {
                if (key.includes(':')) {  // Support dot notation in options
                    this._configManager.set(key, value);
                }
            }
            
            console.log('Simple RAG API initialized with zero-configuration defaults');
            
        } catch (error) {
            throw new Error(
                `Failed to initialize Simple RAG API configuration: ${error.message}`
            );
        }
    }
    
    /**
     * Add documents to the RAG knowledge base.
     * 
     * @param {Array<string|Object>} documents - List of document texts or document objects
     * @param {Object} options - Additional options for document processing
     * @returns {Promise<void>}
     */
    async addDocuments(documents, options = {}) {
        try {
            // Ensure pipeline is initialized
            const pipeline = await this._getPipeline();
            
            // Convert string documents to proper format
            const processedDocs = this._processDocuments(documents);
            
            // Use the pipeline's indexDocuments method for batch processing
            await pipeline.indexDocuments(processedDocs, options);
            
            console.log(`Added ${documents.length} documents to knowledge base`);
            
        } catch (error) {
            throw new Error(`Failed to add documents: ${error.message}`);
        }
    }
    
    /**
     * Query the RAG system and return a simple string answer.
     * 
     * @param {string} queryText - The question or query text
     * @param {Object} options - Additional query options
     * @returns {Promise<string>} String answer to the query
     */
    async query(queryText, options = {}) {
        try {
            // Ensure pipeline is initialized
            const pipeline = await this._getPipeline();
            
            // Execute the search
            const results = await pipeline.search(queryText, {
                topK: options.top_k || 5,
                additionalWhere: options.source_filter,
                minSimilarity: options.min_similarity
            });
            
            // For Simple API, return a basic answer based on search results
            if (results && results.length > 0) {
                // Simple answer generation - concatenate top results
                const topResults = results.slice(0, 3);
                const context = topResults.map(r => r.textContent || r.content).join(' ');
                
                // Basic answer format
                const answer = `Based on the available information: ${context.substring(0, 500)}...`;
                
                console.log(`Query processed: ${queryText.substring(0, 50)}...`);
                return answer;
            } else {
                return "No relevant information found for your query.";
            }
            
        } catch (error) {
            const errorMsg = `Failed to process query: ${error.message}`;
            console.error(errorMsg);
            
            // Return a helpful error message instead of throwing
            return `Error: ${errorMsg}. Please check your configuration and try again.`;
        }
    }
    
    /**
     * Get or initialize the RAG pipeline using lazy initialization.
     * 
     * @returns {Promise<Object>} Initialized RAG pipeline instance
     */
    async _getPipeline() {
        if (!this._initialized) {
            await this._initializePipeline();
        }
        
        return this._pipeline;
    }
    
    /**
     * Initialize the RAG pipeline with lazy loading.
     * 
     * This method defers expensive operations like database connections
     * and model loading until actually needed.
     */
    async _initializePipeline() {
        try {
            // Get database configuration from config manager
            const dbConfig = this._configManager.getDatabaseConfig();
            const embeddingConfig = this._configManager.getEmbeddingConfig();
            
            // Create the vector search pipeline
            this._pipeline = createVectorSearchPipeline({
                connection: dbConfig,
                embeddingModel: embeddingConfig.model,
                dialect: 'iris'
            });
            
            // Initialize the pipeline
            await this._pipeline.initialize();
            
            this._initialized = true;
            console.log('RAG pipeline initialized successfully');
            
        } catch (error) {
            throw new Error(`Failed to initialize RAG pipeline: ${error.message}`);
        }
    }
    
    /**
     * Process input documents into the format expected by the pipeline.
     * 
     * @param {Array<string|Object>} documents - List of document texts or document objects
     * @returns {Array<Object>} List of processed document objects
     */
    _processDocuments(documents) {
        const processed = [];
        
        for (let i = 0; i < documents.length; i++) {
            const doc = documents[i];
            let processedDoc;
            
            if (typeof doc === 'string') {
                // Convert string to document format
                processedDoc = {
                    docId: `simple_api_doc_${i}`,
                    title: `Document ${i + 1}`,
                    content: doc,
                    sourceFile: 'simple_api_input',
                    pageNumber: 1,
                    chunkIndex: i
                };
            } else if (typeof doc === 'object' && doc !== null) {
                // Ensure required fields exist
                if (!doc.content && !doc.textContent) {
                    throw new Error(`Document ${i} missing 'content' or 'textContent' field`);
                }
                
                processedDoc = {
                    docId: doc.docId || `simple_api_doc_${i}`,
                    title: doc.title || `Document ${i + 1}`,
                    content: doc.content || doc.textContent,
                    sourceFile: doc.sourceFile || 'simple_api_input',
                    pageNumber: doc.pageNumber || 1,
                    chunkIndex: doc.chunkIndex || i
                };
            } else {
                throw new Error(`Document ${i} must be string or object, got ${typeof doc}`);
            }
            
            processed.push(processedDoc);
        }
        
        return processed;
    }
    
    /**
     * Get the number of documents in the knowledge base.
     * 
     * @returns {Promise<number>} Number of documents stored
     */
    async getDocumentCount() {
        try {
            if (!this._initialized) {
                return 0;
            }
            
            const stats = await this._pipeline.getStats();
            return stats.totalDocuments || 0;
        } catch (error) {
            console.warn(`Failed to get document count: ${error.message}`);
            return 0;
        }
    }
    
    /**
     * Clear all documents from the knowledge base.
     * 
     * Warning: This operation is irreversible.
     */
    async clearKnowledgeBase() {
        try {
            if (this._initialized && this._pipeline) {
                // Note: This would need to be implemented in the pipeline
                // For now, we'll just log a warning
                console.warn('clearKnowledgeBase not yet implemented in pipeline');
            }
        } catch (error) {
            throw new Error(`Failed to clear knowledge base: ${error.message}`);
        }
    }
    
    /**
     * Get a configuration value.
     * 
     * @param {string} key - Configuration key in dot notation (e.g., "database:iris:host")
     * @param {*} defaultValue - Default value if key not found
     * @returns {*} Configuration value or default
     */
    getConfig(key, defaultValue = null) {
        if (!this._configManager) {
            return defaultValue;
        }
        
        return this._configManager.get(key, defaultValue);
    }
    
    /**
     * Set a configuration value.
     * 
     * @param {string} key - Configuration key in dot notation
     * @param {*} value - Value to set
     */
    setConfig(key, value) {
        if (!this._configManager) {
            throw new Error('Configuration manager not initialized');
        }
        
        this._configManager.set(key, value);
        
        // If pipeline is already initialized, warn that restart may be needed
        if (this._initialized) {
            console.warn(`Configuration changed after initialization: ${key}. ` +
                        'Some changes may require restarting the application.');
        }
    }
    
    /**
     * Validate the current configuration.
     * 
     * @returns {Promise<boolean>} True if configuration is valid
     * @throws {Error} If validation fails
     */
    async validateConfig() {
        if (!this._configManager) {
            throw new Error('Configuration manager not initialized');
        }
        
        try {
            this._configManager.validate();
            return true;
        } catch (error) {
            throw new Error(`Configuration validation failed: ${error.message}`);
        }
    }
    
    /**
     * Return string representation of the RAG instance.
     * 
     * @returns {string} String representation
     */
    toString() {
        const status = this._initialized ? 'initialized' : 'not initialized';
        return `RAG(status=${status})`;
    }
}

module.exports = { RAG };