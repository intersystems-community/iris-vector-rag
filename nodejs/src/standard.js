/**
 * Standard API for RAG Templates Library Consumption Framework - JavaScript Implementation.
 * 
 * This module provides an advanced Standard API that enables configurable RAG usage
 * with technique selection, advanced configuration, and dependency injection while
 * maintaining backward compatibility with the Simple API.
 * 
 * Mirrors the Python Standard API functionality exactly.
 */

const { createVectorSearchPipeline } = require('./index');
const { ConfigManager } = require('./config-manager');

/**
 * Advanced Standard API for configurable RAG operations.
 * 
 * This class provides advanced RAG functionality with technique selection,
 * complex configuration support, and dependency injection while maintaining
 * the simplicity of the Simple API for basic use cases.
 * 
 * Example usage:
 *   // Basic technique selection
 *   const rag = new ConfigurableRAG({technique: "colbert"});
 *   
 *   // Advanced configuration
 *   const rag = new ConfigurableRAG({
 *     technique: "colbert",
 *     llm_provider: "anthropic",
 *     llm_config: {
 *       model: "claude-3-sonnet",
 *       temperature: 0.1
 *     },
 *     technique_config: {
 *       max_query_length: 512,
 *       top_k: 15
 *     }
 *   });
 *   
 *   // Query with advanced options
 *   const result = await rag.query("What is machine learning?", {
 *     include_sources: true,
 *     min_similarity: 0.8,
 *     max_results: 10
 *   });
 */
class ConfigurableRAG {
    /**
     * Initialize the Standard RAG API.
     * 
     * @param {Object} config - Configuration object with technique and options
     * @param {string} configPath - Optional path to configuration file
     */
    constructor(config, configPath = null) {
        this._config = { ...config };
        this._configPath = configPath;
        this._technique = (config.technique || 'basic').toLowerCase();
        this._pipeline = null;
        this._initialized = false;
        
        // Initialize core components
        try {
            // Create enhanced configuration manager
            this._configManager = new ConfigManager(configPath);
            
            // Apply configuration overrides from the config object
            this._applyConfigOverrides();
            
            // Initialize technique registry and pipeline factory (simplified for now)
            this._availableTechniques = ['basic', 'colbert', 'multi_query', 'hyde'];
            
            console.log(`Standard RAG API initialized with technique: ${this._technique}`);
            
        } catch (error) {
            throw new Error(
                `Failed to initialize Standard RAG API: ${error.message}`
            );
        }
    }
    
    /**
     * Apply configuration overrides from the config object.
     */
    _applyConfigOverrides() {
        // Map config keys to configuration manager paths
        const configMappings = {
            'llm_provider': 'llm:provider',
            'llm_model': 'llm:model',
            'llm_api_key': 'llm:api_key',
            'llm_temperature': 'llm:temperature',
            'llm_max_tokens': 'llm:max_tokens',
            'embedding_model': 'embeddings:model',
            'embedding_dimension': 'embeddings:dimension',
            'embedding_provider': 'embeddings:provider',
            'max_results': 'pipelines:basic:default_top_k',
            'chunk_size': 'pipelines:basic:chunk_size',
            'chunk_overlap': 'pipelines:basic:chunk_overlap'
        };
        
        // Apply direct mappings
        for (const [configKey, managerPath] of Object.entries(configMappings)) {
            if (configKey in this._config) {
                this._configManager.set(managerPath, this._config[configKey]);
            }
        }
        
        // Apply nested configurations
        if ('llm_config' in this._config) {
            const llmConfig = this._config.llm_config;
            for (const [key, value] of Object.entries(llmConfig)) {
                this._configManager.set(`llm:${key}`, value);
            }
        }
        
        if ('embedding_config' in this._config) {
            const embeddingConfig = this._config.embedding_config;
            for (const [key, value] of Object.entries(embeddingConfig)) {
                this._configManager.set(`embeddings:${key}`, value);
            }
        }
        
        if ('technique_config' in this._config) {
            const techniqueConfig = this._config.technique_config;
            const techniqueName = this._technique;
            for (const [key, value] of Object.entries(techniqueConfig)) {
                this._configManager.set(`pipelines:${techniqueName}:${key}`, value);
            }
        }
        
        if ('vector_index' in this._config) {
            const vectorConfig = this._config.vector_index;
            for (const [key, value] of Object.entries(vectorConfig)) {
                this._configManager.set(`vector_index:${key}`, value);
            }
        }
    }
    
    /**
     * Query the RAG system with advanced options.
     * 
     * @param {string} queryText - The question or query text
     * @param {Object} options - Advanced query options
     * @returns {Promise<string|Object>} String answer (default) or full result object
     */
    async query(queryText, options = {}) {
        try {
            // Ensure pipeline is initialized
            const pipeline = await this._getPipeline();
            
            // Prepare query options
            const queryOptions = { ...options };
            
            // Map standard options to pipeline parameters
            if ('max_results' in queryOptions) {
                queryOptions.topK = queryOptions.max_results;
                delete queryOptions.max_results;
            }
            
            // Execute the search
            const results = await pipeline.search(queryText, {
                topK: queryOptions.topK || 5,
                additionalWhere: queryOptions.source_filter,
                minSimilarity: queryOptions.min_similarity
            });
            
            // Determine return format
            const includeSourcesOrReturnDict = options.include_sources || options.return_dict;
            
            if (includeSourcesOrReturnDict) {
                // Return enhanced result object
                const enhancedResult = {
                    answer: this._generateAnswer(results),
                    query: queryText,
                    sources: this._extractSources(results),
                    metadata: this._extractMetadata(results),
                    technique: this._technique
                };
                
                // Add retrieved documents if available
                if (results && results.length > 0) {
                    enhancedResult.retrieved_documents = results;
                }
                
                return enhancedResult;
            } else {
                // Return simple string answer for backward compatibility
                return this._generateAnswer(results);
            }
            
        } catch (error) {
            const errorMsg = `Failed to process query with technique '${this._technique}': ${error.message}`;
            console.error(errorMsg);
            
            // Return error in appropriate format
            if (options && (options.include_sources || options.return_dict)) {
                return {
                    answer: `Error: ${errorMsg}`,
                    query: queryText,
                    error: error.message,
                    technique: this._technique
                };
            } else {
                return `Error: ${errorMsg}`;
            }
        }
    }
    
    /**
     * Add documents to the RAG knowledge base.
     * 
     * @param {Array<string|Object>} documents - List of document texts or document objects
     * @param {Object} options - Additional options for document processing
     */
    async addDocuments(documents, options = {}) {
        try {
            // Ensure pipeline is initialized
            const pipeline = await this._getPipeline();
            
            // Convert string documents to proper format
            const processedDocs = this._processDocuments(documents);
            
            // Use the pipeline's indexDocuments method
            await pipeline.indexDocuments(processedDocs, options);
            
            console.log(`Added ${documents.length} documents to knowledge base using ${this._technique}`);
            
        } catch (error) {
            throw new Error(
                `Failed to add documents with technique '${this._technique}': ${error.message}`
            );
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
     * Initialize the RAG pipeline with the selected technique.
     */
    async _initializePipeline() {
        try {
            // Validate technique is available
            if (!this._availableTechniques.includes(this._technique)) {
                throw new Error(
                    `Technique '${this._technique}' is not available. ` +
                    `Available techniques: ${this._availableTechniques.join(', ')}`
                );
            }
            
            // Get database configuration from config manager
            const dbConfig = this._configManager.getDatabaseConfig();
            const embeddingConfig = this._configManager.getEmbeddingConfig();
            
            // Create pipeline using the existing infrastructure
            // For now, we'll use the basic pipeline regardless of technique
            // In a full implementation, this would route to different technique implementations
            this._pipeline = createVectorSearchPipeline({
                connection: dbConfig,
                embeddingModel: embeddingConfig.model,
                dialect: 'iris'
            });
            
            // Initialize the pipeline
            await this._pipeline.initialize();
            
            this._initialized = true;
            console.log(`Pipeline initialized successfully for technique: ${this._technique}`);
            
        } catch (error) {
            throw new Error(
                `Failed to initialize pipeline for technique '${this._technique}': ${error.message}`
            );
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
                    docId: `standard_api_doc_${i}`,
                    title: `Document ${i + 1}`,
                    content: doc,
                    sourceFile: 'standard_api_input',
                    pageNumber: 1,
                    chunkIndex: i
                };
            } else if (typeof doc === 'object' && doc !== null) {
                // Ensure required fields exist
                if (!doc.content && !doc.textContent) {
                    throw new Error(`Document ${i} missing 'content' or 'textContent' field`);
                }
                
                processedDoc = {
                    docId: doc.docId || `standard_api_doc_${i}`,
                    title: doc.title || `Document ${i + 1}`,
                    content: doc.content || doc.textContent,
                    sourceFile: doc.sourceFile || 'standard_api_input',
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
     * Generate an answer from search results.
     * 
     * @param {Array} results - Search results from pipeline
     * @returns {string} Generated answer
     */
    _generateAnswer(results) {
        if (results && results.length > 0) {
            // Simple answer generation - concatenate top results
            const topResults = results.slice(0, 3);
            const context = topResults.map(r => r.textContent || r.content).join(' ');
            
            // Basic answer format
            return `Based on the available information: ${context.substring(0, 500)}...`;
        } else {
            return "No relevant information found for your query.";
        }
    }
    
    /**
     * Extract source information from pipeline result.
     * 
     * @param {Array} results - Search results
     * @returns {Array<Object>} Source information
     */
    _extractSources(results) {
        const sources = [];
        
        if (results && Array.isArray(results)) {
            for (const result of results) {
                const sourceInfo = {
                    content: result.textContent || result.content || '',
                    metadata: {
                        docId: result.docId,
                        sourceFile: result.sourceFile,
                        pageNumber: result.pageNumber,
                        chunkIndex: result.chunkIndex
                    },
                    score: result.score || null
                };
                sources.push(sourceInfo);
            }
        }
        
        return sources;
    }
    
    /**
     * Extract metadata from pipeline result.
     * 
     * @param {Array} results - Search results
     * @returns {Object} Metadata information
     */
    _extractMetadata(results) {
        const metadata = {
            technique: this._technique,
            num_retrieved: results ? results.length : 0,
            execution_time: null, // Would be populated by pipeline
            similarity_scores: []
        };
        
        // Extract similarity scores if available
        if (results && Array.isArray(results)) {
            for (const result of results) {
                if (result.score !== undefined) {
                    metadata.similarity_scores.push(result.score);
                }
            }
        }
        
        return metadata;
    }
    
    /**
     * Get list of available RAG techniques.
     * 
     * @returns {Array<string>} List of available technique names
     */
    getAvailableTechniques() {
        return [...this._availableTechniques];
    }
    
    /**
     * Get information about a technique.
     * 
     * @param {string} techniqueName - Name of technique (uses current if null)
     * @returns {Object} Technique information object
     */
    getTechniqueInfo(techniqueName = null) {
        const name = techniqueName || this._technique;
        
        // Basic technique info - in a full implementation this would come from a registry
        const techniqueInfo = {
            basic: {
                name: 'basic',
                description: 'Basic vector similarity search',
                supports_streaming: false,
                supports_metadata_filtering: true
            },
            colbert: {
                name: 'colbert',
                description: 'ColBERT late interaction retrieval',
                supports_streaming: false,
                supports_metadata_filtering: true
            },
            multi_query: {
                name: 'multi_query',
                description: 'Multi-query retrieval with query expansion',
                supports_streaming: false,
                supports_metadata_filtering: true
            },
            hyde: {
                name: 'hyde',
                description: 'Hypothetical Document Embeddings',
                supports_streaming: false,
                supports_metadata_filtering: true
            }
        };
        
        return techniqueInfo[name] || {};
    }
    
    /**
     * Switch to a different RAG technique.
     * 
     * @param {string} newTechnique - Name of the new technique
     * @param {Object} techniqueConfig - Optional configuration for the new technique
     */
    switchTechnique(newTechnique, techniqueConfig = null) {
        if (!this._availableTechniques.includes(newTechnique)) {
            throw new Error(
                `Technique '${newTechnique}' is not available. ` +
                `Available techniques: ${this._availableTechniques.join(', ')}`
            );
        }
        
        // Update configuration
        this._technique = newTechnique.toLowerCase();
        this._config.technique = this._technique;
        
        if (techniqueConfig) {
            this._config.technique_config = techniqueConfig;
            this._applyConfigOverrides();
        }
        
        // Reset pipeline to force reinitialization
        this._pipeline = null;
        this._initialized = false;
        
        console.log(`Switched to technique: ${this._technique}`);
    }
    
    /**
     * Get a configuration value.
     * 
     * @param {string} key - Configuration key
     * @param {*} defaultValue - Default value if key not found
     * @returns {*} Configuration value or default
     */
    getConfig(key, defaultValue = null) {
        // Check local config first
        if (key in this._config) {
            return this._config[key];
        }
        
        // Check configuration manager
        return this._configManager.get(key, defaultValue);
    }
    
    /**
     * Return string representation of the ConfigurableRAG instance.
     * 
     * @returns {string} String representation
     */
    toString() {
        const status = this._initialized ? 'initialized' : 'not initialized';
        return `ConfigurableRAG(technique=${this._technique}, status=${status})`;
    }
}

module.exports = { ConfigurableRAG };