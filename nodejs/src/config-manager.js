/**
 * Enhanced Configuration Manager for RAG Templates Library Consumption Framework - JavaScript Implementation.
 * 
 * This module provides a three-tier configuration system:
 * 1. Built-in defaults for zero-config operation
 * 2. Configuration file support
 * 3. Environment variable integration
 * 
 * The configuration manager ensures no hard-coded secrets and provides
 * sensible defaults for immediate usability.
 * 
 * Mirrors the Python ConfigManager functionality exactly.
 */

const fs = require('fs');
const path = require('path');
const yaml = require('js-yaml');
const os = require('os');

/**
 * Enhanced Configuration Manager with three-tier configuration system.
 * 
 * This manager provides:
 * 1. Built-in defaults for zero-configuration operation
 * 2. Configuration file loading and parsing
 * 3. Environment variable override support
 * 4. Configuration validation and error handling
 * 5. Fallback strategies for missing configuration
 */
class ConfigManager {
    static ENV_PREFIX = 'RAG_';
    static DELIMITER = '__';  // Double underscore for nesting in env vars
    
    /**
     * Initialize the Enhanced Configuration Manager.
     * 
     * @param {string} configPath - Optional path to configuration file
     * @param {Object} schema - Optional schema for configuration validation
     */
    constructor(configPath = null, schema = null) {
        this._config = {};
        this._schema = schema;
        this._defaultsLoaded = false;
        
        // Load configuration in order of precedence
        this._loadBuiltinDefaults();
        
        if (configPath) {
            this._loadConfigFile(configPath);
        } else {
            this._loadDefaultConfigFile();
        }
        
        this._loadEnvVariables();
        
        console.log('Configuration manager initialized successfully');
    }
    
    /**
     * Load built-in default configuration for zero-config operation.
     * 
     * These defaults ensure the system can operate without any configuration
     * while avoiding hard-coded secrets.
     */
    _loadBuiltinDefaults() {
        this._config = {
            database: {
                iris: {
                    host: 'localhost',
                    port: 1972,
                    namespace: 'USER',
                    username: null,  // No default username - must be provided
                    password: null,  // No default password - must be provided
                    timeout: 30,
                    pool_size: 5
                }
            },
            embeddings: {
                model: 'all-MiniLM-L6-v2',
                dimension: 384,
                provider: 'sentence-transformers',
                batch_size: 32,
                normalize: true
            },
            llm: {
                provider: null,  // No default LLM provider
                model: null,     // No default model
                api_key: null,   // No default API key
                temperature: 0.7,
                max_tokens: 1000
            },
            pipelines: {
                basic: {
                    chunk_size: 1000,
                    chunk_overlap: 200,
                    default_top_k: 5,
                    embedding_batch_size: 32
                }
            },
            vector_index: {
                type: 'HNSW',
                M: 16,
                efConstruction: 200,
                Distance: 'COSINE'
            },
            logging: {
                level: 'INFO',
                format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        };
        
        this._defaultsLoaded = true;
        console.log('Built-in defaults loaded successfully');
    }
    
    /**
     * Load configuration from a YAML file.
     * 
     * @param {string} configPath - Path to the configuration file
     * @throws {Error} If file cannot be loaded or parsed
     */
    _loadConfigFile(configPath) {
        try {
            if (!fs.existsSync(configPath)) {
                throw new Error(`Configuration file not found: ${configPath}`);
            }
            
            const fileContent = fs.readFileSync(configPath, 'utf8');
            const fileConfig = yaml.load(fileContent) || {};
            
            // Deep merge with existing configuration
            this._deepMerge(this._config, fileConfig);
            console.log(`Configuration loaded from: ${configPath}`);
            
        } catch (error) {
            if (error.name === 'YAMLException') {
                throw new Error(`Failed to parse YAML configuration file: ${configPath} - ${error.message}`);
            }
            throw new Error(`Failed to load configuration file: ${configPath} - ${error.message}`);
        }
    }
    
    /**
     * Attempt to load default configuration files.
     * 
     * Looks for configuration files in standard locations without
     * raising errors if they don't exist.
     */
    _loadDefaultConfigFile() {
        const defaultPaths = [
            'config.yaml',
            'config/config.yaml',
            path.join(os.homedir(), '.rag_templates', 'config.yaml')
        ];
        
        for (const configPath of defaultPaths) {
            if (fs.existsSync(configPath)) {
                try {
                    this._loadConfigFile(configPath);
                    console.log(`Loaded default configuration from: ${configPath}`);
                    break;
                } catch (error) {
                    console.log(`Failed to load default config from: ${configPath}`);
                    continue;
                }
            }
        }
    }
    
    /**
     * Load configuration from environment variables.
     * 
     * Environment variables override both defaults and file configuration.
     * Variables should be prefixed with RAG_ and use double underscores
     * for nesting (e.g., RAG_DATABASE__IRIS__HOST).
     */
    _loadEnvVariables() {
        const envOverrides = {};
        
        for (const [envVar, value] of Object.entries(process.env)) {
            if (envVar.startsWith(ConfigManager.ENV_PREFIX)) {
                // Remove prefix and split by delimiter
                const keyPathStr = envVar.slice(ConfigManager.ENV_PREFIX.length);
                const keys = keyPathStr.split(ConfigManager.DELIMITER).map(k => k.toLowerCase());
                
                // Set the value in the nested structure
                this._setNestedValue(envOverrides, keys, value);
            }
        }
        
        if (Object.keys(envOverrides).length > 0) {
            // Deep merge environment overrides
            this._deepMerge(this._config, envOverrides);
            console.log(`Applied ${Object.keys(envOverrides).length} environment variable overrides`);
        }
    }
    
    /**
     * Set a value in a nested dictionary structure.
     * 
     * @param {Object} configDict - The dictionary to modify
     * @param {Array<string>} keys - List of keys representing the path
     * @param {string} value - The value to set
     */
    _setNestedValue(configDict, keys, value) {
        let currentLevel = configDict;
        
        for (let i = 0; i < keys.length; i++) {
            const keyPart = keys[i];
            
            if (i === keys.length - 1) {  // Last key part
                // Try to cast to appropriate type
                const castedValue = this._castEnvValue(value, keys);
                currentLevel[keyPart] = castedValue;
            } else {
                // Ensure we have an object at this level
                if (!(keyPart in currentLevel)) {
                    currentLevel[keyPart] = {};
                } else if (typeof currentLevel[keyPart] !== 'object' || currentLevel[keyPart] === null) {
                    currentLevel[keyPart] = {};
                }
                currentLevel = currentLevel[keyPart];
            }
        }
    }
    
    /**
     * Cast environment variable string to appropriate type.
     * 
     * @param {string} valueStr - The string value from environment variable
     * @param {Array<string>} keys - The key path for context
     * @returns {*} The value cast to appropriate type
     */
    _castEnvValue(valueStr, keys) {
        // Get the original value type for reference
        const originalValue = this._getNestedValue(this._config, keys);
        const targetType = originalValue !== null ? typeof originalValue : null;
        
        try {
            if (targetType === 'boolean') {
                return ['true', '1', 'yes', 'on'].includes(valueStr.toLowerCase());
            } else if (targetType === 'number') {
                const num = Number(valueStr);
                return isNaN(num) ? valueStr : num;
            } else if (Array.isArray(originalValue)) {
                // Simple comma-separated list support
                return valueStr.split(',').map(item => item.trim());
            } else {
                // Return as string for unknown types or null target type
                return valueStr;
            }
        } catch (error) {
            console.warn(`Failed to cast environment variable value: ${valueStr}`);
            return valueStr;
        }
    }
    
    /**
     * Get a value from a nested dictionary structure.
     * 
     * @param {Object} configDict - The dictionary to search
     * @param {Array<string>} keys - List of keys representing the path
     * @returns {*} The value if found, null otherwise
     */
    _getNestedValue(configDict, keys) {
        let current = configDict;
        for (const key of keys) {
            if (typeof current === 'object' && current !== null && key in current) {
                current = current[key];
            } else {
                return null;
            }
        }
        return current;
    }
    
    /**
     * Deep merge two objects, modifying baseDict in place.
     * 
     * @param {Object} baseDict - The base object to merge into
     * @param {Object} updateDict - The object to merge from
     */
    _deepMerge(baseDict, updateDict) {
        for (const [key, value] of Object.entries(updateDict)) {
            if (key in baseDict && 
                typeof baseDict[key] === 'object' && baseDict[key] !== null &&
                typeof value === 'object' && value !== null &&
                !Array.isArray(baseDict[key]) && !Array.isArray(value)) {
                this._deepMerge(baseDict[key], value);
            } else {
                baseDict[key] = value;
            }
        }
    }
    
    /**
     * Retrieve a configuration setting using dot notation.
     * 
     * @param {string} keyString - The configuration key (e.g., "database:iris:host")
     * @param {*} defaultValue - The default value if key is not found
     * @returns {*} The configuration value or default
     */
    get(keyString, defaultValue = null) {
        const keys = keyString.split(':').map(k => k.toLowerCase());
        const value = this._getNestedValue(this._config, keys);
        
        if (value === null) {
            console.log(`Configuration key not found: ${keyString}, using default: ${defaultValue}`);
            return defaultValue;
        }
        
        return value;
    }
    
    /**
     * Set a configuration value using dot notation.
     * 
     * @param {string} keyString - The configuration key (e.g., "database:iris:host")
     * @param {*} value - The value to set
     */
    set(keyString, value) {
        const keys = keyString.split(':').map(k => k.toLowerCase());
        this._setNestedValue(this._config, keys, String(value));
        console.log(`Configuration set: ${keyString} = ${value}`);
    }
    
    /**
     * Validate the current configuration.
     * 
     * @throws {Error} If validation fails
     */
    validate() {
        if (!this._defaultsLoaded) {
            throw new Error('Configuration defaults not loaded');
        }
        
        // Basic validation - ensure critical paths exist
        const criticalPaths = [
            'database:iris:host',
            'database:iris:port',
            'embeddings:model'
        ];
        
        for (const path of criticalPaths) {
            const value = this.get(path);
            if (value === null) {
                throw new Error(`Missing required configuration: ${path}`);
            }
        }
        
        // Validate data types
        this._validateTypes();
        
        console.log('Configuration validation passed');
    }
    
    /**
     * Validate configuration value types.
     */
    _validateTypes() {
        const typeValidations = [
            ['database:iris:port', 'number'],
            ['database:iris:timeout', 'number'],
            ['embeddings:dimension', 'number'],
            ['embeddings:batch_size', 'number'],
            ['pipelines:basic:chunk_size', 'number'],
            ['pipelines:basic:chunk_overlap', 'number']
        ];
        
        for (const [key, expectedType] of typeValidations) {
            const value = this.get(key);
            if (value !== null && typeof value !== expectedType) {
                throw new Error(
                    `Invalid type for ${key}: expected ${expectedType}, got ${typeof value}`
                );
            }
        }
    }
    
    /**
     * Get database configuration with validation.
     * 
     * @returns {Object} Database configuration object
     */
    getDatabaseConfig() {
        const config = {
            host: this.get('database:iris:host'),
            port: this.get('database:iris:port'),
            namespace: this.get('database:iris:namespace'),
            username: this.get('database:iris:username'),
            password: this.get('database:iris:password'),
            timeout: this.get('database:iris:timeout'),
            pool_size: this.get('database:iris:pool_size')
        };
        
        // Validate required fields
        if (!config.host) {
            throw new Error('Database host is required');
        }
        if (!config.port) {
            throw new Error('Database port is required');
        }
        
        return config;
    }
    
    /**
     * Get embedding configuration.
     * 
     * @returns {Object} Embedding configuration object
     */
    getEmbeddingConfig() {
        return {
            model: this.get('embeddings:model'),
            dimension: this.get('embeddings:dimension'),
            provider: this.get('embeddings:provider'),
            batch_size: this.get('embeddings:batch_size'),
            normalize: this.get('embeddings:normalize')
        };
    }
    
    /**
     * Get pipeline-specific configuration.
     * 
     * @param {string} pipelineName - Name of the pipeline (default: "basic")
     * @returns {Object} Pipeline configuration object
     */
    getPipelineConfig(pipelineName = 'basic') {
        const baseKey = `pipelines:${pipelineName}`;
        return {
            chunk_size: this.get(`${baseKey}:chunk_size`),
            chunk_overlap: this.get(`${baseKey}:chunk_overlap`),
            default_top_k: this.get(`${baseKey}:default_top_k`),
            embedding_batch_size: this.get(`${baseKey}:embedding_batch_size`)
        };
    }
    
    /**
     * Return the complete configuration as an object.
     * 
     * @returns {Object} Complete configuration object
     */
    toDict() {
        return JSON.parse(JSON.stringify(this._config));
    }
}

module.exports = { ConfigManager };