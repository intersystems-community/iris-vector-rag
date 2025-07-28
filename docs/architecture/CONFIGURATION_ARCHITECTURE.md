# Configuration Management Architecture

## 1. Overview

This document defines the configuration management architecture for the IRIS RAG MCP server, providing environment-based configuration with zero hard-coded secrets, hierarchical configuration merging, and runtime configuration validation following support-tools-mcp patterns.

## 2. Configuration Architecture Principles

### 2.1 Design Principles

- **Environment-First**: All configuration sourced from environment variables and external files
- **Zero Hard-Coded Secrets**: No sensitive information embedded in code or configuration files
- **Hierarchical Merging**: Layered configuration with clear precedence rules
- **Runtime Validation**: Schema-based validation with detailed error reporting
- **Hot Reloading**: Dynamic configuration updates without service restart

### 2.2 Security Requirements

- **Secret Management**: Integration with external secret management systems
- **Access Control**: Role-based configuration access and modification
- **Audit Logging**: Complete audit trail of configuration changes
- **Encryption**: Sensitive configuration data encrypted at rest and in transit

## 3. Configuration Architecture Overview

### 3.1 Configuration Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                 CONFIGURATION HIERARCHY                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Default Configuration (Lowest Priority)                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ • Built-in defaults in code                                 │ │
│  │ • Framework-level defaults                                  │ │
│  │ • Technique-specific defaults                               │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                ▲                                │
│                                │                                │
│  2. Base Configuration Files                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ • config/default.yaml                                       │ │
│  │ • config/base.yaml                                          │ │
│  │ • config/techniques/*.yaml                                  │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                ▲                                │
│                                │                                │
│  3. Environment-Specific Files                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ • config/development.yaml                                   │ │
│  │ • config/production.yaml                                    │ │
│  │ • config/testing.yaml                                       │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                ▲                                │
│                                │                                │
│  4. Environment Variables (Highest Priority)                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ • RAG_DATABASE__IRIS__HOST                                  │ │
│  │ • RAG_LLM__OPENAI__API_KEY                                  │ │
│  │ • RAG_PERFORMANCE__MAX_WORKERS                              │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Configuration Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   CONFIGURATION FLOW                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Configuration Sources                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Environment │  │ YAML Files  │  │ Secret Mgmt │             │
│  │ Variables   │  │ & Defaults  │  │ Systems     │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│         │                 │                 │                  │
│         └─────────────────┼─────────────────┘                  │
│                           │                                    │
│                           ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              Configuration Loader                          │ │
│  │                                                             │ │
│  │ • Environment variable parsing                              │ │
│  │ • YAML file loading and merging                             │ │
│  │ • Secret resolution and injection                           │ │
│  │ • Hierarchical configuration assembly                       │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                           │                                    │
│                           ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              Configuration Validator                       │ │
│  │                                                             │ │
│  │ • Schema validation                                         │ │
│  │ • Type checking and coercion                                │ │
│  │ • Business rule validation                                  │ │
│  │ • Dependency validation                                     │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                           │                                    │
│                           ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              Configuration Manager                         │ │
│  │                                                             │ │
│  │ • Runtime configuration access                              │ │
│  │ • Hot reloading support                                     │ │
│  │ • Change notification                                       │ │
│  │ • Configuration caching                                     │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                           │                                    │
│                           ▼                                    │
│              Application Components                            │
└─────────────────────────────────────────────────────────────────┘
```

## 4. Environment Variable Schema

### 4.1 Naming Convention

```
RAG_<COMPONENT>__<SUBCOMPONENT>__<SETTING>

Examples:
- RAG_DATABASE__IRIS__HOST=localhost
- RAG_DATABASE__IRIS__PORT=1972
- RAG_DATABASE__IRIS__USERNAME=_SYSTEM
- RAG_DATABASE__IRIS__PASSWORD_SECRET=iris-db-password

- RAG_LLM__OPENAI__API_KEY_SECRET=openai-api-key
- RAG_LLM__OPENAI__MODEL=gpt-4
- RAG_LLM__OPENAI__TEMPERATURE=0.7

- RAG_EMBEDDING__MODEL=all-MiniLM-L6-v2
- RAG_EMBEDDING__DIMENSION=384
- RAG_EMBEDDING__PROVIDER=sentence-transformers

- RAG_PERFORMANCE__MAX_WORKERS=5
- RAG_PERFORMANCE__TIMEOUT_MS=30000
- RAG_PERFORMANCE__CACHE_SIZE=1000

- RAG_SECURITY__ENABLE_AUTH=true
- RAG_SECURITY__JWT_SECRET_SECRET=jwt-signing-key
- RAG_SECURITY__RATE_LIMIT=100

- RAG_MONITORING__ENABLE_METRICS=true
- RAG_MONITORING__METRICS_PORT=9090
- RAG_MONITORING__LOG_LEVEL=INFO
```

### 4.2 Secret Management Integration

```
# Secret references use _SECRET suffix
RAG_DATABASE__IRIS__PASSWORD_SECRET=iris-db-password
RAG_LLM__OPENAI__API_KEY_SECRET=openai-api-key
RAG_SECURITY__JWT_SECRET_SECRET=jwt-signing-key

# Supported secret backends
RAG_SECRETS__BACKEND=vault  # vault, aws-secrets, azure-keyvault, file
RAG_SECRETS__VAULT__URL=https://vault.company.com
RAG_SECRETS__VAULT__TOKEN_SECRET=vault-token
RAG_SECRETS__VAULT__PATH=secret/iris-rag
```

## 5. Configuration Schema Definition

### 5.1 Master Configuration Schema

```yaml
# config/schema.yaml
type: object
properties:
  server:
    type: object
    properties:
      name:
        type: string
        default: "iris-rag-mcp-server"
      version:
        type: string
        default: "1.0.0"
      port:
        type: integer
        minimum: 1024
        maximum: 65535
        default: 3000
      host:
        type: string
        default: "localhost"
    required: [name, version]

  database:
    type: object
    properties:
      iris:
        type: object
        properties:
          host:
            type: string
            default: "localhost"
          port:
            type: integer
            minimum: 1
            maximum: 65535
            default: 1972
          namespace:
            type: string
            default: "USER"
          username:
            type: string
            default: "_SYSTEM"
          password:
            type: string
            secret: true
          connection_pool:
            type: object
            properties:
              min_connections:
                type: integer
                minimum: 1
                default: 2
              max_connections:
                type: integer
                minimum: 1
                default: 10
              idle_timeout_ms:
                type: integer
                minimum: 1000
                default: 30000
        required: [host, port, username, password]

  llm:
    type: object
    properties:
      provider:
        type: string
        enum: [openai, anthropic, azure, local]
        default: "openai"
      openai:
        type: object
        properties:
          api_key:
            type: string
            secret: true
          model:
            type: string
            default: "gpt-4"
          temperature:
            type: number
            minimum: 0.0
            maximum: 2.0
            default: 0.7
          max_tokens:
            type: integer
            minimum: 1
            maximum: 8192
            default: 1024
        required: [api_key]

  embedding:
    type: object
    properties:
      provider:
        type: string
        enum: [sentence-transformers, openai, azure]
        default: "sentence-transformers"
      model:
        type: string
        default: "all-MiniLM-L6-v2"
      dimension:
        type: integer
        minimum: 1
        maximum: 4096
        default: 384
      batch_size:
        type: integer
        minimum: 1
        maximum: 1000
        default: 32

  techniques:
    type: object
    properties:
      enabled:
        type: array
        items:
          type: string
          enum: [basic, crag, hyde, graphrag, hybrid_ifind, colbert, noderag, sqlrag]
        default: [basic, crag, hyde, graphrag, hybrid_ifind, colbert, noderag, sqlrag]
      
      basic:
        $ref: "#/definitions/basic_technique"
      crag:
        $ref: "#/definitions/crag_technique"
      hyde:
        $ref: "#/definitions/hyde_technique"
      # ... other techniques

  performance:
    type: object
    properties:
      max_workers:
        type: integer
        minimum: 1
        maximum: 50
        default: 5
      worker_timeout_ms:
        type: integer
        minimum: 1000
        default: 30000
      request_timeout_ms:
        type: integer
        minimum: 1000
        default: 60000
      cache_size:
        type: integer
        minimum: 100
        default: 1000
      enable_caching:
        type: boolean
        default: true

  monitoring:
    type: object
    properties:
      enable_metrics:
        type: boolean
        default: true
      metrics_port:
        type: integer
        minimum: 1024
        maximum: 65535
        default: 9090
      log_level:
        type: string
        enum: [DEBUG, INFO, WARN, ERROR]
        default: "INFO"
      health_check_interval_ms:
        type: integer
        minimum: 1000
        default: 30000

  security:
    type: object
    properties:
      enable_auth:
        type: boolean
        default: false
      jwt_secret:
        type: string
        secret: true
      rate_limit:
        type: integer
        minimum: 1
        default: 100
      cors_origins:
        type: array
        items:
          type: string
        default: ["*"]

definitions:
  basic_technique:
    type: object
    properties:
      enabled:
        type: boolean
        default: true
      top_k:
        type: integer
        minimum: 1
        maximum: 50
        default: 5
      min_similarity:
        type: number
        minimum: 0.0
        maximum: 1.0
        default: 0.1

  crag_technique:
    type: object
    properties:
      enabled:
        type: boolean
        default: true
      confidence_threshold:
        type: number
        minimum: 0.0
        maximum: 1.0
        default: 0.8
      enable_web_search:
        type: boolean
        default: false
      correction_strategy:
        type: string
        enum: [rewrite, expand, filter]
        default: "rewrite"
```

## 6. Configuration Manager Implementation

### 6.1 Enhanced Configuration Manager

```typescript
// nodejs/src/config/manager.ts
import * as yaml from 'js-yaml';
import * as fs from 'fs';
import * as path from 'path';
import Ajv from 'ajv';
import addFormats from 'ajv-formats';

interface ConfigurationOptions {
  environment?: string;
  configDir?: string;
  schemaPath?: string;
  secretsBackend?: SecretsBackend;
  enableHotReload?: boolean;
}

class ConfigurationManager {
  private config: any = {};
  private schema: any;
  private ajv: Ajv;
  private secretsBackend: SecretsBackend;
  private watchers: Map<string, fs.FSWatcher> = new Map();
  private changeListeners: Set<ConfigChangeListener> = new Set();
  
  constructor(options: ConfigurationOptions = {}) {
    this.ajv = new Ajv({ allErrors: true, useDefaults: true });
    addFormats(this.ajv);
    
    this.secretsBackend = options.secretsBackend || new FileSecretsBackend();
    
    // Load configuration schema
    this.loadSchema(options.schemaPath);
    
    // Load configuration
    this.loadConfiguration(options);
    
    // Setup hot reloading if enabled
    if (options.enableHotReload) {
      this.setupHotReload(options.configDir);
    }
  }
  
  private loadSchema(schemaPath?: string): void {
    const defaultSchemaPath = path.join(__dirname, '../config/schema.yaml');
    const finalSchemaPath = schemaPath || defaultSchemaPath;
    
    if (fs.existsSync(finalSchemaPath)) {
      const schemaContent = fs.readFileSync(finalSchemaPath, 'utf8');
      this.schema = yaml.load(schemaContent);
      this.ajv.addSchema(this.schema, 'config');
    }
  }
  
  private async loadConfiguration(options: ConfigurationOptions): Promise<void> {
    const environment = options.environment || process.env.NODE_ENV || 'development';
    const configDir = options.configDir || path.join(__dirname, '../config');
    
    // 1. Load default configuration
    const defaultConfig = this.loadConfigFile(path.join(configDir, 'default.yaml'));
    
    // 2. Load base configuration
    const baseConfig = this.loadConfigFile(path.join(configDir, 'base.yaml'));
    
    // 3. Load environment-specific configuration
    const envConfig = this.loadConfigFile(path.join(configDir, `${environment}.yaml`));
    
    // 4. Load technique-specific configurations
    const techniqueConfigs = this.loadTechniqueConfigs(configDir);
    
    // 5. Merge configurations (order matters - later configs override earlier ones)
    this.config = this.deepMerge(
      defaultConfig,
      baseConfig,
      techniqueConfigs,
      envConfig
    );
    
    // 6. Apply environment variable overrides
    await this.applyEnvironmentOverrides();
    
    // 7. Resolve secrets
    await this.resolveSecrets();
    
    // 8. Validate final configuration
    this.validateConfiguration();
  }
  
  private loadConfigFile(filePath: string): any {
    if (!fs.existsSync(filePath)) {
      return {};
    }
    
    try {
      const content = fs.readFileSync(filePath, 'utf8');
      return yaml.load(content) || {};
    } catch (error) {
      console.warn(`Failed to load config file ${filePath}:`, error);
      return {};
    }
  }
  
  private loadTechniqueConfigs(configDir: string): any {
    const techniqueConfigDir = path.join(configDir, 'techniques');
    const techniqueConfigs: any = {};
    
    if (!fs.existsSync(techniqueConfigDir)) {
      return techniqueConfigs;
    }
    
    const files = fs.readdirSync(techniqueConfigDir);
    for (const file of files) {
      if (file.endsWith('.yaml') || file.endsWith('.yml')) {
        const techniqueName = path.basename(file, path.extname(file));
        const configPath = path.join(techniqueConfigDir, file);
        techniqueConfigs[techniqueName] = this.loadConfigFile(configPath);
      }
    }
    
    return { techniques: techniqueConfigs };
  }
  
  private async applyEnvironmentOverrides(): Promise<void> {
    const envPrefix = 'RAG_';
    const delimiter = '__';
    
    for (const [envVar, value] of Object.entries(process.env)) {
      if (!envVar.startsWith(envPrefix)) {
        continue;
      }
      
      // Parse environment variable path
      const keyPath = envVar
        .substring(envPrefix.length)
        .split(delimiter)
        .map(part => part.toLowerCase());
      
      // Set value in configuration
      this.setNestedValue(this.config, keyPath, this.parseEnvValue(value));
    }
  }
  
  private async resolveSecrets(): Promise<void> {
    await this.resolveSecretsRecursive(this.config);
  }
  
  private async resolveSecretsRecursive(obj: any): Promise<void> {
    for (const [key, value] of Object.entries(obj)) {
      if (typeof value === 'string' && key.endsWith('_secret')) {
        // Resolve secret reference
        const secretValue = await this.secretsBackend.getSecret(value as string);
        const actualKey = key.replace('_secret', '');
        obj[actualKey] = secretValue;
        delete obj[key];
      } else if (typeof value === 'object' && value !== null) {
        await this.resolveSecretsRecursive(value);
      }
    }
  }
  
  private validateConfiguration(): void {
    if (!this.schema) {
      console.warn('No schema available for configuration validation');
      return;
    }
    
    const validate = this.ajv.getSchema('config');
    if (!validate) {
      throw new Error('Failed to get configuration validator');
    }
    
    const isValid = validate(this.config);
    if (!isValid) {
      const errors = validate.errors?.map(err => 
        `${err.instancePath}: ${err.message}`
      ).join(', ');
      throw new Error(`Configuration validation failed: ${errors}`);
    }
  }
  
  // Public API methods
  get(keyPath: string, defaultValue?: any): any {
    const keys = keyPath.split('.');
    let current = this.config;
    
    for (const key of keys) {
      if (current && typeof current === 'object' && key in current) {
        current = current[key];
      } else {
        return defaultValue;
      }
    }
    
    return current;
  }
  
  set(keyPath: string, value: any): void {
    const keys = keyPath.split('.');
    this.setNestedValue(this.config, keys, value);
    
    // Validate after setting
    this.validateConfiguration();
    
    // Notify listeners
    this.notifyConfigChange(keyPath, value);
  }
  
  async reload(): Promise<void> {
    const oldConfig = { ...this.config };
    await this.loadConfiguration({});
    
    // Notify listeners of changes
    this.notifyConfigReload(oldConfig, this.config);
  }
  
  onConfigChange(listener: ConfigChangeListener): void {
    this.changeListeners.add(listener);
  }
  
  offConfigChange(listener: ConfigChangeListener): void {
    this.changeListeners.delete(listener);
  }
  
  // Utility methods
  private deepMerge(...objects: any[]): any {
    const result: any = {};
    
    for (const obj of objects) {
      if (!obj || typeof obj !== 'object') continue;
      
      for (const [key, value] of Object.entries(obj)) {
        if (value && typeof value === 'object' && !Array.isArray(value)) {
          result[key] = this.deepMerge(result[key] || {}, value);
        } else {
          result[key] = value;
        }
      }
    }
    
    return result;
  }
  
  private setNestedValue(obj: any, keys: string[], value: any): void {
    let current = obj;
    
    for (let i = 0; i < keys.length - 1; i++) {
      const key = keys[i];
      if (!(key in current) || typeof current[key] !== 'object') {
        current[key] = {};
      }
      current = current[key];
    }
    
    current[keys[keys.length - 1]] = value;
  }
  
  private parseEnvValue(value: string | undefined): any {
    if (!value) return undefined;
    
    // Try to parse as JSON first
    try {
      return JSON.parse(value);
    } catch {
      // Return as string if JSON parsing fails
      return value;
    }
  }
}
```

## 7. Secrets Management

### 7.1 Secrets Backend Interface

```typescript
interface SecretsBackend {
  getSecret(secretName: string): Promise<string>;
  setSecret(secretName: string, value: string): Promise<void>;
  deleteSecret(secretName: string): Promise<void>;
  listSecrets(): Promise<string[]>;
}

class VaultSecretsBackend implements SecretsBackend {
  private vaultUrl: string;
  private vaultToken: string;
  private vaultPath: string;
  
  constructor(config: VaultConfig) {
    this.vaultUrl = config.url;
    this.vaultToken = config.token;
    this.vaultPath = config.path;
  }
  
  async getSecret(secretName: string): Promise<string> {
    const response = await fetch(
      `${this.vaultUrl}/v1/${this.vaultPath}/${secretName}`,
      {
        headers: {
          'X-Vault-Token': this.vaultToken
        }
      }
    );
    
    if (!response.ok) {
      throw new Error(`Failed to retrieve secret ${secretName}: ${response.statusText}`);
    }
    
    const data = await response.json();
    return data.data.value;
  }
}

class FileSecretsBackend implements SecretsBackend {
  private secretsDir: string;
  
  constructor(secretsDir: string = '/etc/secrets') {
    this.secretsDir = secretsDir;
  }
  
  async getSecret(secretName: string): Promise<string> {
    const secretPath = path.join(this.secretsDir, secretName);
    
    if (!fs.existsSync(secretPath)) {
      throw new Error(`Secret file not found: ${secretPath}`);
    }
    
    return fs.readFileSync(secretPath, 'utf8').trim();
  }
}
```

This configuration management architecture provides a robust, secure, and flexible foundation for managing all aspects of the IRIS RAG MCP server configuration with environment-based settings, secret management, and runtime validation.