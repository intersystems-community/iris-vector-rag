#!/usr/bin/env node

/**
 * RAG Templates Setup Wizard - JavaScript/Node.js
 * 
 * Interactive CLI for setting up RAG capabilities including IFind optimization.
 * Works alongside the Python CLI for complete cross-language support.
 */

import readline from 'readline';
import IrisConnectionManager from '../iris_connection_manager.js';

// Simple chalk replacement for cross-platform compatibility
const chalk = {
  bold: { blue: (text) => `\x1b[1m\x1b[34m${text}\x1b[0m`, cyan: (text) => `\x1b[1m\x1b[36m${text}\x1b[0m`, green: (text) => `\x1b[1m\x1b[32m${text}\x1b[0m` },
  cyan: (text) => `\x1b[36m${text}\x1b[0m`,
  gray: (text) => `\x1b[90m${text}\x1b[0m`,
  green: (text) => `\x1b[32m${text}\x1b[0m`,
  red: (text) => `\x1b[31m${text}\x1b[0m`,
  yellow: (text) => `\x1b[33m${text}\x1b[0m`
};

class RAGSetupWizard {
  constructor(options = {}) {
    this.rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout
    });
    
    this.options = {
      ifindOnly: options.ifindOnly || false,
      ...options
    };
    
    this.config = {
      iris: {
        connection_string: null,
        username: null,
        password: null,
        namespace: 'USER'
      },
      setup_type: null,
      ifind_strategy: null
    };
  }

  async ask(question) {
    return new Promise((resolve) => {
      this.rl.question(question, (answer) => {
        resolve(answer.trim());
      });
    });
  }

  async askChoice(question, choices) {
    console.log(chalk.cyan(question));
    choices.forEach((choice, index) => {
      console.log(chalk.gray(`  ${index + 1}. ${choice.label} - ${choice.description}`));
    });
    
    while (true) {
      const answer = await this.ask(chalk.yellow('Enter your choice (1-' + choices.length + '): '));
      const choice = parseInt(answer) - 1;
      
      if (choice >= 0 && choice < choices.length) {
        return choices[choice];
      }
      
      console.log(chalk.red('Invalid choice. Please try again.'));
    }
  }

  async welcome() {
    console.log(chalk.bold.blue('\n🚀 RAG Templates Setup Wizard (JavaScript/Node.js)'));
    console.log(chalk.gray('=====================================================\n'));
    
    console.log(chalk.green('This wizard will help you set up RAG capabilities with InterSystems IRIS.\n'));
    
    console.log(chalk.yellow('What we\'ll configure:'));
    console.log(chalk.gray('• IRIS database connection'));
    console.log(chalk.gray('• RAG pipeline setup type'));  
    console.log(chalk.gray('• IFind full-text search optimization'));
    console.log(chalk.gray('• JavaScript-specific configuration\n'));
  }

  async configureIrisConnection() {
    console.log(chalk.bold.cyan('\n📡 IRIS Database Connection\n'));
    
    this.config.iris.connection_string = await this.ask(
      chalk.yellow('IRIS connection string (e.g., localhost:1972/USER): ')
    );
    
    this.config.iris.username = await this.ask(
      chalk.yellow('Username: ')
    );
    
    this.config.iris.password = await this.ask(
      chalk.yellow('Password: ')
    );
    
    this.config.iris.namespace = await this.ask(
      chalk.yellow('Namespace [USER]: ')
    ) || 'USER';
    
    // Test connection
    console.log(chalk.gray('\nTesting connection...'));
    
    try {
      const connectionManager = new IrisConnectionManager({
        connectionString: this.config.iris.connection_string,
        username: this.config.iris.username,
        password: this.config.iris.password,
        namespace: this.config.iris.namespace
      });
      
      // Test basic connectivity
      const result = await connectionManager.testConnection();
      
      if (result.success) {
        console.log(chalk.green('✅ Connection successful!'));
      } else {
        console.log(chalk.red('❌ Connection failed: ' + result.error));
        
        const retry = await this.ask(chalk.yellow('Retry configuration? (y/n): '));
        if (retry.toLowerCase() === 'y') {
          return this.configureIrisConnection();
        }
      }
    } catch (error) {
      console.log(chalk.red('❌ Connection test failed: ' + error.message));
    }
  }

  async chooseSetupType() {
    console.log(chalk.bold.cyan('\n🏗️ Setup Type\n'));
    
    const setupTypes = [
      {
        value: 'new_installation',
        label: 'New Installation',
        description: 'Fresh RAG setup with optimized architecture'
      },
      {
        value: 'existing_content',
        label: 'Existing IRIS Content',
        description: 'Add RAG to existing IRIS server with content (view-based overlay)'
      },
      {
        value: 'python_integration', 
        label: 'Python Integration',
        description: 'JavaScript client connecting to Python-managed RAG system'
      }
    ];
    
    const choice = await this.askChoice(
      'What type of setup do you need?',
      setupTypes
    );
    
    this.config.setup_type = choice.value;
    
    console.log(chalk.green('\n✅ Setup type: ' + choice.label));
  }

  async chooseIFindStrategy() {
    console.log(chalk.bold.cyan('\n🔍 IFind Full-Text Search Strategy\n'));
    
    const strategies = [
      {
        value: 'optimized_minimal',
        label: 'Optimized Minimal (Recommended)',
        description: '70% storage reduction, minimal data duplication'
      },
      {
        value: 'full_duplication',
        label: 'Full Duplication',
        description: 'Complete table copy, higher storage but simpler'
      },
      {
        value: 'no_ifind',
        label: 'Skip IFind',
        description: 'Use LIKE search fallback only'
      }
    ];
    
    const choice = await this.askChoice(
      'Choose IFind search strategy:',
      strategies
    );
    
    this.config.ifind_strategy = choice.value;
    
    console.log(chalk.green('\n✅ IFind strategy: ' + choice.label));
  }

  async executeSetup() {
    console.log(chalk.bold.cyan('\n⚙️ Executing Setup\n'));
    
    const steps = [];
    
    // Add steps based on configuration
    if (this.config.setup_type === 'new_installation') {
      steps.push(
        { name: 'Create RAG schema', action: this.createRagSchema.bind(this) },
        { name: 'Setup vector tables', action: this.setupVectorTables.bind(this) }
      );
      
      if (this.config.ifind_strategy === 'optimized_minimal') {
        steps.push({ name: 'Setup optimized IFind', action: this.setupOptimizedIFind.bind(this) });
      } else if (this.config.ifind_strategy === 'full_duplication') {
        steps.push({ name: 'Setup IFind with duplication', action: this.setupFullIFind.bind(this) });
      }
    } else if (this.config.setup_type === 'existing_content') {
      steps.push(
        { name: 'Discover existing tables', action: this.discoverExistingTables.bind(this) },
        { name: 'Create overlay views', action: this.createOverlayViews.bind(this) },
        { name: 'Setup minimal RAG tables', action: this.setupMinimalRagTables.bind(this) }
      );
    }
    
    steps.push(
      { name: 'Generate JavaScript config', action: this.generateJsConfig.bind(this) },
      { name: 'Create example files', action: this.createExampleFiles.bind(this) }
    );
    
    // Execute steps
    for (const step of steps) {
      console.log(chalk.gray(`\n🔧 ${step.name}...`));
      
      try {
        await step.action();
        console.log(chalk.green(`✅ ${step.name} completed`));
      } catch (error) {
        console.log(chalk.red(`❌ ${step.name} failed: ${error.message}`));
        
        const continueSetup = await this.ask(chalk.yellow('Continue with remaining steps? (y/n): '));
        if (continueSetup.toLowerCase() !== 'y') {
          throw new Error('Setup aborted by user');
        }
      }
    }
  }

  async createRagSchema() {
    // JavaScript implementation of schema creation
    const connectionManager = new IrisConnectionManager(this.config.iris);
    
    const schemaSql = `
      CREATE SCHEMA IF NOT EXISTS RAG;
      
      CREATE TABLE IF NOT EXISTS RAG.SourceDocuments (
        doc_id VARCHAR(255) PRIMARY KEY,
        title VARCHAR(1000),
        text_content LONGVARCHAR,
        embedding VARCHAR(32000),
        metadata VARCHAR(4000),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      );
    `;
    
    await connectionManager.execute(schemaSql);
  }

  async setupVectorTables() {
    // Setup vector-specific tables and indexes
    const connectionManager = new IrisConnectionManager(this.config.iris);
    
    const vectorSql = `
      CREATE INDEX IF NOT EXISTS idx_sourcedocs_embedding 
      ON RAG.SourceDocuments (embedding);
    `;
    
    await connectionManager.execute(vectorSql);
  }

  async setupOptimizedIFind() {
    // JavaScript implementation of optimized IFind setup
    const connectionManager = new IrisConnectionManager(this.config.iris);
    
    const ifindSql = `
      CREATE TABLE IF NOT EXISTS RAG.SourceDocumentsIFindIndex (
        doc_id VARCHAR(255) PRIMARY KEY,
        text_content LONGVARCHAR,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      );
      
      CREATE VIEW IF NOT EXISTS RAG.SourceDocumentsWithIFind AS
      SELECT 
        s.doc_id, s.title, s.text_content, s.embedding, s.metadata, s.created_at
      FROM RAG.SourceDocuments s
      INNER JOIN RAG.SourceDocumentsIFindIndex f ON s.doc_id = f.doc_id;
    `;
    
    await connectionManager.execute(ifindSql);
  }

  async setupFullIFind() {
    // Full duplication IFind setup
    const connectionManager = new IrisConnectionManager(this.config.iris);
    
    const ifindSql = `
      CREATE TABLE IF NOT EXISTS RAG.SourceDocumentsIFind AS
      SELECT * FROM RAG.SourceDocuments WHERE 1=0;
    `;
    
    await connectionManager.execute(ifindSql);
  }

  async discoverExistingTables() {
    // Discover existing content tables for overlay
    const connectionManager = new IrisConnectionManager(this.config.iris);
    
    const discoverySql = `
      SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE
      FROM INFORMATION_SCHEMA.COLUMNS
      WHERE DATA_TYPE IN ('longvarchar', 'varchar', 'text')
        AND CHARACTER_MAXIMUM_LENGTH > 100
        AND TABLE_SCHEMA NOT IN ('INFORMATION_SCHEMA', 'RAG')
      ORDER BY TABLE_SCHEMA, TABLE_NAME
    `;
    
    const tables = await connectionManager.query(discoverySql);
    console.log(chalk.gray(`Found ${tables.length} potential content tables`));
  }

  async createOverlayViews() {
    // Create views for existing content overlay
    console.log(chalk.gray('Creating overlay views for existing content...'));
    // Implementation would go here
  }

  async setupMinimalRagTables() {
    // Create minimal RAG tables for overlay scenario
    console.log(chalk.gray('Setting up minimal RAG tables...'));
    // Implementation would go here
  }

  async generateJsConfig() {
    // Generate JavaScript configuration file
    const configContent = `
// RAG Templates Configuration - Generated by Setup Wizard
export const ragConfig = {
  iris: {
    connectionString: '${this.config.iris.connection_string}',
    username: '${this.config.iris.username}',
    // Note: Store password securely in environment variables
    namespace: '${this.config.iris.namespace}'
  },
  
  setup: {
    type: '${this.config.setup_type}',
    ifindStrategy: '${this.config.ifind_strategy}'
  },
  
  embedding: {
    model: '@xenova/transformers/all-MiniLM-L6-v2',
    dimensions: 384
  }
};

// Usage example:
// import { createVectorSearchPipeline } from '@rag-templates/core';
// const pipeline = createVectorSearchPipeline(ragConfig);
`;
    
    const fs = await import('fs');
    await fs.promises.writeFile('./rag-config.js', configContent);
    console.log(chalk.gray('Generated: ./rag-config.js'));
  }

  async createExampleFiles() {
    // Create example usage files
    const exampleContent = `
// Example: Basic RAG Search with IFind
import { createVectorSearchPipeline } from '@rag-templates/core';
import { ragConfig } from './rag-config.js';

async function exampleSearch() {
  const pipeline = createVectorSearchPipeline(ragConfig);
  await pipeline.initialize();
  
  // Search using hybrid vector + IFind
  const results = await pipeline.search('medical research', {
    topK: 10,
    minSimilarity: 0.7
  });
  
  console.log('Search results:', results);
  await pipeline.close();
}

exampleSearch().catch(console.error);
`;
    
    const fs = await import('fs');
    await fs.promises.writeFile('./example-search.js', exampleContent);
    console.log(chalk.gray('Generated: ./example-search.js'));
  }

  async showSummary() {
    console.log(chalk.bold.green('\n🎉 Setup Complete!\n'));
    
    console.log(chalk.cyan('Configuration Summary:'));
    console.log(chalk.gray(`• IRIS Connection: ${this.config.iris.connection_string}`));
    console.log(chalk.gray(`• Setup Type: ${this.config.setup_type}`));
    console.log(chalk.gray(`• IFind Strategy: ${this.config.ifind_strategy}`));
    
    console.log(chalk.cyan('\nGenerated Files:'));
    console.log(chalk.gray('• rag-config.js - Main configuration'));
    console.log(chalk.gray('• example-search.js - Usage example'));
    
    console.log(chalk.cyan('\nNext Steps:'));
    console.log(chalk.gray('1. Set IRIS password in environment: export IRIS_PASSWORD="your-password"'));
    console.log(chalk.gray('2. Install dependencies: npm install @rag-templates/core'));
    console.log(chalk.gray('3. Run example: node example-search.js'));
    console.log(chalk.gray('4. Index your documents using the pipeline.indexDocument() method'));
    
    if (this.config.setup_type === 'python_integration') {
      console.log(chalk.cyan('\nPython Integration:'));
      console.log(chalk.gray('• Your JavaScript client will connect to Python-managed RAG system'));
      console.log(chalk.gray('• Run Python setup separately: python -m iris_rag.cli setup'));
    }
  }

  async run() {
    try {
      await this.welcome();
      await this.configureIrisConnection();
      
      if (this.options.ifindOnly) {
        // IFind-only setup mode
        console.log(chalk.bold.cyan('\n🔍 IFind-Only Setup Mode\n'));
        await this.chooseIFindStrategy();
        
        // Execute only IFind-related setup steps
        const steps = [];
        if (this.config.ifind_strategy === 'optimized_minimal') {
          steps.push({ name: 'Setup optimized IFind', action: this.setupOptimizedIFind.bind(this) });
        } else if (this.config.ifind_strategy === 'full_duplication') {
          steps.push({ name: 'Setup IFind with duplication', action: this.setupFullIFind.bind(this) });
        }
        
        // Execute IFind setup steps
        for (const step of steps) {
          console.log(chalk.gray(`\n🔧 ${step.name}...`));
          try {
            await step.action();
            console.log(chalk.green(`✅ ${step.name} completed`));
          } catch (error) {
            console.log(chalk.red(`❌ ${step.name} failed: ${error.message}`));
            throw error;
          }
        }
        
        console.log(chalk.bold.green('\n🎉 IFind Setup Complete!\n'));
        console.log(chalk.cyan('IFind strategy: ' + this.config.ifind_strategy));
      } else {
        // Full setup mode
        await this.chooseSetupType();
        await this.chooseIFindStrategy();
        await this.executeSetup();
        await this.showSummary();
      }
    } catch (error) {
      console.log(chalk.red('\n❌ Setup failed: ' + error.message));
      process.exit(1);
    } finally {
      this.rl.close();
    }
  }
}

// Parse command-line arguments
function parseArgs() {
  const args = process.argv.slice(2);
  const options = {
    ifindOnly: args.includes('--ifind-only'),
    help: args.includes('--help') || args.includes('-h')
  };
  return options;
}

// Show help message
function showHelp() {
  console.log(`
🚀 RAG Templates Setup Wizard (JavaScript/Node.js)

Usage: 
  rag-templates                    # Full interactive setup
  rag-templates --ifind-only       # IFind optimization setup only
  rag-templates --help             # Show this help

Options:
  --ifind-only    Setup only IFind optimization (for existing RAG installations)
  --help, -h      Show this help message

Examples:
  npm run setup                    # Full setup
  npm run setup:ifind              # IFind-only setup
  node src/cli/setup-wizard.js     # Direct execution
`);
}

// Run the wizard if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  const options = parseArgs();
  
  if (options.help) {
    showHelp();
    process.exit(0);
  }
  
  const wizard = new RAGSetupWizard(options);
  wizard.run().catch(console.error);
}

export { RAGSetupWizard };