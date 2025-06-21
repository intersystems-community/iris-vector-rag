#!/usr/bin/env node

/**
 * Test script to verify table creation with real IRIS connection
 */

const { createVectorSearchPipeline } = require('./src/index');

async function testTableCreation() {
  console.log('Testing table creation with real IRIS connection...');
  
  const pipeline = createVectorSearchPipeline({
    connection: {
      host: 'localhost',
      port: 1972,
      namespace: 'USER',
      username: '_SYSTEM',
      password: 'SYS'
    },
    embeddingModel: 'Xenova/all-MiniLM-L6-v2'
  });

  try {
    console.log('Initializing pipeline with table creation...');
    await pipeline.initialize({
      ensureTableSchema: true,
      tableConfig: {
        tableName: 'SourceDocuments',
        vectorDimension: 384,
        createIndexes: true
      }
    });
    
    console.log('✅ Table creation successful!');
    
    // Test a simple query to verify table exists
    const connection = await pipeline.connectionManager.connect();
    const result = await connection.query(
      "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'SourceDocuments'"
    );
    
    console.log('Table verification result:', result);
    
    if (result && result.length > 0) {
      console.log('✅ Table exists and is accessible!');
    } else {
      console.log('❌ Table not found in database');
    }
    
  } catch (error) {
    console.error('❌ Error during table creation test:', error.message);
    console.error('Full error:', error);
  } finally {
    await pipeline.close();
  }
}

// Run the test
testTableCreation().catch(console.error);