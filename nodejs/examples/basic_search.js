// rag-templates/nodejs/examples/basic_search.js
// Basic example demonstrating vector search functionality

const { createVectorSearchPipeline } = require('../src/index');

async function basicSearchExample() {
  console.log('🚀 Starting IRIS RAG Basic Search Example');
  
  // Create the RAG pipeline with configuration
  const pipeline = createVectorSearchPipeline({
    connection: {
      host: process.env.IRIS_HOST || 'localhost',
      port: parseInt(process.env.IRIS_PORT) || 1972,
      namespace: process.env.IRIS_NAMESPACE || 'USER',
      username: process.env.IRIS_USERNAME || 'SuperUser',
      password: process.env.IRIS_PASSWORD || 'SYS'
    },
    embeddingModel: 'Xenova/all-MiniLM-L6-v2'
  });

  try {
    // Initialize the embedding model
    console.log('📚 Initializing embedding model...');
    await pipeline.initialize();
    console.log('✅ Embedding model ready');

    // Get current statistics
    console.log('\n📊 Current document statistics:');
    const stats = await pipeline.getStats();
    console.log(`Total documents: ${stats.totalDocuments}`);
    console.log(`Total files: ${stats.totalFiles}`);
    
    if (stats.totalDocuments === 0) {
      console.log('\n⚠️  No documents found in the database.');
      console.log('Please run the document_indexing.js example first to add some documents.');
      return;
    }

    // Example search queries
    const searchQueries = [
      'machine learning algorithms',
      'database optimization techniques',
      'vector similarity search',
      'artificial intelligence applications'
    ];

    console.log('\n🔍 Performing search queries:');
    
    for (const query of searchQueries) {
      console.log(`\n--- Searching for: "${query}" ---`);
      
      try {
        const results = await pipeline.search(query, {
          topK: 3,
          minSimilarity: 0.1
        });

        if (results.length === 0) {
          console.log('No results found');
        } else {
          results.forEach((result, index) => {
            console.log(`${index + 1}. Score: ${result.score.toFixed(4)}`);
            console.log(`   Doc ID: ${result.docId}`);
            console.log(`   Source: ${result.sourceFile} (Page ${result.pageNumber}, Chunk ${result.chunkIndex})`);
            console.log(`   Content: ${result.textContent.substring(0, 100)}...`);
          });
        }
      } catch (error) {
        console.error(`Search failed for "${query}": ${error.message}`);
      }
    }

    // Example: Search with additional filters
    console.log('\n🎯 Searching with source file filter:');
    try {
      const filteredResults = await pipeline.search('data processing', {
        topK: 5,
        additionalWhere: "source_file LIKE '%.pdf'"
      });

      console.log(`Found ${filteredResults.length} results from PDF files`);
      filteredResults.forEach((result, index) => {
        console.log(`${index + 1}. ${result.sourceFile} - Score: ${result.score.toFixed(4)}`);
      });
    } catch (error) {
      console.error(`Filtered search failed: ${error.message}`);
    }

    // Example: Get documents by source
    console.log('\n📄 Getting documents by source file:');
    try {
      const sourceFiles = await pipeline.getDocumentsBySource('example.pdf', 5);
      console.log(`Found ${sourceFiles.length} documents from example.pdf`);
      sourceFiles.forEach((doc, index) => {
        console.log(`${index + 1}. ${doc.docId} - Page ${doc.pageNumber}, Chunk ${doc.chunkIndex}`);
      });
    } catch (error) {
      console.error(`Get documents by source failed: ${error.message}`);
    }

    // Display pipeline information
    console.log('\n🔧 Pipeline Information:');
    const info = pipeline.getInfo();
    console.log('Connection:', {
      host: info.connection.host,
      port: info.connection.port,
      namespace: info.connection.namespace
    });
    console.log('Embedding Model:', info.embedding.modelName);
    console.log('Model Ready:', info.isReady);

  } catch (error) {
    console.error('❌ Example failed:', error.message);
  } finally {
    // Clean up resources
    console.log('\n🧹 Cleaning up...');
    await pipeline.close();
    console.log('✅ Example completed');
  }
}

// Run the example if this file is executed directly
if (require.main === module) {
  basicSearchExample().catch(console.error);
}

module.exports = { basicSearchExample };