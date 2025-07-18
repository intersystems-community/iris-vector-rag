// rag-templates/nodejs/examples/document_indexing.js
// Example demonstrating document indexing functionality

const { createVectorSearchPipeline } = require('../src/index');

async function documentIndexingExample() {
  console.log('🚀 Starting IRIS RAG Document Indexing Example');
  
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

    // Sample documents to index
    const sampleDocuments = [
      {
        docId: 'ml_intro_001',
        title: 'Introduction to Machine Learning',
        content: 'Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or classifications based on those patterns.',
        sourceFile: 'ml_handbook.pdf',
        pageNumber: 1,
        chunkIndex: 0
      },
      {
        docId: 'ml_intro_002',
        title: 'Types of Machine Learning',
        content: 'There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. Supervised learning uses labeled data to train models, unsupervised learning finds patterns in unlabeled data, and reinforcement learning learns through interaction with an environment.',
        sourceFile: 'ml_handbook.pdf',
        pageNumber: 1,
        chunkIndex: 1
      },
      {
        docId: 'db_opt_001',
        title: 'Database Optimization Fundamentals',
        content: 'Database optimization involves improving the performance of database queries and operations. Key techniques include proper indexing, query optimization, normalization, and efficient schema design. Vector databases require specialized optimization strategies for similarity searches.',
        sourceFile: 'database_guide.pdf',
        pageNumber: 5,
        chunkIndex: 0
      },
      {
        docId: 'vector_search_001',
        title: 'Vector Similarity Search',
        content: 'Vector similarity search is a technique used to find similar items in high-dimensional vector spaces. It uses distance metrics like cosine similarity, Euclidean distance, or dot product to measure similarity between vectors. This is fundamental to modern AI applications like recommendation systems and semantic search.',
        sourceFile: 'vector_db_manual.pdf',
        pageNumber: 12,
        chunkIndex: 0
      },
      {
        docId: 'ai_apps_001',
        title: 'AI Applications in Industry',
        content: 'Artificial intelligence has numerous applications across industries including healthcare, finance, manufacturing, and transportation. Common applications include natural language processing, computer vision, predictive analytics, and automated decision-making systems.',
        sourceFile: 'ai_industry_report.pdf',
        pageNumber: 3,
        chunkIndex: 0
      }
    ];

    // Example 1: Index individual documents
    console.log('\n📝 Indexing individual documents...');
    for (let i = 0; i < 2; i++) {
      const doc = sampleDocuments[i];
      console.log(`Indexing: ${doc.title}`);
      
      try {
        await pipeline.indexDocument(
          doc.docId,
          doc.title,
          doc.content,
          doc.sourceFile,
          doc.pageNumber,
          doc.chunkIndex
        );
        console.log(`✅ Successfully indexed: ${doc.docId}`);
      } catch (error) {
        console.error(`❌ Failed to index ${doc.docId}: ${error.message}`);
      }
    }

    // Example 2: Batch index multiple documents
    console.log('\n📦 Batch indexing remaining documents...');
    const remainingDocs = sampleDocuments.slice(2);
    
    try {
      await pipeline.indexDocuments(remainingDocs, { batchSize: 5 });
      console.log(`✅ Successfully batch indexed ${remainingDocs.length} documents`);
    } catch (error) {
      console.error(`❌ Batch indexing failed: ${error.message}`);
    }

    // Example 3: Process and index a long document with chunking
    console.log('\n📄 Processing and indexing a long document with chunking...');
    const longDocument = `
      Retrieval-Augmented Generation (RAG) is a powerful technique that combines the strengths of large language models with external knowledge retrieval systems. 
      
      The RAG approach works by first retrieving relevant documents or passages from a knowledge base using vector similarity search, then using these retrieved documents as context for generating responses with a language model.
      
      This technique addresses several limitations of standalone language models, including knowledge cutoffs, hallucination, and the inability to access up-to-date information. By grounding the generation process in retrieved factual content, RAG systems can provide more accurate and reliable responses.
      
      The implementation of RAG typically involves several key components: a document encoder that converts text into vector embeddings, a vector database for efficient similarity search, a retrieval mechanism that finds relevant documents, and a generation model that produces responses based on the retrieved context.
      
      Vector databases play a crucial role in RAG systems by enabling fast and accurate similarity search across large document collections. Modern vector databases like IRIS provide specialized indexing and query capabilities optimized for high-dimensional vector operations.
    `;

    try {
      const chunkIds = await pipeline.processAndIndexDocument(
        'rag_guide_001',
        'RAG Implementation Guide',
        longDocument,
        'rag_implementation.pdf',
        1,
        {
          preprocessing: {
            maxLength: 512,
            removeExtraWhitespace: true
          },
          chunking: {
            chunkSize: 300,
            overlap: 50,
            splitOnSentences: true
          },
          batchSize: 3
        }
      );
      
      console.log(`✅ Successfully processed and indexed document into ${chunkIds.length} chunks`);
      console.log('Chunk IDs:', chunkIds);
    } catch (error) {
      console.error(`❌ Document processing failed: ${error.message}`);
    }

    // Example 4: Update an existing document
    console.log('\n🔄 Updating an existing document...');
    try {
      const exists = await pipeline.documentExists('ml_intro_001');
      if (exists) {
        const updatedContent = 'Machine learning is a rapidly evolving field of artificial intelligence that enables systems to automatically learn and improve from experience without being explicitly programmed. Modern ML techniques include deep learning, neural networks, and transformer architectures.';
        
        await pipeline.updateDocument('ml_intro_001', updatedContent);
        console.log('✅ Successfully updated document: ml_intro_001');
      } else {
        console.log('⚠️  Document ml_intro_001 does not exist');
      }
    } catch (error) {
      console.error(`❌ Document update failed: ${error.message}`);
    }

    // Display final statistics
    console.log('\n📊 Final document statistics:');
    const finalStats = await pipeline.getStats();
    console.log(`Total documents: ${finalStats.totalDocuments}`);
    console.log(`Total files: ${finalStats.totalFiles}`);
    console.log(`Oldest document: ${finalStats.oldestDocument}`);
    console.log(`Newest document: ${finalStats.newestDocument}`);

    // Test a quick search to verify indexing worked
    console.log('\n🔍 Testing search functionality...');
    const testResults = await pipeline.search('machine learning algorithms', { topK: 3 });
    console.log(`Found ${testResults.length} relevant documents for test search`);
    
    if (testResults.length > 0) {
      console.log('Top result:', {
        docId: testResults[0].docId,
        score: testResults[0].score.toFixed(4),
        preview: testResults[0].textContent.substring(0, 100) + '...'
      });
    }

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
  documentIndexingExample().catch(console.error);
}

module.exports = { documentIndexingExample };