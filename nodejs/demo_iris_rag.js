/**
 * IRIS RAG Demonstration Script
 * This script demonstrates actual PDF indexing and search functionality with IRIS
 */

const { IRIS } = require('intersystems-iris');
const fs = require('fs');
const path = require('path');
const pdfParse = require('pdf-parse');

// Configuration
const CONFIG = {
  iris: {
    host: 'localhost',
    port: 1972,
    namespace: 'USER',
    username: 'demo',
    password: 'demo'
  },
  pdfDirectory: '/Users/tdyar/ws/isc_md/pdfs',
  maxPdfs: 3 // Limit for demo
};

let irisConnection = null;

/**
 * Initialize IRIS connection
 */
async function connectToIris() {
  if (!irisConnection) {
    console.log('🔌 Connecting to IRIS...');
    irisConnection = new IRIS(
      CONFIG.iris.host,
      CONFIG.iris.port,
      CONFIG.iris.namespace,
      CONFIG.iris.username,
      CONFIG.iris.password
    );
    console.log('✅ Connected to IRIS successfully');
  }
  return irisConnection;
}

/**
 * Create table for documents
 */
async function createTable() {
  const iris = await connectToIris();
  
  const createSQL = `
    CREATE TABLE IF NOT EXISTS SimpleDocuments (
      id VARCHAR(255) PRIMARY KEY,
      title VARCHAR(500),
      content LONGVARCHAR,
      sourceFile VARCHAR(255),
      created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
  `;
  
  try {
    await iris.sql(createSQL);
    console.log('📋 Table SimpleDocuments created/verified');
  } catch (error) {
    console.log('📋 Table note:', error.message);
  }
}

/**
 * Extract text from PDF
 */
async function extractPdfText(pdfPath) {
  try {
    const dataBuffer = fs.readFileSync(pdfPath);
    const data = await pdfParse(dataBuffer);
    return data.text.replace(/\s+/g, ' ').trim().substring(0, 2000); // Limit size for demo
  } catch (error) {
    console.error(`❌ Error extracting ${path.basename(pdfPath)}:`, error.message);
    return null;
  }
}

/**
 * Index a document
 */
async function indexDocument(id, title, content, sourceFile) {
  const iris = await connectToIris();
  
  const insertSQL = `
    INSERT INTO SimpleDocuments (id, title, content, sourceFile)
    VALUES (?, ?, ?, ?)
  `;
  
  try {
    await iris.sql(insertSQL, [id, title, content, sourceFile]);
    return true;
  } catch (error) {
    console.error(`❌ Error indexing ${id}:`, error.message);
    return false;
  }
}

/**
 * Search documents using simple text matching
 */
async function searchDocuments(query, limit = 5) {
  const iris = await connectToIris();
  
  const searchSQL = `
    SELECT TOP ? id, title, sourceFile, 
           SUBSTRING(content, 1, 200) as excerpt
    FROM SimpleDocuments
    WHERE UPPER(content) LIKE UPPER(?)
    ORDER BY id
  `;
  
  try {
    const result = await iris.sql(searchSQL, [limit, `%${query}%`]);
    return result.rows || [];
  } catch (error) {
    console.error(`❌ Search error:`, error.message);
    return [];
  }
}

/**
 * Main indexing function
 */
async function indexPdfs() {
  console.log('\n🚀 === IRIS RAG DEMONSTRATION ===');
  console.log(`📁 PDF Directory: ${CONFIG.pdfDirectory}`);
  
  // Get PDF files
  const pdfFiles = fs.readdirSync(CONFIG.pdfDirectory)
    .filter(file => file.toLowerCase().endsWith('.pdf'))
    .slice(0, CONFIG.maxPdfs)
    .map(file => path.join(CONFIG.pdfDirectory, file));
  
  console.log(`📚 Processing ${pdfFiles.length} PDFs for demonstration`);
  
  // Initialize
  await connectToIris();
  await createTable();
  
  let indexed = 0;
  
  // Process PDFs
  for (let i = 0; i < pdfFiles.length; i++) {
    const pdfFile = pdfFiles[i];
    const fileName = path.basename(pdfFile);
    console.log(`\n📖 [${i + 1}/${pdfFiles.length}] Processing: ${fileName}`);
    
    const content = await extractPdfText(pdfFile);
    if (!content) continue;
    
    const docId = `doc_${i + 1}`;
    const title = fileName.replace('.pdf', '');
    
    const success = await indexDocument(docId, title, content, fileName);
    if (success) {
      indexed++;
      console.log(`   ✅ Indexed successfully`);
    }
  }
  
  console.log(`\n📊 === INDEXING COMPLETE ===`);
  console.log(`✅ Successfully indexed: ${indexed} documents`);
}

/**
 * Demonstrate search functionality
 */
async function demonstrateSearch() {
  console.log('\n🔍 === SEARCH DEMONSTRATION ===');
  
  const queries = [
    'IRIS',
    'analytics',
    'database',
    'SQL',
    'vector'
  ];
  
  for (const query of queries) {
    console.log(`\n🔍 Searching for: "${query}"`);
    
    const results = await searchDocuments(query, 3);
    
    if (results.length > 0) {
      console.log(`   ✅ Found ${results.length} results:`);
      results.forEach((result, index) => {
        console.log(`   ${index + 1}. 📄 ${result[1]} (${result[2]})`);
        console.log(`      📝 ${result[3]}...`);
      });
    } else {
      console.log(`   ❌ No results found`);
    }
  }
}

/**
 * Show current indexed documents
 */
async function showIndexedDocuments() {
  console.log('\n📋 === INDEXED DOCUMENTS ===');
  
  const iris = await connectToIris();
  
  try {
    const result = await iris.sql('SELECT id, title, sourceFile FROM SimpleDocuments ORDER BY id');
    const docs = result.rows || [];
    
    if (docs.length > 0) {
      console.log(`📚 Found ${docs.length} indexed documents:`);
      docs.forEach((doc, index) => {
        console.log(`   ${index + 1}. ${doc[1]} (${doc[2]})`);
      });
    } else {
      console.log('📭 No documents found in index');
    }
  } catch (error) {
    console.error('❌ Error retrieving documents:', error.message);
  }
}

/**
 * Main execution
 */
async function main() {
  try {
    const args = process.argv.slice(2);
    
    if (args.includes('--search-only')) {
      await connectToIris();
      await showIndexedDocuments();
      await demonstrateSearch();
    } else if (args.includes('--show-docs')) {
      await connectToIris();
      await showIndexedDocuments();
    } else {
      await indexPdfs();
      await showIndexedDocuments();
      await demonstrateSearch();
    }
    
    console.log('\n🎉 === DEMONSTRATION COMPLETE ===');
    console.log('✅ IRIS RAG functionality working successfully!');
    
  } catch (error) {
    console.error('💥 Fatal error:', error.message);
  } finally {
    if (irisConnection) {
      try {
        await irisConnection.close();
        console.log('🔌 IRIS connection closed');
      } catch (e) {
        // Ignore close errors
      }
    }
  }
}

// Run the demonstration
if (require.main === module) {
  main();
}

module.exports = { indexPdfs, demonstrateSearch, searchDocuments };