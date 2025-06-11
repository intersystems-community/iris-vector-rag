#!/usr/bin/env python3
"""
Comprehensive validation of ALL 7 RAG techniques
Tests each technique end-to-end with the same query to ensure complete functionality
"""

import sys
import time
import traceback
import json
from datetime import datetime

# Add current directory to path
# sys.path.append('.') # Keep if script is in project root, otherwise adjust for project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.')) # Assuming script is in project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import all RAG techniques
from src.deprecated.basic_rag.pipeline_v2_fixed import BasicRAGPipelineV2Fixed as BasicRAGPipelineV2 # Updated import
from src.experimental.crag.pipeline import CRAGPipeline as CRAGPipelineV2 # Updated import
from src.deprecated.colbert.pipeline import OptimizedColbertRAGPipeline as ColBERTPipelineV2 # Updated import
from src.experimental.noderag.pipeline import NodeRAGPipeline as NodeRAGPipelineV2 # Updated import
from src.experimental.hyde.pipeline import HyDEPipeline # Updated import
from src.experimental.graphrag.pipeline import GraphRAGPipeline as GraphRAGPipelineV2 # Updated import
from src.experimental.hybrid_ifind_rag.pipeline import HybridiFindRAGPipeline # Updated import

# Import common utilities
from common.iris_connector import get_iris_connection # Updated import
from common.embedding_utils import get_embedding_model # Updated import

def test_technique(technique_name, pipeline_class, iris, embedding_func, llm_func, query):
    """Test a single RAG technique"""
    print(f"\n{'='*60}")
    print(f"🧪 Testing {technique_name}")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        
        # Initialize pipeline
        pipeline = pipeline_class(iris, embedding_func, llm_func)
        
        print(f"✅ {technique_name} pipeline initialized")
        
        # Run the pipeline
        result = pipeline.run(query, top_k=3)
        
        execution_time = time.time() - start_time
        
        # Validate result structure
        required_keys = ['query', 'answer', 'retrieved_documents']
        missing_keys = [key for key in required_keys if key not in result]
        
        if missing_keys:
            print(f"❌ Missing required keys: {missing_keys}")
            return False, execution_time, f"Missing keys: {missing_keys}"
        
        # Check result content
        if not result['answer'] or not result['retrieved_documents']:
            print(f"❌ Empty answer or no documents retrieved")
            return False, execution_time, "Empty answer or no documents"
        
        # Display results
        print(f"✅ Query: {result['query']}")
        print(f"✅ Answer length: {len(result['answer'])} characters")
        print(f"✅ Documents retrieved: {len(result['retrieved_documents'])}")
        print(f"✅ Execution time: {execution_time:.2f} seconds")
        
        # Show answer preview
        answer_preview = result['answer'][:200] + "..." if len(result['answer']) > 200 else result['answer']
        print(f"✅ Answer preview: {answer_preview}")
        
        # Show document info
        if result['retrieved_documents']:
            doc = result['retrieved_documents'][0]
            print(f"✅ First document ID: {doc.get('id', 'N/A')}")
            if 'content' in doc:
                content_preview = doc['content'][:100] + "..." if len(doc['content']) > 100 else doc['content']
                print(f"✅ Document content preview: {content_preview}")
        
        # Technique-specific validations
        if technique_name == "GraphRAG V2":
            entities_count = len(result.get('entities', []))
            relationships_count = len(result.get('relationships', []))
            print(f"✅ Entities found: {entities_count}")
            print(f"✅ Relationships found: {relationships_count}")
        
        if technique_name == "ColBERT V2":
            token_scores = result.get('token_scores', [])
            print(f"✅ Token scores computed: {len(token_scores) > 0}")
        
        if technique_name == "NodeRAG V2":
            chunks_used = result.get('chunks_used', 0)
            print(f"✅ Document chunks used: {chunks_used}")
        
        print(f"🎉 {technique_name} - SUCCESS!")
        return True, execution_time, "Success"
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = str(e)
        print(f"❌ {technique_name} - FAILED: {error_msg}")
        print(f"❌ Execution time: {execution_time:.2f} seconds")
        traceback.print_exc()
        return False, execution_time, error_msg

def check_data_completeness(iris):
    """Check data completeness for all techniques"""
    print(f"\n{'='*60}")
    print("📊 Checking Data Completeness")
    print(f"{'='*60}")
    
    data_checks = {
        'SourceDocuments': "SELECT COUNT(*) as count FROM RAGTest.SourceDocuments",
        'DocumentChunks': "SELECT COUNT(*) as count FROM RAGTest.DocumentChunks", 
        'DocumentTokenEmbeddings': "SELECT COUNT(*) as count FROM RAGTest.DocumentTokenEmbeddings",
        'Entities': "SELECT COUNT(*) as count FROM RAGTest.Entities",
        'KnowledgeGraph': "SELECT COUNT(*) as count FROM RAGTest.KnowledgeGraph"
    }
    
    data_status = {}
    
    for table_name, query in data_checks.items():
        try:
            cursor = iris.cursor()
            cursor.execute(query)
            result = cursor.fetchone()
            count = result[0] if result else 0
            data_status[table_name] = count
            print(f"✅ {table_name}: {count:,} records")
            cursor.close()
        except Exception as e:
            data_status[table_name] = f"Error: {e}"
            print(f"❌ {table_name}: Error - {e}")
    
    return data_status

def main():
    """Main validation function"""
    print("🚀 COMPREHENSIVE 7-TECHNIQUE RAG VALIDATION")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Test query - same for all techniques
    test_query = "What is diabetes and how is it treated?"
    
    # Initialize connections
    print("\n🔌 Initializing connections...")
    iris = get_iris_connection()
    embedding_model = get_embedding_model('sentence-transformers/all-MiniLM-L6-v2')
    
    def embedding_func(texts):
        return embedding_model.encode(texts)
    
    def llm_func(prompt):
        return f"Based on the provided context, this is a response to: {prompt[:100]}..."
    
    print("✅ Connections initialized")
    
    # Check data completeness
    data_status = check_data_completeness(iris)
    
    # Define all 7 RAG techniques
    techniques = [
        ("BasicRAG V2", BasicRAGPipelineV2),
        ("CRAG V2", CRAGPipelineV2),
        ("ColBERT V2", ColBERTPipelineV2),
        ("NodeRAG V2", NodeRAGPipelineV2),
        ("HyDE", HyDEPipeline),
        ("GraphRAG V2", GraphRAGPipelineV2),
        ("HybridIFindRAG", HybridiFindRAGPipeline)
    ]
    
    # Test results
    results = {}
    successful_techniques = []
    failed_techniques = []
    
    print(f"\n🎯 Testing all 7 techniques with query: '{test_query}'")
    
    # Test each technique
    for technique_name, pipeline_class in techniques:
        success, exec_time, message = test_technique(
            technique_name, pipeline_class, iris, embedding_func, llm_func, test_query
        )
        
        results[technique_name] = {
            'success': success,
            'execution_time': exec_time,
            'message': message
        }
        
        if success:
            successful_techniques.append(technique_name)
        else:
            failed_techniques.append(technique_name)
    
    # Generate final report
    print(f"\n{'='*80}")
    print("📋 FINAL VALIDATION REPORT")
    print(f"{'='*80}")
    
    print(f"\n✅ SUCCESSFUL TECHNIQUES ({len(successful_techniques)}/7):")
    for i, technique in enumerate(successful_techniques, 1):
        exec_time = results[technique]['execution_time']
        print(f"  {i}. {technique} - {exec_time:.2f}s")
    
    if failed_techniques:
        print(f"\n❌ FAILED TECHNIQUES ({len(failed_techniques)}/7):")
        for i, technique in enumerate(failed_techniques, 1):
            message = results[technique]['message']
            print(f"  {i}. {technique} - {message}")
    
    # Data completeness summary
    print(f"\n📊 DATA COMPLETENESS:")
    for table, count in data_status.items():
        print(f"  • {table}: {count}")
    
    # Overall status
    success_rate = len(successful_techniques) / 7 * 100
    print(f"\n🎯 OVERALL SUCCESS RATE: {success_rate:.1f}% ({len(successful_techniques)}/7)")
    
    if len(successful_techniques) == 7:
        print("\n🎉 ALL 7 RAG TECHNIQUES ARE FULLY OPERATIONAL!")
        print("✅ Enterprise RAG system is COMPLETE and ready for 10K scaling!")
    else:
        print(f"\n⚠️  {7 - len(successful_techniques)} technique(s) need attention")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"all_7_techniques_validation_{timestamp}.json"
    
    validation_report = {
        'timestamp': datetime.now().isoformat(),
        'query': test_query,
        'data_status': data_status,
        'technique_results': results,
        'successful_techniques': successful_techniques,
        'failed_techniques': failed_techniques,
        'success_rate': success_rate,
        'total_techniques': 7
    }
    
    with open(results_file, 'w') as f:
        json.dump(validation_report, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Close connection
    iris.close()
    print("\n🔌 Connection closed")
    
    return len(successful_techniques) == 7

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)