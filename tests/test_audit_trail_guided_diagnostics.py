#!/usr/bin/env python3
"""
Audit Trail Guided Pipeline Diagnostics

Uses the SQL audit trail system to diagnose exactly what's failing
in each broken pipeline, providing precise fixes guided by real database operations.
"""

import pytest
import json
import logging
from typing import Dict, Any, List

from common.sql_audit_logger import get_sql_audit_logger, sql_audit_context
from common.database_audit_middleware import patch_iris_connection_manager, DatabaseOperationCounter
from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager
from common.utils import get_llm_func

# Import broken pipelines for diagnosis
from iris_rag.pipelines.hyde import HyDERAGPipeline
from iris_rag.pipelines.colbert import ColBERTRAGPipeline
from iris_rag.pipelines.crag import CRAGPipeline
from iris_rag.pipelines.graphrag import GraphRAGPipeline
from iris_rag.pipelines.noderag import NodeRAGPipeline

# Import proper data ingestion fixtures
from tests.fixtures.data_ingestion import (
    clean_database,
    basic_test_documents,
    colbert_test_data,
    graphrag_test_data,
    crag_test_data,
    complete_test_data
)

logger = logging.getLogger(__name__)


@pytest.mark.integration
class TestAuditTrailGuidedDiagnostics:
    """
    Audit trail guided diagnostics for broken pipelines.
    
    Each test uses the SQL audit trail to pinpoint exactly where
    real database operations fail vs where mocks succeeded.
    """
    
    @pytest.fixture(autouse=True)
    def setup_audit_logging(self):
        """Setup SQL audit logging for diagnostic tests."""
        # Clear audit trail
        audit_logger = get_sql_audit_logger()
        audit_logger.clear_audit_trail()
        
        # Patch for real operation logging
        patch_iris_connection_manager()
        
        yield
        
        # Generate diagnostic report
        report = audit_logger.generate_audit_report()
        counter = DatabaseOperationCounter()
        analysis = counter.count_operations()
        
        print(f"\n🔍 DIAGNOSTIC AUDIT REPORT:")
        print(f"📊 {report.get('total_operations', 0)} total operations")
        print(f"🔴 {report.get('real_database_operations', 0)} real database operations")
        print(f"🟡 {report.get('mocked_operations', 0)} mocked operations")
        
        if analysis['real_operations_detail']:
            print(f"\n🔴 Real SQL Operations:")
            for op in analysis['real_operations_detail']:
                print(f"   {op['operation_id']}: {op['sql'][:80]}...")
                if op['execution_time_ms']:
                    print(f"      Time: {op['execution_time_ms']:.2f}ms, Results: {op['result_count']}")
    
    def test_hyde_pipeline_diagnostic(self, basic_test_documents):
        """
        Diagnose HyDE pipeline failure: 'Document missing page_content'
        
        Real database test showed: HyDE: Document missing page_content
        
        This test uses proper data ingestion fixtures to ensure consistent test data.
        """
        audit_logger = get_sql_audit_logger()
        
        print(f"\n🔍 DIAGNOSING HYDE PIPELINE WITH PROPER TEST DATA")
        print(f"Test documents loaded: {len(basic_test_documents)}")
        print(f"Real database error: 'Document missing page_content'")
        
        try:
            with sql_audit_context('real_database', 'HyDE', 'hyde_diagnostic'):
                connection_manager = ConnectionManager()
                config_manager = ConfigurationManager()
                llm_func = get_llm_func(provider='stub')
                
                pipeline = HyDERAGPipeline(connection_manager, config_manager, llm_func=llm_func)
                
                print(f"✅ HyDE pipeline creation successful")
                
                # Test the query execution step by step
                print(f"🔍 Testing HyDE query execution...")
                result = pipeline.query("diabetes treatment", top_k=3)
                
                print(f"📊 HyDE result keys: {list(result.keys())}")
                print(f"📊 Retrieved documents count: {len(result.get('retrieved_documents', []))}")
                
                # Examine document structure
                docs = result.get('retrieved_documents', [])
                if docs:
                    for i, doc in enumerate(docs[:2]):
                        print(f"🔍 Document {i+1} analysis:")
                        print(f"   Type: {type(doc)}")
                        print(f"   Has page_content attr: {hasattr(doc, 'page_content')}")
                        if hasattr(doc, 'page_content'):
                            content = getattr(doc, 'page_content')
                            print(f"   page_content type: {type(content)}")
                            print(f"   page_content length: {len(str(content)) if content else 0}")
                            print(f"   page_content preview: {str(content)[:100] if content else 'None'}...")
                        
                        print(f"   Has metadata attr: {hasattr(doc, 'metadata')}")
                        if hasattr(doc, 'metadata'):
                            print(f"   Metadata: {getattr(doc, 'metadata', {})}")
                        
                        # Check all attributes
                        attrs = [attr for attr in dir(doc) if not attr.startswith('_')]
                        print(f"   All attributes: {attrs}")
                else:
                    print(f"❌ No documents retrieved - this might be the issue")
        
        except Exception as e:
            print(f"❌ HyDE diagnostic failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Analyze SQL operations for HyDE
        hyde_ops = audit_logger.get_operations_by_pipeline('HyDE')
        print(f"\n📊 HyDE SQL Operations: {len(hyde_ops)}")
        for op in hyde_ops:
            print(f"   {op.operation_id}: {op.sql_statement[:80]}...")
            if op.error:
                print(f"      ❌ ERROR: {op.error}")
    
    def test_colbert_pipeline_diagnostic(self, colbert_test_data):
        """
        Diagnose ColBERT pipeline failure: 'No retrieved_documents in result'
        
        Real database test showed: ColBERT: No retrieved_documents in result
        
        This test uses proper data ingestion fixtures to ensure consistent test data.
        """
        audit_logger = get_sql_audit_logger()
        
        print(f"\n🔍 DIAGNOSING COLBERT PIPELINE WITH PROPER TEST DATA")
        print(f"Test documents loaded: {len(colbert_test_data)}")
        print(f"Real database error: 'No retrieved_documents in result'")
        
        try:
            with sql_audit_context('real_database', 'ColBERT', 'colbert_diagnostic'):
                connection_manager = ConnectionManager()
                config_manager = ConfigurationManager()
                llm_func = get_llm_func(provider='stub')
                
                pipeline = ColBERTRAGPipeline(connection_manager, config_manager, llm_func=llm_func)
                
                print(f"✅ ColBERT pipeline creation successful")
                
                # Test query execution
                print(f"🔍 Testing ColBERT query execution...")
                result = pipeline.query("diabetes treatment", top_k=3)
                
                print(f"📊 ColBERT result type: {type(result)}")
                print(f"📊 ColBERT result: {result}")
                
                # Check if result is a list instead of dict
                if isinstance(result, list):
                    print(f"❌ ISSUE FOUND: Result is list, not dict with 'retrieved_documents' key")
                    print(f"   List contents: {result}")
                elif isinstance(result, dict):
                    print(f"📊 Result keys: {list(result.keys())}")
                    if 'retrieved_documents' not in result:
                        print(f"❌ ISSUE FOUND: 'retrieved_documents' key missing from result dict")
                        print(f"   Available keys: {list(result.keys())}")
                else:
                    print(f"❌ ISSUE FOUND: Unexpected result type: {type(result)}")
        
        except Exception as e:
            print(f"❌ ColBERT diagnostic failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Analyze SQL operations for ColBERT
        colbert_ops = audit_logger.get_operations_by_pipeline('ColBERT')
        print(f"\n📊 ColBERT SQL Operations: {len(colbert_ops)}")
        for op in colbert_ops:
            print(f"   {op.operation_id}: {op.sql_statement[:80]}...")
            if op.error:
                print(f"      ❌ ERROR: {op.error}")
    
    def test_crag_pipeline_diagnostic(self, crag_test_data):
        """
        Diagnose CRAG pipeline failure: 'No retrieved_documents in result'
        
        Real database test showed: CRAG: No retrieved_documents in result
        
        This test uses proper data ingestion fixtures to ensure consistent test data.
        """
        audit_logger = get_sql_audit_logger()
        
        print(f"\n🔍 DIAGNOSING CRAG PIPELINE WITH PROPER TEST DATA")
        print(f"Test documents loaded: {len(crag_test_data)}")
        print(f"Real database error: 'No retrieved_documents in result'")
        
        try:
            with sql_audit_context('real_database', 'CRAG', 'crag_diagnostic'):
                connection_manager = ConnectionManager()
                config_manager = ConfigurationManager()
                llm_func = get_llm_func(provider='stub')
                
                pipeline = CRAGPipeline(connection_manager, config_manager, llm_func=llm_func)
                
                print(f"✅ CRAG pipeline creation successful")
                
                # Test query execution
                print(f"🔍 Testing CRAG query execution...")
                result = pipeline.query("diabetes treatment", top_k=3)
                
                print(f"📊 CRAG result type: {type(result)}")
                print(f"📊 CRAG result: {result}")
                
                # Detailed analysis
                if isinstance(result, dict):
                    print(f"📊 Result keys: {list(result.keys())}")
                    if 'retrieved_documents' in result:
                        docs = result['retrieved_documents']
                        print(f"📊 Retrieved documents type: {type(docs)}")
                        print(f"📊 Retrieved documents count: {len(docs) if docs else 0}")
                        if docs:
                            print(f"📊 First document: {docs[0]}")
                    else:
                        print(f"❌ ISSUE FOUND: 'retrieved_documents' key missing")
        
        except Exception as e:
            print(f"❌ CRAG diagnostic failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Analyze SQL operations for CRAG
        crag_ops = audit_logger.get_operations_by_pipeline('CRAG')
        print(f"\n📊 CRAG SQL Operations: {len(crag_ops)}")
        for op in crag_ops:
            print(f"   {op.operation_id}: {op.sql_statement[:80]}...")
            if op.error:
                print(f"      ❌ ERROR: {op.error}")
    
    def test_graphrag_pipeline_diagnostic(self, graphrag_test_data):
        """
        Diagnose GraphRAG pipeline failure: 'No documents retrieved'
        
        Real database test showed: GraphRAG: No documents retrieved
        
        This test uses proper data ingestion fixtures to ensure consistent test data.
        """
        audit_logger = get_sql_audit_logger()
        
        print(f"\n🔍 DIAGNOSING GRAPHRAG PIPELINE WITH PROPER TEST DATA")
        print(f"Test documents loaded: {len(graphrag_test_data)}")
        print(f"Real database error: 'No documents retrieved'")
        
        try:
            with sql_audit_context('real_database', 'GraphRAG', 'graphrag_diagnostic'):
                connection_manager = ConnectionManager()
                config_manager = ConfigurationManager()
                llm_func = get_llm_func(provider='stub')
                
                pipeline = GraphRAGPipeline(connection_manager, config_manager, llm_func=llm_func)
                
                print(f"✅ GraphRAG pipeline creation successful")
                
                # DEBUGGING: Check entities are visible to GraphRAG using proper abstractions
                print(f"🔍 Checking if entities are visible to GraphRAG pipeline...")
                
                # Use SchemaManager for proper abstraction instead of direct SQL
                from iris_rag.storage.schema_manager import SchemaManager
                schema_manager = SchemaManager(connection_manager, config_manager)
                
                entity_count = schema_manager.get_table_count("RAG.DocumentEntities")
                print(f"   DocumentEntities count from GraphRAG connection: {entity_count}")
                
                if entity_count > 0:
                    sample_entities = schema_manager.get_sample_entities(limit=3)
                    entity_names = [entity['name'] for entity in sample_entities]
                    print(f"   Sample entities: {entity_names}")
                
                node_count = schema_manager.get_table_count("RAG.KnowledgeGraphNodes")
                print(f"   KnowledgeGraphNodes count from GraphRAG connection: {node_count}")
                
                # Test query execution
                print(f"🔍 Testing GraphRAG query execution...")
                result = pipeline.query("diabetes treatment", top_k=3)
                
                print(f"📊 GraphRAG result: {result}")
                
                # Check retrieved documents
                docs = result.get('retrieved_documents', [])
                print(f"📊 Retrieved documents count: {len(docs)}")
                
                if len(docs) == 0:
                    print(f"❌ ISSUE CONFIRMED: Zero documents retrieved")
                    
                    # Check if GraphRAG tables exist and have data using SchemaManager abstractions
                    schema_manager = SchemaManager(connection_manager, config_manager)
                    
                    try:
                        # Use SchemaManager for comprehensive GraphRAG table analysis
                        entity_statistics = schema_manager.get_entity_statistics()
                        print(f"📊 DocumentEntities count: {entity_statistics['total_entities']}")
                        print(f"📊 Documents with entities: {entity_statistics['documents_with_entities']}")
                        
                        # Check table existence using schema manager abstraction
                        nodes_exist = schema_manager.table_exists("KnowledgeGraphNodes")
                        print(f"📊 KnowledgeGraphNodes table exists: {nodes_exist}")
                        
                        if nodes_exist:
                            node_count = schema_manager.get_table_count("RAG.KnowledgeGraphNodes")
                            print(f"📊 KnowledgeGraphNodes count: {node_count}")
                        else:
                            node_count = 0
                            print(f"📊 KnowledgeGraphNodes table missing")
                        
                        # Check additional GraphRAG tables using abstractions
                        edges_exist = schema_manager.table_exists("KnowledgeGraphEdges")
                        print(f"📊 KnowledgeGraphEdges table exists: {edges_exist}")
                        
                        # Validate GraphRAG data completeness
                        if entity_statistics['total_entities'] == 0 and node_count == 0:
                            print(f"❌ ROOT CAUSE: GraphRAG requires entity/graph data but tables are empty")
                            print(f"   Solution: Populate entities using SetupOrchestrator.setup_pipeline('graphrag')")
                        elif entity_statistics['total_entities'] > 0 and node_count == 0:
                            print(f"⚠️ PARTIAL SETUP: Entities exist but knowledge graph nodes missing")
                        
                    except Exception as table_error:
                        print(f"❌ ROOT CAUSE: GraphRAG tables don't exist or are inaccessible: {table_error}")
                        print(f"   Solution: Use SetupOrchestrator to ensure proper table creation")
        
        except Exception as e:
            print(f"❌ GraphRAG diagnostic failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Analyze SQL operations for GraphRAG
        graphrag_ops = audit_logger.get_operations_by_pipeline('GraphRAG')
        print(f"\n📊 GraphRAG SQL Operations: {len(graphrag_ops)}")
        for op in graphrag_ops:
            print(f"   {op.operation_id}: {op.sql_statement[:80]}...")
            if op.error:
                print(f"      ❌ ERROR: {op.error}")
    
    def test_noderag_pipeline_diagnostic(self, crag_test_data):
        """
        Diagnose NodeRAG pipeline failure: 'No retrieved_documents in result'
        
        Real database test showed: NodeRAG: No retrieved_documents in result
        
        This test uses proper data ingestion fixtures to ensure consistent test data.
        NodeRAG uses the same chunk data as CRAG.
        """
        audit_logger = get_sql_audit_logger()
        
        print(f"\n🔍 DIAGNOSING NODERAG PIPELINE WITH PROPER TEST DATA")
        print(f"Test documents loaded: {len(crag_test_data)}")
        print(f"Real database error: 'No retrieved_documents in result'")
        
        try:
            with sql_audit_context('real_database', 'NodeRAG', 'noderag_diagnostic'):
                connection_manager = ConnectionManager()
                config_manager = ConfigurationManager()
                llm_func = get_llm_func(provider='stub')
                
                pipeline = NodeRAGPipeline(connection_manager, config_manager, llm_func=llm_func)
                
                print(f"✅ NodeRAG pipeline creation successful")
                
                # Test query execution
                print(f"🔍 Testing NodeRAG query execution...")
                result = pipeline.query("diabetes treatment", top_k=3)
                
                print(f"📊 NodeRAG result type: {type(result)}")
                print(f"📊 NodeRAG result: {result}")
                
                # Check the result structure
                if isinstance(result, dict):
                    print(f"📊 Result keys: {list(result.keys())}")
                    if 'retrieved_documents' in result:
                        docs = result['retrieved_documents']
                        print(f"📊 Retrieved documents: {docs}")
                        print(f"📊 Document count: {len(docs) if docs else 0}")
                    else:
                        print(f"❌ ISSUE FOUND: 'retrieved_documents' key missing from result")
        
        except Exception as e:
            print(f"❌ NodeRAG diagnostic failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Analyze SQL operations for NodeRAG
        noderag_ops = audit_logger.get_operations_by_pipeline('NodeRAG')
        print(f"\n📊 NodeRAG SQL Operations: {len(noderag_ops)}")
        for op in noderag_ops:
            print(f"   {op.operation_id}: {op.sql_statement[:80]}...")
            if op.error:
                print(f"      ❌ ERROR: {op.error}")
    
    def test_comprehensive_failure_analysis(self):
        """
        Comprehensive analysis of all pipeline failures using audit trail.
        """
        audit_logger = get_sql_audit_logger()
        
        print(f"\n🔍 COMPREHENSIVE FAILURE ANALYSIS")
        print(f"Using SQL audit trail to identify common failure patterns")
        
        # Summary of known failures from real database test
        failures = {
            'HyDE': 'Document missing page_content',
            'ColBERT': 'No retrieved_documents in result', 
            'CRAG': 'No retrieved_documents in result',
            'GraphRAG': 'No documents retrieved',
            'NodeRAG': 'No retrieved_documents in result'
        }
        
        print(f"\n📊 FAILURE PATTERN ANALYSIS:")
        
        # Group by failure type
        missing_key_failures = [p for p, error in failures.items() if 'No retrieved_documents in result' in error]
        content_failures = [p for p, error in failures.items() if 'page_content' in error]
        empty_result_failures = [p for p, error in failures.items() if 'No documents retrieved' in error]
        
        print(f"🔴 Missing 'retrieved_documents' key: {missing_key_failures}")
        print(f"🔴 Document content structure issues: {content_failures}")
        print(f"🔴 Empty results: {empty_result_failures}")
        
        print(f"\n💡 HYPOTHESIS:")
        print(f"1. Missing key failures suggest inconsistent query() method return format")
        print(f"2. Content failures suggest document construction/parsing issues")
        print(f"3. Empty results suggest missing dependencies (tables, embeddings)")
        
        # Get all operations to see patterns
        all_ops = audit_logger.operations
        error_ops = [op for op in all_ops if op.error]
        
        if error_ops:
            print(f"\n❌ SQL ERRORS DETECTED: {len(error_ops)}")
            for op in error_ops:
                print(f"   {op.pipeline_name}: {op.error}")
        else:
            print(f"\n🟡 NO SQL ERRORS: Issues are likely in result processing, not database access")
        
        return {
            'missing_key_failures': missing_key_failures,
            'content_failures': content_failures,
            'empty_result_failures': empty_result_failures,
            'sql_errors': len(error_ops)
        }


if __name__ == "__main__":
    # Run diagnostic tests
    pytest.main([__file__, "-v", "-s", "--tb=short"])