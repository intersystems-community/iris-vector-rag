#!/usr/bin/env python3
"""
End-to-End GraphRAG Validation Script

Tests complete GraphRAG system deployment and validation:
1. Database connectivity
2. Schema deployment via SchemaManager
3. Document loading with entity extraction  
4. Knowledge graph traversal queries
5. Fail-hard validation
6. Performance comparison with BasicRAG
"""

import os
import sys
import time
import json
import logging
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager
from iris_rag.storage.schema_manager import SchemaManager
from iris_rag.pipelines.graphrag import GraphRAGPipeline, KnowledgeGraphNotPopulatedException
from iris_rag.pipelines.basic import BasicRAGPipeline
from iris_rag.core.models import Document

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GraphRAGValidator:
    """Complete GraphRAG system validation."""
    
    def __init__(self):
        """Initialize validator with system components."""
        self.config_manager = None
        self.connection_manager = None
        self.schema_manager = None
        self.graphrag_pipeline = None
        self.basicrag_pipeline = None
        self.validation_results = {}
        
    def initialize_components(self) -> bool:
        """Initialize all system components."""
        try:
            logger.info("🔧 Initializing system components...")
            
            # Initialize configuration manager
            self.config_manager = ConfigurationManager()
            logger.info("✅ ConfigurationManager initialized")
            
            # Initialize connection manager
            self.connection_manager = ConnectionManager()
            logger.info("✅ ConnectionManager initialized")
            
            # Test database connectivity
            connection = self.connection_manager.get_connection()
            cursor = connection.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            
            if result[0] == 1:
                logger.info("✅ Database connectivity confirmed")
            else:
                logger.error("❌ Database connectivity test failed")
                return False
                
            # Initialize schema manager
            self.schema_manager = SchemaManager(self.connection_manager, self.config_manager)
            logger.info("✅ SchemaManager initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            return False
    
    def deploy_graphrag_schema(self) -> bool:
        """Deploy GraphRAG schema using SchemaManager."""
        try:
            logger.info("🔧 Deploying GraphRAG schema...")
            
            # Check if entities table needs migration
            entities_needs_migration = self.schema_manager.needs_migration("Entities", "graphrag")
            relationships_needs_migration = self.schema_manager.needs_migration("EntityRelationships", "graphrag")
            
            logger.info(f"Entities table needs migration: {entities_needs_migration}")
            logger.info(f"EntityRelationships table needs migration: {relationships_needs_migration}")
            
            # Deploy Entities table
            if entities_needs_migration:
                logger.info("🔧 Migrating Entities table...")
                success = self.schema_manager.migrate_table("Entities", preserve_data=False, pipeline_type="graphrag")
                if not success:
                    logger.error("❌ Failed to migrate Entities table")
                    return False
                logger.info("✅ Entities table migrated successfully")
            else:
                logger.info("✅ Entities table already up to date")
            
            # Deploy EntityRelationships table
            if relationships_needs_migration:
                logger.info("🔧 Migrating EntityRelationships table...")
                success = self.schema_manager.migrate_table("EntityRelationships", preserve_data=False, pipeline_type="graphrag")
                if not success:
                    logger.error("❌ Failed to migrate EntityRelationships table")
                    return False
                logger.info("✅ EntityRelationships table migrated successfully")
            else:
                logger.info("✅ EntityRelationships table already up to date")
            
            # Ensure SourceDocuments table exists for document storage
            sourcedocs_needs_migration = self.schema_manager.needs_migration("SourceDocuments", "graphrag")
            if sourcedocs_needs_migration:
                logger.info("🔧 Migrating SourceDocuments table...")
                success = self.schema_manager.migrate_table("SourceDocuments", preserve_data=False, pipeline_type="graphrag")
                if not success:
                    logger.error("❌ Failed to migrate SourceDocuments table")
                    return False
                logger.info("✅ SourceDocuments table migrated successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Schema deployment failed: {e}")
            return False
    
    def validate_schema_deployment(self) -> bool:
        """Validate that schema was deployed correctly."""
        try:
            logger.info("🔍 Validating schema deployment...")
            
            connection = self.connection_manager.get_connection()
            cursor = connection.cursor()
            
            # Check that tables exist and have correct structure
            tables_to_check = [
                ("RAG.Entities", ["entity_id", "entity_name", "entity_type", "source_doc_id"]),
                ("RAG.EntityRelationships", ["relationship_id", "source_entity_id", "target_entity_id", "relationship_type"]),
                ("RAG.SourceDocuments", ["doc_id", "text_content", "title"])
            ]
            
            for table_name, required_columns in tables_to_check:
                try:
                    # Check table exists by querying its structure
                    cursor.execute(f"SELECT TOP 1 * FROM {table_name}")
                    columns = [desc[0].lower() for desc in cursor.description]
                    
                    # Verify required columns exist
                    missing_columns = []
                    for col in required_columns:
                        if col.lower() not in columns:
                            missing_columns.append(col)
                    
                    if missing_columns:
                        logger.error(f"❌ Table {table_name} missing columns: {missing_columns}")
                        return False
                    else:
                        logger.info(f"✅ Table {table_name} validated successfully")
                        
                except Exception as e:
                    logger.error(f"❌ Failed to validate table {table_name}: {e}")
                    return False
            
            cursor.close()
            logger.info("✅ Schema validation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Schema validation failed: {e}")
            return False
    
    def load_sample_documents(self) -> bool:
        """Load sample documents with entity extraction."""
        try:
            logger.info("🔧 Loading sample documents with entity extraction...")
            
            # Initialize GraphRAG pipeline
            self.graphrag_pipeline = GraphRAGPipeline(
                connection_manager=self.connection_manager,
                config_manager=self.config_manager
            )
            
            # Create sample documents
            sample_docs = [
                Document(
                    id="test_doc_1",
                    page_content="""
                    Diabetes is a chronic disease that affects how your body turns food into energy.
                    Most of the food you eat is broken down into sugar (glucose) and released into your bloodstream.
                    When your blood sugar goes up, it signals your pancreas to release insulin.
                    Common treatments include metformin, insulin therapy, and lifestyle changes.
                    Symptoms include frequent urination, excessive thirst, and fatigue.
                    """,
                    metadata={"source": "medical_text_1", "title": "Understanding Diabetes"}
                ),
                Document(
                    id="test_doc_2", 
                    page_content="""
                    Hypertension, also known as high blood pressure, is a common condition where
                    the long-term force of the blood against your artery walls is high enough
                    to eventually cause health problems. ACE inhibitors like lisinopril and
                    ARBs like losartan are commonly prescribed medications. Regular exercise
                    and a low-sodium diet can help manage blood pressure.
                    """,
                    metadata={"source": "medical_text_2", "title": "Hypertension Management"}
                ),
                Document(
                    id="test_doc_3",
                    page_content="""
                    Cancer treatment has evolved significantly with immunotherapy drugs like
                    pembrolizumab and nivolumab. These checkpoint inhibitors help the immune
                    system recognize and attack cancer cells. Traditional treatments include
                    chemotherapy, radiation therapy, and surgery.
                    """,
                    metadata={"source": "medical_text_3", "title": "Cancer Treatment Advances"}
                )
            ]
            
            # Load documents with entity extraction
            self.graphrag_pipeline.load_documents("", documents=sample_docs)
            
            logger.info("✅ Sample documents loaded successfully with entity extraction")
            return True
            
        except Exception as e:
            logger.error(f"❌ Document loading failed: {e}")
            return False
    
    def verify_knowledge_graph_population(self) -> bool:
        """Verify entities and relationships are stored in database."""
        try:
            logger.info("🔍 Verifying knowledge graph population...")
            
            connection = self.connection_manager.get_connection()
            cursor = connection.cursor()
            
            # Check entities table
            cursor.execute("SELECT COUNT(*) FROM RAG.Entities")
            entity_count = cursor.fetchone()[0]
            
            # Check relationships table
            cursor.execute("SELECT COUNT(*) FROM RAG.EntityRelationships")
            relationship_count = cursor.fetchone()[0]
            
            cursor.close()
            
            logger.info(f"📊 Knowledge graph contains:")
            logger.info(f"   - {entity_count} entities")
            logger.info(f"   - {relationship_count} relationships")
            
            if entity_count == 0:
                logger.error("❌ No entities found in knowledge graph")
                return False
            
            if relationship_count == 0:
                logger.warning("⚠️ No relationships found in knowledge graph")
                # Continue validation even without relationships
            
            logger.info("✅ Knowledge graph population verified")
            return True
            
        except Exception as e:
            logger.error(f"❌ Knowledge graph verification failed: {e}")
            return False
    
    def test_knowledge_graph_traversal(self) -> bool:
        """Test GraphRAG knowledge graph traversal queries."""
        try:
            logger.info("🔍 Testing knowledge graph traversal...")
            
            test_queries = [
                "What are the symptoms of diabetes?",
                "How is hypertension treated?",
                "What are common cancer treatments?"
            ]
            
            for query in test_queries:
                logger.info(f"🔍 Testing query: '{query}'")
                
                result = self.graphrag_pipeline.query(query, top_k=5)
                
                # Verify result structure
                required_keys = ["query", "answer", "retrieved_documents", "metadata"]
                for key in required_keys:
                    if key not in result:
                        logger.error(f"❌ Missing key '{key}' in query result")
                        return False
                
                # Verify retrieval method
                if result["metadata"]["retrieval_method"] != "knowledge_graph_traversal":
                    logger.error(f"❌ Expected knowledge_graph_traversal, got {result['metadata']['retrieval_method']}")
                    return False
                
                logger.info(f"✅ Query processed successfully - {result['metadata']['num_retrieved']} docs retrieved")
                logger.info(f"   Method: {result['metadata']['retrieval_method']}")
                
            logger.info("✅ Knowledge graph traversal tests passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Knowledge graph traversal test failed: {e}")
            return False
    
    def test_fail_hard_validation(self) -> bool:
        """Test fail-hard validation with empty knowledge graph."""
        try:
            logger.info("🔍 Testing fail-hard validation...")
            
            # Clear knowledge graph tables
            connection = self.connection_manager.get_connection()
            cursor = connection.cursor()
            
            cursor.execute("DELETE FROM RAG.EntityRelationships")
            cursor.execute("DELETE FROM RAG.Entities")
            connection.commit()
            cursor.close()
            
            logger.info("🔧 Cleared knowledge graph tables")
            
            # Create new GraphRAG pipeline instance
            empty_graphrag_pipeline = GraphRAGPipeline(
                connection_manager=self.connection_manager,
                config_manager=self.config_manager
            )
            
            # Attempt query - should fail with KnowledgeGraphNotPopulatedException
            try:
                result = empty_graphrag_pipeline.query("What are the symptoms of diabetes?")
                logger.error("❌ Query should have failed with empty knowledge graph")
                return False
                
            except KnowledgeGraphNotPopulatedException as e:
                logger.info(f"✅ Correctly failed with KnowledgeGraphNotPopulatedException: {e}")
                return True
                
            except Exception as e:
                logger.error(f"❌ Unexpected exception type: {type(e).__name__}: {e}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Fail-hard validation test failed: {e}")
            return False
    
    def compare_graphrag_vs_basicrag(self) -> bool:
        """Compare GraphRAG vs BasicRAG results."""
        try:
            logger.info("🔍 Comparing GraphRAG vs BasicRAG...")
            
            # Re-load documents for comparison
            self.load_sample_documents()
            
            # Initialize BasicRAG pipeline
            self.basicrag_pipeline = BasicRAGPipeline(
                connection_manager=self.connection_manager,
                config_manager=self.config_manager
            )
            
            test_query = "What medications treat diabetes and hypertension?"
            
            # Test GraphRAG
            logger.info("🔍 Testing GraphRAG...")
            graphrag_result = self.graphrag_pipeline.query(test_query, top_k=5)
            
            # Test BasicRAG  
            logger.info("🔍 Testing BasicRAG...")
            basicrag_result = self.basicrag_pipeline.query(test_query, top_k=5)
            
            # Compare results
            logger.info("📊 Comparison Results:")
            logger.info(f"   GraphRAG - Method: {graphrag_result['metadata']['retrieval_method']}")
            logger.info(f"   GraphRAG - Documents: {graphrag_result['metadata']['num_retrieved']}")
            logger.info(f"   GraphRAG - Time: {graphrag_result['metadata']['processing_time']:.2f}s")
            
            logger.info(f"   BasicRAG - Method: {basicrag_result['metadata']['retrieval_method']}")  
            logger.info(f"   BasicRAG - Documents: {basicrag_result['metadata']['num_retrieved']}")
            logger.info(f"   BasicRAG - Time: {basicrag_result['metadata']['processing_time']:.2f}s")
            
            # Validate that GraphRAG used knowledge graph traversal
            if graphrag_result["metadata"]["retrieval_method"] != "knowledge_graph_traversal":
                logger.error(f"❌ GraphRAG did not use knowledge graph traversal")
                return False
            
            logger.info("✅ GraphRAG vs BasicRAG comparison completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ GraphRAG vs BasicRAG comparison failed: {e}")
            return False
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        timestamp = int(time.time())
        
        report = {
            "validation_timestamp": timestamp,
            "validation_results": self.validation_results,
            "summary": {
                "total_tests": len(self.validation_results),
                "passed_tests": sum(1 for result in self.validation_results.values() if result["success"]),
                "failed_tests": sum(1 for result in self.validation_results.values() if not result["success"]),
            }
        }
        
        report["summary"]["success_rate"] = (
            report["summary"]["passed_tests"] / report["summary"]["total_tests"] * 100
            if report["summary"]["total_tests"] > 0 else 0
        )
        
        return report
    
    def run_validation(self) -> bool:
        """Run complete GraphRAG validation."""
        logger.info("🚀 Starting End-to-End GraphRAG Validation")
        
        tests = [
            ("initialize_components", "System Component Initialization"),
            ("deploy_graphrag_schema", "GraphRAG Schema Deployment"),
            ("validate_schema_deployment", "Schema Deployment Validation"),
            ("load_sample_documents", "Document Loading with Entity Extraction"),
            ("verify_knowledge_graph_population", "Knowledge Graph Population Verification"),
            ("test_knowledge_graph_traversal", "Knowledge Graph Traversal Testing"),
            ("test_fail_hard_validation", "Fail-Hard Validation Testing"),
            ("compare_graphrag_vs_basicrag", "GraphRAG vs BasicRAG Comparison")
        ]
        
        for test_method, test_name in tests:
            logger.info(f"\n{'='*60}")
            logger.info(f"🔍 {test_name}")
            logger.info(f"{'='*60}")
            
            start_time = time.time()
            try:
                success = getattr(self, test_method)()
                execution_time = time.time() - start_time
                
                self.validation_results[test_name] = {
                    "success": success,
                    "execution_time": execution_time,
                    "error": None
                }
                
                if success:
                    logger.info(f"✅ {test_name} PASSED ({execution_time:.2f}s)")
                else:
                    logger.error(f"❌ {test_name} FAILED ({execution_time:.2f}s)")
                    
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"❌ {test_name} FAILED with exception: {e}")
                
                self.validation_results[test_name] = {
                    "success": False,
                    "execution_time": execution_time,
                    "error": str(e)
                }
        
        # Generate final report
        report = self.generate_validation_report()
        
        # Save report
        report_path = f"outputs/graphrag_validation_report_{report['validation_timestamp']}.json"
        os.makedirs("outputs", exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 VALIDATION SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total Tests: {report['summary']['total_tests']}")
        logger.info(f"Passed: {report['summary']['passed_tests']}")
        logger.info(f"Failed: {report['summary']['failed_tests']}")
        logger.info(f"Success Rate: {report['summary']['success_rate']:.1f}%")
        logger.info(f"Report saved to: {report_path}")
        
        return report['summary']['failed_tests'] == 0


def main():
    """Main validation function."""
    validator = GraphRAGValidator()
    success = validator.run_validation()
    
    if success:
        logger.info("🎉 GraphRAG validation completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ GraphRAG validation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()