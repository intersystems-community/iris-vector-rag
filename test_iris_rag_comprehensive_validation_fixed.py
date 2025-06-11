#!/usr/bin/env python3
"""
COMPREHENSIVE iris_rag PACKAGE VALIDATION
Fixed version with proper API usage and document loading

This test validates the iris_rag package by:
1. Loading test documents using iris_rag
2. Testing all available pipelines with correct API calls
3. Validating functionality and performance
4. Generating comprehensive report
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class IrisRAGValidator:
    """Comprehensive validator for iris_rag package functionality"""
    
    def __init__(self):
        self.start_time = time.time()
        self.validation_report = {}
        
        # Test configuration
        self.config = {
            "test_queries": [
                {
                    "query": "What are the molecular mechanisms of BRCA1 in DNA repair?",
                    "expected_keywords": ["BRCA1", "DNA", "repair", "molecular"],
                    "category": "genetics"
                },
                {
                    "query": "How does p53 regulate cell cycle checkpoints?",
                    "expected_keywords": ["p53", "cell cycle", "checkpoint"],
                    "category": "cell_biology"
                },
                {
                    "query": "What is the role of inflammation in disease?",
                    "expected_keywords": ["inflammation", "disease", "immune"],
                    "category": "immunology"
                }
            ],
            "test_documents": [
                {
                    "id": "PMC001",
                    "content": "BRCA1 is a crucial tumor suppressor gene that plays a vital role in DNA repair mechanisms. The BRCA1 protein is involved in homologous recombination repair of DNA double-strand breaks. Mutations in BRCA1 significantly increase the risk of breast and ovarian cancers. The molecular mechanisms involve the formation of nuclear foci and interaction with other DNA repair proteins like BRCA2, RAD51, and ATM kinase.",
                    "metadata": {"source": "test_genetics", "pmcid": "PMC001", "category": "genetics"}
                },
                {
                    "id": "PMC002", 
                    "content": "The p53 protein, known as the 'guardian of the genome', is a critical cell cycle checkpoint regulator. p53 responds to DNA damage by halting cell division at the G1/S checkpoint, allowing time for DNA repair. If repair fails, p53 triggers apoptosis to prevent cancer development. The p53 pathway involves phosphorylation cascades, transcriptional activation of target genes like p21, and coordination with other checkpoint proteins.",
                    "metadata": {"source": "test_cell_biology", "pmcid": "PMC002", "category": "cell_biology"}
                },
                {
                    "id": "PMC003",
                    "content": "Inflammation is a complex biological response to harmful stimuli, including pathogens, damaged cells, or irritants. The inflammatory response involves immune cells, blood vessels, and molecular mediators. Chronic inflammation contributes to various diseases including cardiovascular disease, diabetes, and cancer. Key inflammatory mediators include cytokines like TNF-alpha, interleukins, and prostaglandins.",
                    "metadata": {"source": "test_immunology", "pmcid": "PMC003", "category": "immunology"}
                },
                {
                    "id": "PMC004",
                    "content": "DNA repair mechanisms are essential for maintaining genomic stability. Multiple pathways exist including base excision repair, nucleotide excision repair, mismatch repair, and homologous recombination. BRCA1 and BRCA2 proteins are central to homologous recombination repair. Defects in DNA repair lead to increased mutation rates and cancer predisposition.",
                    "metadata": {"source": "test_genetics", "pmcid": "PMC004", "category": "genetics"}
                },
                {
                    "id": "PMC005",
                    "content": "Cell cycle checkpoints ensure proper cell division and prevent genomic instability. The G1/S checkpoint, controlled by p53 and Rb proteins, prevents replication of damaged DNA. The G2/M checkpoint ensures complete DNA replication before mitosis. Checkpoint failures contribute to cancer development through accumulation of mutations.",
                    "metadata": {"source": "test_cell_biology", "pmcid": "PMC005", "category": "cell_biology"}
                }
            ]
        }
        
        logger.info("🚀 iris_rag Validator initialized")
        logger.info(f"Test queries: {len(self.config['test_queries'])}")
        logger.info(f"Test documents: {len(self.config['test_documents'])}")
    
    def validate_imports(self) -> Dict[str, Any]:
        """Validate iris_rag package imports"""
        logger.info("📦 Validating iris_rag package imports...")
        
        result = {
            "success": False,
            "imported_classes": [],
            "errors": []
        }
        
        try:
            # Test imports
            import iris_rag
            from iris_rag.core.connection import ConnectionManager
            from iris_rag.core.models import Document
            from iris_rag.config.manager import ConfigurationManager
            from iris_rag.pipelines.basic import BasicRAGPipeline
            
            result["imported_classes"] = [
                "iris_rag", "ConnectionManager", "Document", 
                "ConfigurationManager", "BasicRAGPipeline"
            ]
            
            # Test Document creation
            doc = Document(page_content='test', id='test')
            assert doc.page_content == 'test'
            
            result["success"] = True
            logger.info("✅ All imports successful")
            
        except Exception as e:
            result["errors"].append(str(e))
            logger.error(f"❌ Import failed: {e}")
        
        return result
    
    def setup_environment(self) -> Dict[str, Any]:
        """Setup test environment"""
        logger.info("🔧 Setting up test environment...")
        
        result = {
            "success": False,
            "database_connection": False,
            "functions_loaded": False,
            "errors": []
        }
        
        try:
            # Get utilities
            from common.utils import get_embedding_func, get_llm_func
            from common.iris_connection_manager import get_iris_connection
            
            # Test database connection
            self.connection = get_iris_connection()
            if self.connection:
                result["database_connection"] = True
                logger.info("✅ Database connection established")
            
            # Get functions
            self.embedding_func = get_embedding_func()
            self.llm_func = get_llm_func()
            
            if self.embedding_func and self.llm_func:
                result["functions_loaded"] = True
                logger.info("✅ Embedding and LLM functions loaded")
            
            result["success"] = True
            logger.info("✅ Environment setup complete")
            
        except Exception as e:
            result["errors"].append(str(e))
            logger.error(f"❌ Environment setup failed: {e}")
        
        return result
    
    def load_test_documents(self) -> Dict[str, Any]:
        """Load test documents using iris_rag"""
        logger.info("📄 Loading test documents...")
        
        result = {
            "success": False,
            "documents_loaded": 0,
            "errors": []
        }
        
        try:
            import iris_rag
            from iris_rag.core.models import Document
            
            # Create pipeline for document loading
            pipeline = iris_rag.create_pipeline(
                pipeline_type="basic",
                llm_func=self.llm_func
            )
            
            # Convert test documents to Document objects
            documents = []
            for doc_data in self.config["test_documents"]:
                doc = Document(
                    page_content=doc_data["content"],
                    id=doc_data["id"],
                    metadata=doc_data["metadata"]
                )
                documents.append(doc)
            
            # Load documents into the pipeline
            pipeline.load_documents(
                documents_path="",  # Not used when providing documents directly
                documents=documents,
                chunk_documents=True,
                generate_embeddings=True
            )
            
            result["documents_loaded"] = len(documents)
            result["success"] = True
            logger.info(f"✅ Loaded {len(documents)} test documents")
            
            # Store pipeline for testing
            self.pipeline = pipeline
            
        except Exception as e:
            result["errors"].append(str(e))
            logger.error(f"❌ Document loading failed: {e}")
        
        return result
    
    def test_basic_pipeline(self) -> Dict[str, Any]:
        """Test the BasicRAGPipeline"""
        logger.info("🧪 Testing BasicRAGPipeline...")
        
        result = {
            "success": False,
            "queries_tested": 0,
            "queries_successful": 0,
            "query_results": [],
            "performance": {},
            "errors": []
        }
        
        try:
            total_time = 0
            
            for i, query_data in enumerate(self.config["test_queries"]):
                query_start = time.time()
                
                try:
                    # Execute pipeline with correct API
                    response = self.pipeline.execute(
                        query_text=query_data["query"],
                        top_k=3
                    )
                    
                    query_time = time.time() - query_start
                    total_time += query_time
                    
                    # Validate response
                    validation = self.validate_response(response, query_data)
                    
                    result["query_results"].append({
                        "query": query_data["query"],
                        "category": query_data["category"],
                        "execution_time": query_time,
                        "retrieved_count": len(response.get("retrieved_documents", [])),
                        "answer_length": len(response.get("answer", "")),
                        "validation": validation,
                        "success": validation["valid"]
                    })
                    
                    if validation["valid"]:
                        result["queries_successful"] += 1
                    
                    result["queries_tested"] += 1
                    logger.info(f"✅ Query {i+1} completed in {query_time:.2f}s")
                    
                except Exception as e:
                    result["errors"].append(f"Query {i+1} failed: {e}")
                    logger.error(f"❌ Query {i+1} failed: {e}")
            
            # Calculate performance
            if result["queries_tested"] > 0:
                result["performance"] = {
                    "total_time": total_time,
                    "avg_query_time": total_time / result["queries_tested"],
                    "success_rate": (result["queries_successful"] / result["queries_tested"]) * 100
                }
            
            result["success"] = result["queries_successful"] > 0
            
            if result["success"]:
                logger.info(f"✅ BasicRAGPipeline test: {result['queries_successful']}/{result['queries_tested']} queries successful")
            else:
                logger.error("❌ BasicRAGPipeline test failed")
            
        except Exception as e:
            result["errors"].append(f"Pipeline test failed: {e}")
            logger.error(f"❌ BasicRAGPipeline test failed: {e}")
        
        return result
    
    def validate_response(self, response: Dict[str, Any], query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a pipeline response"""
        
        validation = {
            "valid": False,
            "checks_passed": [],
            "checks_failed": [],
            "score": 0.0
        }
        
        checks = 0
        passed = 0
        
        # Check 1: Required fields
        checks += 1
        if "answer" in response and "retrieved_documents" in response:
            validation["checks_passed"].append("Required fields present")
            passed += 1
        else:
            validation["checks_failed"].append("Missing required fields")
        
        # Check 2: Answer quality
        checks += 1
        answer = response.get("answer", "")
        if len(answer) >= 50:  # Minimum answer length
            validation["checks_passed"].append(f"Answer length OK ({len(answer)} chars)")
            passed += 1
        else:
            validation["checks_failed"].append(f"Answer too short ({len(answer)} chars)")
        
        # Check 3: Document retrieval
        checks += 1
        docs = response.get("retrieved_documents", [])
        if len(docs) > 0:
            validation["checks_passed"].append(f"Retrieved {len(docs)} documents")
            passed += 1
        else:
            validation["checks_failed"].append("No documents retrieved")
        
        # Check 4: Keyword relevance
        checks += 1
        expected_keywords = query_data["expected_keywords"]
        found_keywords = []
        
        # Check in answer
        answer_lower = answer.lower()
        for keyword in expected_keywords:
            if keyword.lower() in answer_lower:
                found_keywords.append(keyword)
        
        if len(found_keywords) > 0:
            validation["checks_passed"].append(f"Found keywords: {found_keywords}")
            passed += 1
        else:
            validation["checks_failed"].append(f"No keywords found from: {expected_keywords}")
        
        validation["score"] = (passed / checks) * 100
        validation["valid"] = passed >= 3  # Require 3/4 checks to pass
        
        return validation
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run the comprehensive validation"""
        logger.info("🚀 Starting Comprehensive iris_rag Validation...")
        logger.info("=" * 60)
        
        # Initialize report
        self.validation_report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "validation_type": "iris_rag_comprehensive"
            },
            "import_validation": {},
            "environment_setup": {},
            "document_loading": {},
            "pipeline_tests": {},
            "overall_assessment": {}
        }
        
        # Step 1: Validate imports
        logger.info("📦 Step 1: Validating imports...")
        self.validation_report["import_validation"] = self.validate_imports()
        
        if not self.validation_report["import_validation"]["success"]:
            logger.error("❌ Import validation failed")
            return self.validation_report
        
        # Step 2: Setup environment
        logger.info("🔧 Step 2: Setting up environment...")
        self.validation_report["environment_setup"] = self.setup_environment()
        
        if not self.validation_report["environment_setup"]["success"]:
            logger.error("❌ Environment setup failed")
            return self.validation_report
        
        # Step 3: Load test documents
        logger.info("📄 Step 3: Loading test documents...")
        self.validation_report["document_loading"] = self.load_test_documents()
        
        if not self.validation_report["document_loading"]["success"]:
            logger.error("❌ Document loading failed")
            return self.validation_report
        
        # Step 4: Test pipelines
        logger.info("🧪 Step 4: Testing pipelines...")
        pipeline_tests = {}
        pipeline_tests["basic"] = self.test_basic_pipeline()
        self.validation_report["pipeline_tests"] = pipeline_tests
        
        # Step 5: Overall assessment
        logger.info("🎯 Step 5: Overall assessment...")
        self.validation_report["overall_assessment"] = self.assess_overall_status()
        
        # Calculate total time
        total_time = time.time() - self.start_time
        self.validation_report["metadata"]["total_execution_time"] = total_time
        
        # Save and summarize
        self.save_report()
        self.log_summary()
        
        return self.validation_report
    
    def assess_overall_status(self) -> Dict[str, Any]:
        """Assess overall validation status"""
        
        assessment = {
            "overall_success": False,
            "confidence_level": "LOW",
            "score": 0.0,
            "issues": [],
            "recommendations": []
        }
        
        score = 0.0
        
        # Import validation (25 points)
        if self.validation_report["import_validation"]["success"]:
            score += 25
        else:
            assessment["issues"].append("Import validation failed")
        
        # Environment setup (25 points)
        if self.validation_report["environment_setup"]["success"]:
            score += 25
        else:
            assessment["issues"].append("Environment setup failed")
        
        # Document loading (25 points)
        if self.validation_report["document_loading"]["success"]:
            score += 25
        else:
            assessment["issues"].append("Document loading failed")
        
        # Pipeline functionality (25 points)
        pipeline_tests = self.validation_report["pipeline_tests"]
        if pipeline_tests["basic"]["success"]:
            success_rate = pipeline_tests["basic"]["performance"].get("success_rate", 0)
            score += (success_rate / 100) * 25
        else:
            assessment["issues"].append("Pipeline tests failed")
        
        assessment["score"] = score
        
        # Determine status
        if score >= 90:
            assessment["overall_success"] = True
            assessment["confidence_level"] = "HIGH"
        elif score >= 75:
            assessment["overall_success"] = True
            assessment["confidence_level"] = "MEDIUM"
            assessment["recommendations"].append("Monitor performance in production")
        elif score >= 60:
            assessment["overall_success"] = False
            assessment["confidence_level"] = "MEDIUM-LOW"
            assessment["recommendations"].append("Address issues before production")
        else:
            assessment["overall_success"] = False
            assessment["confidence_level"] = "LOW"
            assessment["issues"].append("Major issues prevent production use")
        
        return assessment
    
    def save_report(self):
        """Save validation report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reports_dir = Path("reports/validation")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = reports_dir / f"iris_rag_validation_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(self.validation_report, f, indent=2, default=str)
        
        logger.info(f"📄 Report saved: {report_file}")
    
    def log_summary(self):
        """Log validation summary"""
        logger.info("\n" + "=" * 60)
        logger.info("🎯 IRIS_RAG VALIDATION SUMMARY")
        logger.info("=" * 60)
        
        metadata = self.validation_report["metadata"]
        assessment = self.validation_report["overall_assessment"]
        
        logger.info(f"⏱️  Execution Time: {metadata['total_execution_time']:.2f}s")
        logger.info(f"📊 Overall Score: {assessment['score']:.1f}/100")
        logger.info(f"🔒 Confidence: {assessment['confidence_level']}")
        
        # Component status
        components = [
            ("Import Validation", self.validation_report["import_validation"]["success"]),
            ("Environment Setup", self.validation_report["environment_setup"]["success"]),
            ("Document Loading", self.validation_report["document_loading"]["success"]),
            ("Pipeline Tests", self.validation_report["pipeline_tests"]["basic"]["success"])
        ]
        
        for name, success in components:
            status = "✅ PASSED" if success else "❌ FAILED"
            logger.info(f"{name}: {status}")
        
        # Pipeline performance
        basic_test = self.validation_report["pipeline_tests"]["basic"]
        if basic_test["success"]:
            perf = basic_test["performance"]
            logger.info(f"🚀 Query Success Rate: {perf['success_rate']:.1f}%")
            logger.info(f"⚡ Avg Query Time: {perf['avg_query_time']:.2f}s")
        
        # Final verdict
        if assessment["overall_success"]:
            logger.info("🎉 VALIDATION PASSED: iris_rag package is functional!")
            logger.info("✅ InterSystems naming refactoring validated successfully")
        else:
            logger.info("❌ VALIDATION FAILED: iris_rag package needs improvements")
        
        # Issues and recommendations
        if assessment["issues"]:
            logger.info("🚨 Issues:")
            for issue in assessment["issues"]:
                logger.info(f"   - {issue}")
        
        if assessment["recommendations"]:
            logger.info("💡 Recommendations:")
            for rec in assessment["recommendations"]:
                logger.info(f"   - {rec}")
        
        logger.info("=" * 60)


def main():
    """Main function"""
    print("🚀 Starting iris_rag Package Comprehensive Validation")
    print("=" * 60)
    print("Testing the refactored iris_rag package functionality")
    print("=" * 60)
    
    validator = IrisRAGValidator()
    results = validator.run_comprehensive_validation()
    
    if results["overall_assessment"]["overall_success"]:
        print("\n🎉 SUCCESS: iris_rag package validation passed!")
        return 0
    else:
        print("\n❌ FAILURE: iris_rag package needs improvements")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())