#!/usr/bin/env python3
"""
Comprehensive System Evaluation Script

Evaluates the current state of all RAG pipelines and system components.
"""

import sys
import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from iris_rag.config.manager import ConfigurationManager
from iris_rag.controllers.declarative_state import DeclarativeStateManager, DeclarativeStateSpec
from common.iris_connection_manager import get_iris_connection
from rag_templates import RAG
from rag_templates.standard import ConfigurableRAG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SystemEvaluator:
    """Evaluates the health and functionality of the RAG system."""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "database": {},
            "pipelines": {},
            "apis": {},
            "declarative_state": {},
            "performance": {},
            "issues": []
        }
        
    def evaluate_all(self) -> Dict[str, Any]:
        """Run complete system evaluation."""
        print("\n" + "="*80)
        print("RAG TEMPLATES SYSTEM EVALUATION")
        print("="*80 + "\n")
        
        # 1. Database connectivity
        self.check_database_connectivity()
        
        # 2. Database state
        self.check_database_state()
        
        # 3. Simple API
        self.test_simple_api()
        
        # 4. Standard API
        self.test_standard_api()
        
        # 5. Each pipeline
        self.test_all_pipelines()
        
        # 6. Declarative state management
        self.test_declarative_state()
        
        # 7. Performance check
        self.run_performance_check()
        
        # 8. Generate summary
        self.generate_summary()
        
        return self.results
    
    def check_database_connectivity(self):
        """Check IRIS database connectivity."""
        print("1. Checking Database Connectivity...")
        
        try:
            conn = get_iris_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute("SELECT CURRENT_TIMESTAMP")
                timestamp = cursor.fetchone()[0]
                cursor.close()
                conn.close()
                
                self.results["database"]["connected"] = True
                self.results["database"]["timestamp"] = str(timestamp)
                print("   ✓ Database connected successfully")
            else:
                self.results["database"]["connected"] = False
                self.results["issues"].append("Failed to connect to database")
                print("   ✗ Database connection failed")
        except Exception as e:
            self.results["database"]["connected"] = False
            self.results["database"]["error"] = str(e)
            self.results["issues"].append(f"Database error: {e}")
            print(f"   ✗ Database error: {e}")
    
    def check_database_state(self):
        """Check current database state."""
        print("\n2. Checking Database State...")
        
        try:
            conn = get_iris_connection()
            cursor = conn.cursor()
            
            # Check document count
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
            doc_count = cursor.fetchone()[0]
            
            # Check chunk count
            cursor.execute("SELECT COUNT(*) FROM RAG.DocumentChunks") 
            chunk_count = cursor.fetchone()[0]
            
            # Check token embeddings
            cursor.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings")
            token_count = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            self.results["database"]["state"] = {
                "documents": doc_count,
                "chunks": chunk_count,
                "token_embeddings": token_count
            }
            
            print(f"   Documents: {doc_count}")
            print(f"   Chunks: {chunk_count}")
            print(f"   Token Embeddings: {token_count}")
            
        except Exception as e:
            self.results["database"]["state_error"] = str(e)
            self.results["issues"].append(f"Failed to check database state: {e}")
            print(f"   ✗ Error checking state: {e}")
    
    def test_simple_api(self):
        """Test Simple API functionality."""
        print("\n3. Testing Simple API...")
        
        try:
            # Test zero-config initialization
            rag = RAG()
            self.results["apis"]["simple"] = {"initialized": True}
            
            # Test query (with minimal data)
            try:
                result = rag.query("What is machine learning?")
                self.results["apis"]["simple"]["query_works"] = True
                self.results["apis"]["simple"]["sample_response"] = result[:100] + "..."
                print("   ✓ Simple API working")
            except Exception as e:
                self.results["apis"]["simple"]["query_works"] = False
                self.results["apis"]["simple"]["query_error"] = str(e)
                print(f"   ⚠ Simple API query failed: {e}")
                
        except Exception as e:
            self.results["apis"]["simple"] = {"initialized": False, "error": str(e)}
            self.results["issues"].append(f"Simple API initialization failed: {e}")
            print(f"   ✗ Simple API failed: {e}")
    
    def test_standard_api(self):
        """Test Standard API functionality."""
        print("\n4. Testing Standard API...")
        
        try:
            # Test with configuration
            rag = ConfigurableRAG(config={"technique": "basic"})
            self.results["apis"]["standard"] = {"initialized": True}
            
            # Get available techniques
            from rag_templates.core.technique_registry import TechniqueRegistry
            registry = TechniqueRegistry()
            techniques = registry.list_techniques()
            self.results["apis"]["standard"]["available_techniques"] = techniques
            print(f"   Available techniques: {', '.join(techniques)}")
            
        except Exception as e:
            self.results["apis"]["standard"] = {"initialized": False, "error": str(e)}
            self.results["issues"].append(f"Standard API initialization failed: {e}")
            print(f"   ✗ Standard API failed: {e}")
    
    def test_all_pipelines(self):
        """Test each RAG pipeline."""
        print("\n5. Testing Individual Pipelines...")
        
        pipelines = ["basic", "colbert", "hyde", "crag", "graphrag", "noderag", "hybrid_ifind"]
        
        for pipeline in pipelines:
            print(f"\n   Testing {pipeline}...")
            
            try:
                # Try to create pipeline
                rag = ConfigurableRAG(config={"technique": pipeline})
                
                # Check if pipeline can be initialized
                self.results["pipelines"][pipeline] = {
                    "status": "initialized",
                    "error": None
                }
                
                # Try a simple operation
                try:
                    # Just check if we can access the pipeline
                    if hasattr(rag, 'pipeline') and rag.pipeline:
                        self.results["pipelines"][pipeline]["ready"] = True
                        print(f"      ✓ {pipeline} pipeline ready")
                    else:
                        self.results["pipelines"][pipeline]["ready"] = False
                        print(f"      ⚠ {pipeline} pipeline not fully ready")
                except Exception as e:
                    self.results["pipelines"][pipeline]["operational_error"] = str(e)
                    print(f"      ⚠ {pipeline} operational check failed: {e}")
                    
            except Exception as e:
                self.results["pipelines"][pipeline] = {
                    "status": "failed",
                    "error": str(e)
                }
                self.results["issues"].append(f"{pipeline} pipeline failed: {e}")
                print(f"      ✗ {pipeline} failed: {e}")
    
    def test_declarative_state(self):
        """Test declarative state management."""
        print("\n6. Testing Declarative State Management...")
        
        try:
            # Create manager
            manager = DeclarativeStateManager()
            
            # Create test spec
            spec = DeclarativeStateSpec(
                document_count=10,
                pipeline_type="basic",
                validation_mode="lenient"
            )
            
            # Declare state
            manager.declare_state(spec)
            
            # Get drift report
            drift = manager.get_drift_report()
            
            self.results["declarative_state"] = {
                "functional": True,
                "has_drift": drift["has_drift"],
                "drift_summary": drift.get("summary", "No summary")
            }
            
            print("   ✓ Declarative state management functional")
            print(f"   Drift detected: {drift['has_drift']}")
            
        except Exception as e:
            self.results["declarative_state"] = {
                "functional": False,
                "error": str(e)
            }
            self.results["issues"].append(f"Declarative state failed: {e}")
            print(f"   ✗ Declarative state failed: {e}")
    
    def run_performance_check(self):
        """Run basic performance check."""
        print("\n7. Running Performance Check...")
        
        try:
            rag = RAG()
            
            # Time a simple query
            start = time.time()
            result = rag.query("test query")
            duration = time.time() - start
            
            self.results["performance"]["simple_query_time"] = duration
            print(f"   Simple query time: {duration:.3f}s")
            
            # Check if caching is enabled
            config = ConfigurationManager()
            cache_config = config.get("llm_cache", {})
            self.results["performance"]["cache_enabled"] = cache_config.get("enabled", False)
            print(f"   LLM cache enabled: {cache_config.get('enabled', False)}")
            
        except Exception as e:
            self.results["performance"]["error"] = str(e)
            print(f"   ✗ Performance check failed: {e}")
    
    def generate_summary(self):
        """Generate evaluation summary."""
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        # Overall health
        total_issues = len(self.results["issues"])
        
        if total_issues == 0:
            health = "EXCELLENT"
        elif total_issues <= 2:
            health = "GOOD"
        elif total_issues <= 5:
            health = "FAIR"
        else:
            health = "NEEDS ATTENTION"
        
        self.results["summary"] = {
            "overall_health": health,
            "total_issues": total_issues,
            "working_pipelines": sum(1 for p in self.results.get("pipelines", {}).values() 
                                   if p.get("status") == "initialized"),
            "database_connected": self.results.get("database", {}).get("connected", False)
        }
        
        print(f"\nOverall System Health: {health}")
        print(f"Total Issues Found: {total_issues}")
        
        if self.results["issues"]:
            print("\nKey Issues:")
            for i, issue in enumerate(self.results["issues"][:5], 1):
                print(f"  {i}. {issue}")
        
        # Pipeline status
        if self.results.get("pipelines"):
            print("\nPipeline Status:")
            for pipeline, status in self.results["pipelines"].items():
                status_icon = "✓" if status["status"] == "initialized" else "✗"
                print(f"  {status_icon} {pipeline}")
        
        # Recommendations
        print("\nRecommendations:")
        
        if not self.results.get("database", {}).get("connected"):
            print("  1. Ensure IRIS database container is running")
        
        if self.results.get("database", {}).get("state", {}).get("documents", 0) < 100:
            print("  2. Load more test documents for better evaluation")
        
        failed_pipelines = [p for p, s in self.results.get("pipelines", {}).items() 
                           if s.get("status") == "failed"]
        if failed_pipelines:
            print(f"  3. Fix failing pipelines: {', '.join(failed_pipelines)}")


def main():
    """Run system evaluation."""
    evaluator = SystemEvaluator()
    results = evaluator.evaluate_all()
    
    # Save results
    output_file = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nFull results saved to: {output_file}")
    
    # Return exit code based on health
    if results["summary"]["overall_health"] in ["EXCELLENT", "GOOD"]:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())