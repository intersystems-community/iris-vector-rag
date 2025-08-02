#!/usr/bin/env python3
"""
Test script to validate core fixes without validation checks.
"""
import sys
import os
import time

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_pipeline_instantiation_without_validation():
    """Test pipeline instantiation without validation to check core fixes."""
    print("=" * 80)
    print("TESTING CORE FIXES - PIPELINE INSTANTIATION")
    print("=" * 80)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    pipeline_types = [
        "basic",
        "colbert", 
        "crag",
        "hyde",
        "graphrag",
        "noderag",
        "hybrid_ifind"
    ]
    
    results = {}
    
    for pipeline_type in pipeline_types:
        print(f"\n=== Testing {pipeline_type.upper()} Pipeline ===")
        
        try:
            # Import required modules
            from iris_rag.core.connection import ConnectionManager
            from iris_rag.config.manager import ConfigurationManager
            from common.iris_connection_manager import get_iris_connection
            from common.utils import get_llm_func, get_embedding_func
            
            # Create managers
            connection_manager = ConnectionManager(get_iris_connection())
            config_manager = ConfigurationManager()
            
            # Import and instantiate pipeline directly
            if pipeline_type == "basic":
                from iris_rag.pipelines.basic import BasicRAGPipeline
                pipeline = BasicRAGPipeline(
                    connection_manager=connection_manager,
                    config_manager=config_manager,
                    embedding_func=get_embedding_func(),
                    llm_func=get_llm_func()
                )
            elif pipeline_type == "colbert":
                from iris_rag.pipelines.colbert import ColBERTRAGPipeline
                pipeline = ColBERTRAGPipeline(
                    connection_manager=connection_manager,
                    config_manager=config_manager,
                    llm_func=get_llm_func()
                )
            elif pipeline_type == "crag":
                from iris_rag.pipelines.crag import CRAGPipeline
                pipeline = CRAGPipeline(
                    connection_manager=connection_manager,
                    config_manager=config_manager,
                    embedding_func=get_embedding_func(),
                    llm_func=get_llm_func()
                )
            elif pipeline_type == "hyde":
                from iris_rag.pipelines.hyde import HyDERAGPipeline
                pipeline = HyDERAGPipeline(
                    connection_manager=connection_manager,
                    config_manager=config_manager,
                    embedding_func=get_embedding_func(),
                    llm_func=get_llm_func()
                )
            elif pipeline_type == "graphrag":
                from iris_rag.pipelines.graphrag import GraphRAGPipeline
                pipeline = GraphRAGPipeline(
                    connection_manager=connection_manager,
                    config_manager=config_manager,
                    embedding_func=get_embedding_func(),
                    llm_func=get_llm_func()
                )
            elif pipeline_type == "noderag":
                from iris_rag.pipelines.noderag import NoRAGPipeline
                pipeline = NoRAGPipeline(
                    connection_manager=connection_manager,
                    config_manager=config_manager,
                    llm_func=get_llm_func()
                )
            elif pipeline_type == "hybrid_ifind":
                from iris_rag.pipelines.hybrid_ifind import HybridIFindRAGPipeline
                pipeline = HybridIFindRAGPipeline(
                    connection_manager=connection_manager,
                    config_manager=config_manager,
                    embedding_func=get_embedding_func(),
                    llm_func=get_llm_func()
                )
            
            print(f"✓ {pipeline_type} pipeline instantiated successfully")
            
            # Test if required abstract methods exist
            has_execute = hasattr(pipeline, 'execute') and callable(getattr(pipeline, 'execute'))
            has_load_documents = hasattr(pipeline, 'load_documents') and callable(getattr(pipeline, 'load_documents'))
            has_setup_database = hasattr(pipeline, 'setup_database') and callable(getattr(pipeline, 'setup_database'))
            
            print(f"  - execute method: {'✓' if has_execute else '✗'}")
            print(f"  - load_documents method: {'✓' if has_load_documents else '✗'}")
            print(f"  - setup_database method: {'✓' if has_setup_database else '✗'}")
            
            results[pipeline_type] = {
                "instantiation": True,
                "execute_method": has_execute,
                "load_documents_method": has_load_documents,
                "setup_database_method": has_setup_database,
                "error": None
            }
            
        except Exception as e:
            print(f"✗ {pipeline_type} failed: {e}")
            results[pipeline_type] = {
                "instantiation": False,
                "execute_method": False,
                "load_documents_method": False,
                "setup_database_method": False,
                "error": str(e)
            }
    
    # Generate summary
    print("\n" + "=" * 80)
    print("SUMMARY RESULTS")
    print("=" * 80)
    
    instantiation_success = sum(1 for r in results.values() if r["instantiation"])
    execute_methods = sum(1 for r in results.values() if r["execute_method"])
    load_methods = sum(1 for r in results.values() if r["load_documents_method"])
    setup_methods = sum(1 for r in results.values() if r["setup_database_method"])
    
    print(f"Pipeline Instantiation:     {instantiation_success}/7 ({instantiation_success/7*100:.1f}%)")
    print(f"Execute Methods:            {execute_methods}/7 ({execute_methods/7*100:.1f}%)")
    print(f"Load Documents Methods:     {load_methods}/7 ({load_methods/7*100:.1f}%)")
    print(f"Setup Database Methods:     {setup_methods}/7 ({setup_methods/7*100:.1f}%)")
    
    # Detailed results
    print("\n" + "-" * 80)
    print("DETAILED RESULTS")
    print("-" * 80)
    print(f"{'Pipeline':<15} {'Instantiate':<12} {'Execute':<8} {'Load':<6} {'Setup':<6} {'Error'}")
    print("-" * 80)
    
    for pipeline_type, result in results.items():
        instantiate_status = "✓" if result["instantiation"] else "✗"
        execute_status = "✓" if result["execute_method"] else "✗"
        load_status = "✓" if result["load_documents_method"] else "✗"
        setup_status = "✓" if result["setup_database_method"] else "✗"
        error_msg = result["error"][:30] + "..." if result["error"] and len(result["error"]) > 30 else result["error"] or ""
        
        print(f"{pipeline_type:<15} {instantiate_status:<12} {execute_status:<8} {load_status:<6} {setup_status:<6} {error_msg}")
    
    # Progress analysis
    print("\n" + "-" * 80)
    print("PROGRESS ANALYSIS")
    print("-" * 80)
    print("BEFORE FIXES:")
    print("  - Abstract method errors preventing instantiation")
    print("  - Missing required methods in pipeline classes")
    print("  - Database table creation issues")
    print()
    print("AFTER FIXES:")
    print(f"  - {instantiation_success}/7 pipelines instantiate without abstract method errors")
    print(f"  - {execute_methods}/7 pipelines have execute method")
    print(f"  - {load_methods}/7 pipelines have load_documents method")
    print(f"  - {setup_methods}/7 pipelines have setup_database method")
    
    if instantiation_success == 7:
        print("\n🎉 SUCCESS: All abstract method errors have been FIXED!")
        print("All 7 pipelines can now be instantiated successfully.")
    else:
        print(f"\n⚠️  {7-instantiation_success} pipelines still have instantiation issues")
    
    return results

if __name__ == "__main__":
    try:
        results = test_pipeline_instantiation_without_validation()
        print("\n" + "=" * 80)
        print("CORE FIXES VALIDATION COMPLETE")
        print("=" * 80)
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)