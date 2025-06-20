#!/usr/bin/env python3
"""
Examples of using declarative state management for different pipeline setups.

This shows how to configure the system for different development scenarios:
1. Lightweight HyDE-only setup
2. Full ColBERT setup with token embeddings
3. Production setup with all pipelines
"""

from iris_rag.controllers.declarative_state import DeclarativeStateSpec, DeclarativeStateManager
from iris_rag.config.manager import ConfigurationManager


def lightweight_hyde_setup():
    """Example: Lightweight dev setup with just HyDE pipeline."""
    print("=== Lightweight HyDE Setup ===")
    
    # Define desired state for HyDE-only development
    state_spec = DeclarativeStateSpec(
        document_count=100,  # Just 100 docs for quick dev
        pipeline_type="hyde",
        embedding_model="all-MiniLM-L6-v2",
        embedding_dimension=384,
        # No token embeddings needed for HyDE
        force_regenerate=False
    )
    
    # Initialize manager
    config_manager = ConfigurationManager()
    state_manager = DeclarativeStateManager(config_manager)
    
    # Apply the state
    print(f"Applying state for {state_spec.pipeline_type} pipeline...")
    result = state_manager.sync_to_state(state_spec)
    
    if result.success:
        print("✅ HyDE setup complete!")
        print(f"   Documents: {result.document_stats.get('total_documents', 0)}")
        print(f"   Embeddings: {result.document_stats.get('documents_with_embeddings', 0)}")
        print("   Token embeddings: Not required")
    else:
        print(f"❌ Setup failed: {result.drift_analysis}")


def full_colbert_setup():
    """Example: Full ColBERT setup with token embeddings."""
    print("\n=== Full ColBERT Setup ===")
    
    # Define desired state for ColBERT
    state_spec = DeclarativeStateSpec(
        document_count=1000,  # More docs for better results
        pipeline_type="colbert",
        embedding_model="all-MiniLM-L6-v2",
        embedding_dimension=384,
        # Token embeddings automatically required for ColBERT
        force_regenerate=False
    )
    
    # Initialize manager
    config_manager = ConfigurationManager()
    state_manager = DeclarativeStateManager(config_manager)
    
    # Apply the state
    print(f"Applying state for {state_spec.pipeline_type} pipeline...")
    result = state_manager.sync_to_state(state_spec)
    
    if result.success:
        print("✅ ColBERT setup complete!")
        print(f"   Documents: {result.document_stats.get('total_documents', 0)}")
        print(f"   Embeddings: {result.document_stats.get('documents_with_embeddings', 0)}")
        print(f"   Token embeddings: {result.document_stats.get('token_embeddings_count', 0)}")
    else:
        print(f"❌ Setup failed: {result.drift_analysis}")


def production_multi_pipeline_setup():
    """Example: Production setup supporting multiple pipelines."""
    print("\n=== Production Multi-Pipeline Setup ===")
    
    # For production, we might want to support all pipelines
    # This means we need the superset of all requirements
    state_spec = DeclarativeStateSpec(
        document_count=5000,  # Full dataset
        pipeline_type="all",  # Special value to indicate all pipelines
        embedding_model="all-MiniLM-L6-v2",
        embedding_dimension=384,
        force_regenerate=False,
        # Higher quality requirements for production
        min_embedding_diversity=0.2,
        max_contamination_ratio=0.01,
        validation_mode="strict"
    )
    
    # Note: When pipeline_type="all", the system should:
    # 1. Generate document embeddings (needed by all)
    # 2. Generate token embeddings (needed by ColBERT)
    # 3. Create chunked documents (needed by CRAG)
    # 4. Extract entities (needed by GraphRAG)
    
    print("This would set up the system for all pipelines...")
    print("Including:")
    print("- Document embeddings (all pipelines)")
    print("- Token embeddings (ColBERT)")
    print("- Chunked documents (CRAG)")
    print("- Entity extraction (GraphRAG)")


def check_current_state():
    """Check the current state of the system."""
    print("\n=== Current System State ===")
    
    config_manager = ConfigurationManager()
    state_manager = DeclarativeStateManager(config_manager)
    
    current_state = state_manager.get_current_state()
    print(f"Documents: {current_state.document_stats.get('total_documents', 0)}")
    print(f"With embeddings: {current_state.document_stats.get('documents_with_embeddings', 0)}")
    print(f"Token embeddings: {current_state.document_stats.get('token_embeddings_count', 0)}")
    print(f"Current issues: {len(current_state.quality_issues.issues) if current_state.quality_issues else 0}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "hyde":
            lightweight_hyde_setup()
        elif sys.argv[1] == "colbert":
            full_colbert_setup()
        elif sys.argv[1] == "production":
            production_multi_pipeline_setup()
        elif sys.argv[1] == "check":
            check_current_state()
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Usage: python declarative_state_examples.py [hyde|colbert|production|check]")
    else:
        # Show all examples
        lightweight_hyde_setup()
        full_colbert_setup()
        production_multi_pipeline_setup()
        check_current_state()