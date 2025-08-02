"""
Standard API Demo for RAG Templates Library Consumption Framework.

This demo showcases the advanced Standard API capabilities including:
- Technique selection and configuration
- Advanced query options
- Complex configuration management
- Backward compatibility with Simple API
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from rag_templates.standard import ConfigurableRAG
from rag_templates.simple import RAG


def demo_basic_technique_selection():
    """Demonstrate basic technique selection."""
    print("=== Basic Technique Selection ===")
    
    # Basic technique selection
    basic_rag = ConfigurableRAG({"technique": "basic"})
    print(f"Created RAG with technique: {basic_rag._technique}")
    
    # ColBERT technique
    colbert_rag = ConfigurableRAG({"technique": "colbert"})
    print(f"Created RAG with technique: {colbert_rag._technique}")
    
    # HyDE technique
    hyde_rag = ConfigurableRAG({"technique": "hyde"})
    print(f"Created RAG with technique: {hyde_rag._technique}")
    
    print()


def demo_advanced_configuration():
    """Demonstrate advanced configuration capabilities."""
    print("=== Advanced Configuration ===")
    
    # Complex configuration
    advanced_config = {
        "technique": "colbert",
        "llm_provider": "anthropic",
        "llm_config": {
            "model": "claude-3-sonnet",
            "temperature": 0.1,
            "max_tokens": 2000
        },
        "embedding_model": "text-embedding-3-large",
        "embedding_config": {
            "dimension": 3072,
            "batch_size": 16
        },
        "technique_config": {
            "max_query_length": 512,
            "doc_maxlen": 180,
            "top_k": 15
        },
        "vector_index": {
            "type": "HNSW",
            "M": 32,
            "efConstruction": 400
        }
    }
    
    rag = ConfigurableRAG(advanced_config)
    print(f"Created advanced RAG with technique: {rag._technique}")
    print(f"LLM config: {rag.get_config('llm_config')}")
    print(f"Technique config: {rag.get_config('technique_config')}")
    print()


def demo_technique_registry():
    """Demonstrate technique registry capabilities."""
    print("=== Technique Registry ===")
    
    rag = ConfigurableRAG({"technique": "basic"})
    
    # List available techniques
    techniques = rag.get_available_techniques()
    print(f"Available techniques: {techniques}")
    
    # Get technique information
    basic_info = rag.get_technique_info("basic")
    print(f"Basic technique info: {basic_info}")
    
    colbert_info = rag.get_technique_info("colbert")
    print(f"ColBERT technique info: {colbert_info}")
    print()


def demo_technique_switching():
    """Demonstrate dynamic technique switching."""
    print("=== Technique Switching ===")
    
    # Start with basic technique
    rag = ConfigurableRAG({"technique": "basic"})
    print(f"Initial technique: {rag._technique}")
    
    # Switch to ColBERT
    rag.switch_technique("colbert", {
        "max_query_length": 256,
        "top_k": 10
    })
    print(f"Switched to technique: {rag._technique}")
    
    # Switch to HyDE
    rag.switch_technique("hyde")
    print(f"Switched to technique: {rag._technique}")
    print()


def demo_backward_compatibility():
    """Demonstrate backward compatibility with Simple API."""
    print("=== Backward Compatibility ===")
    
    # Simple API still works
    simple_rag = RAG()
    print(f"Simple API: {simple_rag}")
    
    # Standard API works alongside
    standard_rag = ConfigurableRAG({"technique": "basic"})
    print(f"Standard API: {standard_rag}")
    
    # Both are independent
    print(f"Different types: {type(simple_rag)} vs {type(standard_rag)}")
    print()


def demo_configuration_inheritance():
    """Demonstrate configuration inheritance and overrides."""
    print("=== Configuration Inheritance ===")
    
    # Base configuration
    base_config = {
        "technique": "basic",
        "max_results": 5,
        "chunk_size": 1000
    }
    
    rag = ConfigurableRAG(base_config)
    print(f"Base max_results: {rag.get_config('max_results')}")
    print(f"Base chunk_size: {rag.get_config('chunk_size')}")
    
    # Override with technique-specific config
    override_config = {
        "technique": "colbert",
        "max_results": 15,
        "technique_config": {
            "max_query_length": 512,
            "doc_maxlen": 180
        }
    }
    
    rag2 = ConfigurableRAG(override_config)
    print(f"Override max_results: {rag2.get_config('max_results')}")
    print(f"Technique config: {rag2.get_config('technique_config')}")
    print()


def main():
    """Run all demos."""
    print("RAG Templates Standard API Demo")
    print("=" * 50)
    print()
    
    try:
        demo_basic_technique_selection()
        demo_advanced_configuration()
        demo_technique_registry()
        demo_technique_switching()
        demo_backward_compatibility()
        demo_configuration_inheritance()
        
        print("✅ All demos completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()