#!/usr/bin/env python3
"""
Research script to explore modern Python chunking libraries
"""

import subprocess
import sys
import json
from typing import Dict, List, Any

def check_library_availability(library_name: str) -> Dict[str, Any]:
    """Check if a library is available and get basic info."""
    try:
        result = subprocess.run([
            sys.executable, "-c", 
            f"import {library_name}; print('Available'); print(getattr({library_name}, '__version__', 'Unknown version'))"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            return {
                "available": True,
                "version": lines[1] if len(lines) > 1 else "Unknown",
                "error": None
            }
        else:
            return {
                "available": False,
                "version": None,
                "error": result.stderr.strip()
            }
    except Exception as e:
        return {
            "available": False,
            "version": None,
            "error": str(e)
        }

def research_chunking_libraries():
    """Research available chunking libraries."""
    
    # List of libraries to research
    libraries_to_check = [
        # LangChain ecosystem
        ("langchain", "LangChain - Popular RAG framework with text splitters"),
        ("langchain_text_splitters", "LangChain Text Splitters - Dedicated text splitting module"),
        ("langchain_community", "LangChain Community - Additional text splitters"),
        
        # Semantic chunking libraries
        ("semantic_text_splitter", "Semantic Text Splitter - Rust-based semantic chunking"),
        ("chunking", "Chunking - Simple text chunking library"),
        ("text_chunker", "Text Chunker - Advanced text chunking"),
        
        # NLP libraries with chunking capabilities
        ("spacy", "spaCy - NLP library with sentence segmentation"),
        ("nltk", "NLTK - Natural Language Toolkit"),
        ("transformers", "Hugging Face Transformers - For tokenization"),
        ("sentence_transformers", "Sentence Transformers - For semantic similarity"),
        
        # Document processing libraries
        ("unstructured", "Unstructured - Document parsing and chunking"),
        ("llama_index", "LlamaIndex - RAG framework with chunking"),
        ("haystack", "Haystack - NLP framework with document processing"),
        
        # Specialized chunking libraries
        ("tiktoken", "TikToken - OpenAI's tokenizer"),
        ("tokenizers", "Tokenizers - Fast tokenization library"),
        ("textstat", "TextStat - Text statistics and readability"),
    ]
    
    print("🔍 Researching Modern Python Chunking Libraries")
    print("=" * 60)
    
    available_libraries = []
    unavailable_libraries = []
    
    for lib_name, description in libraries_to_check:
        print(f"\n📦 Checking {lib_name}...")
        result = check_library_availability(lib_name)
        
        if result["available"]:
            print(f"   ✅ Available - Version: {result['version']}")
            available_libraries.append({
                "name": lib_name,
                "description": description,
                "version": result["version"]
            })
        else:
            print(f"   ❌ Not available - {result['error'][:100]}...")
            unavailable_libraries.append({
                "name": lib_name,
                "description": description,
                "error": result["error"]
            })
    
    print(f"\n📊 Summary:")
    print(f"   Available: {len(available_libraries)}")
    print(f"   Unavailable: {len(unavailable_libraries)}")
    
    # Try to get more detailed info about available libraries
    print(f"\n🔬 Detailed Analysis of Available Libraries:")
    print("=" * 60)
    
    for lib in available_libraries:
        print(f"\n📚 {lib['name']} (v{lib['version']})")
        print(f"   {lib['description']}")
        
        # Try to get more info about the library
        if lib['name'] == 'langchain':
            try:
                result = subprocess.run([
                    sys.executable, "-c", 
                    "from langchain.text_splitter import *; import inspect; "
                    "splitters = [name for name, obj in globals().items() if 'Splitter' in name and inspect.isclass(obj)]; "
                    "print('Text Splitters:', splitters[:10])"
                ], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print(f"   Available splitters: {result.stdout.strip()}")
            except:
                pass
        
        elif lib['name'] == 'spacy':
            try:
                result = subprocess.run([
                    sys.executable, "-c", 
                    "import spacy; print('Models available:', spacy.util.get_installed_models())"
                ], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print(f"   {result.stdout.strip()}")
            except:
                pass
    
    return {
        "available": available_libraries,
        "unavailable": unavailable_libraries
    }

def analyze_chunking_approaches():
    """Analyze different chunking approaches available."""
    
    print(f"\n🧠 Modern Chunking Approaches Analysis:")
    print("=" * 60)
    
    approaches = {
        "Fixed-Size Chunking": {
            "description": "Split text into fixed-size chunks with optional overlap",
            "libraries": ["langchain.text_splitter.CharacterTextSplitter", "tiktoken"],
            "pros": ["Predictable chunk sizes", "Simple implementation", "Good for token limits"],
            "cons": ["May break semantic boundaries", "Context loss at boundaries"],
            "best_for": "Embedding models with strict token limits"
        },
        
        "Recursive Character Splitting": {
            "description": "Split on multiple separators hierarchically",
            "libraries": ["langchain.text_splitter.RecursiveCharacterTextSplitter"],
            "pros": ["Preserves structure", "Flexible separators", "Widely used"],
            "cons": ["Still character-based", "Limited semantic awareness"],
            "best_for": "General-purpose text splitting with structure preservation"
        },
        
        "Semantic Chunking": {
            "description": "Split based on semantic similarity between sentences",
            "libraries": ["semantic_text_splitter", "sentence_transformers"],
            "pros": ["Preserves semantic coherence", "Better context retention"],
            "cons": ["Computationally expensive", "Variable chunk sizes"],
            "best_for": "High-quality retrieval where coherence matters"
        },
        
        "Token-Based Chunking": {
            "description": "Split based on actual token counts from specific models",
            "libraries": ["tiktoken", "transformers.tokenizers"],
            "pros": ["Exact token control", "Model-specific", "No token overflow"],
            "cons": ["Model-dependent", "May break semantic boundaries"],
            "best_for": "Specific model compatibility and token budget control"
        },
        
        "Document Structure Chunking": {
            "description": "Split based on document structure (headers, paragraphs, etc.)",
            "libraries": ["unstructured", "langchain.text_splitter.MarkdownHeaderTextSplitter"],
            "pros": ["Preserves document structure", "Natural boundaries"],
            "cons": ["Document format dependent", "Variable sizes"],
            "best_for": "Structured documents with clear hierarchy"
        },
        
        "Hybrid Chunking": {
            "description": "Combine multiple strategies for optimal results",
            "libraries": ["Custom implementations", "llama_index"],
            "pros": ["Best of multiple approaches", "Adaptive", "Flexible"],
            "cons": ["Complex implementation", "More parameters to tune"],
            "best_for": "Production systems requiring optimal performance"
        }
    }
    
    for approach, details in approaches.items():
        print(f"\n🔧 {approach}")
        print(f"   Description: {details['description']}")
        print(f"   Libraries: {', '.join(details['libraries'])}")
        print(f"   Pros: {', '.join(details['pros'])}")
        print(f"   Cons: {', '.join(details['cons'])}")
        print(f"   Best for: {details['best_for']}")

def recommend_for_biomedical_use_case():
    """Provide recommendations for biomedical PMC documents."""
    
    print(f"\n🏥 Recommendations for Biomedical PMC Documents:")
    print("=" * 60)
    
    recommendations = [
        {
            "priority": "HIGH",
            "library": "LangChain Text Splitters",
            "reason": "Mature, well-tested, multiple splitting strategies",
            "specific_splitters": [
                "RecursiveCharacterTextSplitter - For general text",
                "TokenTextSplitter - For exact token control",
                "SemanticChunker - For semantic coherence (if available)"
            ]
        },
        {
            "priority": "HIGH", 
            "library": "Semantic Text Splitter (Rust-based)",
            "reason": "Fast semantic chunking, good for scientific text coherence",
            "benefits": "Preserves scientific concepts and terminology"
        },
        {
            "priority": "MEDIUM",
            "library": "spaCy + sentence-transformers",
            "reason": "Custom semantic chunking with biomedical models",
            "benefits": "Can use biomedical sentence transformers for better domain understanding"
        },
        {
            "priority": "MEDIUM",
            "library": "Unstructured",
            "reason": "Good for processing various document formats",
            "benefits": "Handles PDFs, XMLs, and other scientific document formats"
        },
        {
            "priority": "LOW",
            "library": "TikToken + Custom Logic",
            "reason": "For exact OpenAI model compatibility",
            "benefits": "Precise token counting for GPT models"
        }
    ]
    
    for rec in recommendations:
        print(f"\n🎯 {rec['priority']} PRIORITY: {rec['library']}")
        print(f"   Reason: {rec['reason']}")
        if 'specific_splitters' in rec:
            print(f"   Specific splitters:")
            for splitter in rec['specific_splitters']:
                print(f"     - {splitter}")
        if 'benefits' in rec:
            print(f"   Benefits: {rec['benefits']}")

if __name__ == "__main__":
    # Run the research
    library_results = research_chunking_libraries()
    analyze_chunking_approaches()
    recommend_for_biomedical_use_case()
    
    # Save results to file
    with open("chunking_library_research.json", "w") as f:
        json.dump(library_results, f, indent=2)
    
    print(f"\n💾 Results saved to chunking_library_research.json")