#!/usr/bin/env python3
"""
Detailed exploration of LangChain text splitters for our chunking needs
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

def explore_langchain_splitters():
    """Explore available LangChain text splitters and their capabilities."""
    
    print("🔍 Exploring LangChain Text Splitters")
    print("=" * 60)
    
    try:
        from langchain.text_splitter import (
            RecursiveCharacterTextSplitter,
            CharacterTextSplitter,
            TokenTextSplitter,
            NLTKTextSplitter,
            SpacyTextSplitter,
            SentenceTransformersTokenTextSplitter,
            LatexTextSplitter,
            PythonCodeTextSplitter,
            RecursiveJsonSplitter
        )
        
        # Test text - biomedical abstract sample
        test_text = """
        Background: Alzheimer's disease (AD) is a progressive neurodegenerative disorder characterized by cognitive decline and memory loss. The pathological hallmarks of AD include amyloid-beta (Aβ) plaques and neurofibrillary tangles composed of hyperphosphorylated tau protein.
        
        Methods: We conducted a systematic review of recent literature on AD biomarkers. Our analysis included 150 studies published between 2020-2023, focusing on cerebrospinal fluid (CSF), blood-based, and neuroimaging biomarkers.
        
        Results: The study revealed that plasma phosphorylated tau (p-tau181 and p-tau217) showed high diagnostic accuracy for AD detection. Neurofilament light chain (NfL) levels were significantly elevated in AD patients compared to controls (p < 0.001). Neuroimaging studies demonstrated that amyloid PET and tau PET scans provided complementary information for disease staging.
        
        Conclusions: Multi-modal biomarker approaches combining blood-based markers with neuroimaging show promise for early AD detection and monitoring disease progression. These findings support the development of precision medicine approaches for AD diagnosis and treatment.
        """
        
        splitters_to_test = [
            ("RecursiveCharacterTextSplitter", RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )),
            ("CharacterTextSplitter", CharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separator="\n\n"
            )),
            ("TokenTextSplitter", TokenTextSplitter(
                chunk_size=128,  # tokens
                chunk_overlap=10
            )),
            ("NLTKTextSplitter", NLTKTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )),
        ]
        
        print(f"📄 Test Text Length: {len(test_text)} characters")
        print(f"📄 Test Text Word Count: {len(test_text.split())} words")
        print()
        
        for splitter_name, splitter in splitters_to_test:
            print(f"🔧 Testing {splitter_name}")
            print("-" * 40)
            
            try:
                chunks = splitter.split_text(test_text)
                
                print(f"   Chunks created: {len(chunks)}")
                print(f"   Avg chunk length: {sum(len(c) for c in chunks) / len(chunks):.1f} chars")
                print(f"   Min chunk length: {min(len(c) for c in chunks)} chars")
                print(f"   Max chunk length: {max(len(c) for c in chunks)} chars")
                
                # Show first chunk as example
                if chunks:
                    first_chunk = chunks[0]
                    print(f"   First chunk preview: {first_chunk[:100]}...")
                
                print()
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                print()
        
        # Test semantic chunking if available
        try:
            print("🧠 Testing Semantic Chunking (if available)")
            print("-" * 40)
            
            # Try to import semantic chunker
            try:
                from langchain_experimental.text_splitter import SemanticChunker
                from langchain_openai.embeddings import OpenAIEmbeddings
                
                print("   SemanticChunker available but requires OpenAI API key")
                print("   Would use: SemanticChunker.from_tiktoken_encoder()")
                
            except ImportError:
                print("   SemanticChunker not available in current installation")
            
        except Exception as e:
            print(f"   Error exploring semantic chunking: {e}")
        
        print()
        
    except ImportError as e:
        print(f"❌ Error importing LangChain splitters: {e}")

def explore_tiktoken_integration():
    """Explore TikToken for precise token-based chunking."""
    
    print("🎯 Exploring TikToken for Token-Based Chunking")
    print("=" * 60)
    
    try:
        import tiktoken
        
        # Test with different encodings
        encodings_to_test = [
            ("cl100k_base", "GPT-4, GPT-3.5-turbo"),
            ("p50k_base", "GPT-3 (davinci, curie, babbage, ada)"),
            ("r50k_base", "GPT-3 (davinci, curie, babbage, ada) - older"),
        ]
        
        test_text = "Alzheimer's disease is a progressive neurodegenerative disorder."
        
        for encoding_name, description in encodings_to_test:
            try:
                encoding = tiktoken.get_encoding(encoding_name)
                tokens = encoding.encode(test_text)
                
                print(f"📊 {encoding_name} ({description})")
                print(f"   Text: {test_text}")
                print(f"   Token count: {len(tokens)}")
                print(f"   Tokens: {tokens[:10]}..." if len(tokens) > 10 else f"   Tokens: {tokens}")
                print()
                
            except Exception as e:
                print(f"   ❌ Error with {encoding_name}: {e}")
                print()
        
        # Demonstrate chunking function
        print("🔧 Token-Based Chunking Function Example")
        print("-" * 40)
        
        def chunk_by_tokens(text: str, max_tokens: int = 100, overlap_tokens: int = 10, encoding_name: str = "cl100k_base"):
            """Chunk text by token count with overlap."""
            encoding = tiktoken.get_encoding(encoding_name)
            tokens = encoding.encode(text)
            
            chunks = []
            start = 0
            
            while start < len(tokens):
                end = min(start + max_tokens, len(tokens))
                chunk_tokens = tokens[start:end]
                chunk_text = encoding.decode(chunk_tokens)
                chunks.append({
                    'text': chunk_text,
                    'token_count': len(chunk_tokens),
                    'start_token': start,
                    'end_token': end
                })
                
                # Move start with overlap
                start = max(end - overlap_tokens, start + 1)
                if start >= end:
                    break
            
            return chunks
        
        # Test the function
        long_text = """
        Alzheimer's disease (AD) is the most common cause of dementia, affecting millions worldwide. 
        The disease is characterized by progressive cognitive decline, memory loss, and behavioral changes. 
        Pathologically, AD is marked by the accumulation of amyloid-beta plaques and neurofibrillary tangles 
        in the brain. Early detection through biomarkers is crucial for effective intervention and treatment.
        """
        
        chunks = chunk_by_tokens(long_text, max_tokens=50, overlap_tokens=5)
        
        print(f"   Original text: {len(long_text)} characters")
        print(f"   Chunks created: {len(chunks)}")
        
        for i, chunk in enumerate(chunks):
            print(f"   Chunk {i+1}: {chunk['token_count']} tokens - {chunk['text'][:50]}...")
        
        print()
        
    except ImportError as e:
        print(f"❌ TikToken not available: {e}")
    except Exception as e:
        print(f"❌ Error exploring TikToken: {e}")

def compare_with_current_implementation():
    """Compare modern libraries with our current implementation."""
    
    print("⚖️  Comparison with Current Implementation")
    print("=" * 60)
    
    comparison = {
        "Current Custom Implementation": {
            "pros": [
                "Tailored to our specific needs",
                "IRIS database integration",
                "Hybrid strategy combining semantic + fixed-size",
                "Custom metadata tracking"
            ],
            "cons": [
                "Limited semantic analysis (simple heuristics)",
                "No real sentence embeddings",
                "Reinventing the wheel",
                "Less battle-tested than mature libraries"
            ],
            "complexity": "Medium",
            "maintenance": "High (custom code to maintain)"
        },
        
        "LangChain RecursiveCharacterTextSplitter": {
            "pros": [
                "Mature and well-tested",
                "Preserves document structure",
                "Configurable separators",
                "Wide community adoption"
            ],
            "cons": [
                "Character-based, not semantic",
                "No built-in embedding integration",
                "May break semantic boundaries"
            ],
            "complexity": "Low",
            "maintenance": "Low (library maintained)"
        },
        
        "LangChain TokenTextSplitter + TikToken": {
            "pros": [
                "Exact token control",
                "Model-specific compatibility",
                "No token overflow issues",
                "Fast and efficient"
            ],
            "cons": [
                "May break semantic boundaries",
                "Model-dependent",
                "No semantic awareness"
            ],
            "complexity": "Low",
            "maintenance": "Low"
        },
        
        "Hybrid: LangChain + Custom Semantic Layer": {
            "pros": [
                "Best of both worlds",
                "Proven base with custom enhancements",
                "Can add semantic analysis on top",
                "Flexible and extensible"
            ],
            "cons": [
                "More complex implementation",
                "Requires integration work",
                "Multiple dependencies"
            ],
            "complexity": "Medium-High",
            "maintenance": "Medium"
        }
    }
    
    for approach, details in comparison.items():
        print(f"🔍 {approach}")
        print(f"   Pros: {', '.join(details['pros'])}")
        print(f"   Cons: {', '.join(details['cons'])}")
        print(f"   Complexity: {details['complexity']}")
        print(f"   Maintenance: {details['maintenance']}")
        print()

def recommend_implementation_strategy():
    """Provide specific recommendations for our use case."""
    
    print("🎯 Implementation Strategy Recommendations")
    print("=" * 60)
    
    recommendations = [
        {
            "phase": "Phase 1: Quick Win",
            "approach": "Replace custom splitters with LangChain",
            "details": [
                "Use RecursiveCharacterTextSplitter for general chunking",
                "Use TokenTextSplitter for token-aware chunking",
                "Keep existing IRIS integration and metadata",
                "Maintain current hybrid strategy concept"
            ],
            "effort": "Low",
            "timeline": "1-2 days",
            "benefits": "Immediate improvement in chunking quality"
        },
        {
            "phase": "Phase 2: Enhanced Semantic Chunking",
            "approach": "Add semantic analysis layer",
            "details": [
                "Install sentence-transformers for biomedical models",
                "Implement semantic similarity-based chunking",
                "Use biomedical sentence transformers (e.g., BioBERT-based)",
                "Combine with LangChain base splitters"
            ],
            "effort": "Medium",
            "timeline": "3-5 days",
            "benefits": "Better semantic coherence for biomedical text"
        },
        {
            "phase": "Phase 3: Advanced Integration",
            "approach": "Full integration with RAG pipelines",
            "details": [
                "Integrate chunking with all 7 RAG techniques",
                "Add chunk-level retrieval and context reconstruction",
                "Implement adaptive chunking based on query type",
                "Add performance monitoring and optimization"
            ],
            "effort": "High",
            "timeline": "1-2 weeks",
            "benefits": "Production-ready chunking system"
        }
    ]
    
    for rec in recommendations:
        print(f"📋 {rec['phase']}")
        print(f"   Approach: {rec['approach']}")
        print(f"   Details:")
        for detail in rec['details']:
            print(f"     - {detail}")
        print(f"   Effort: {rec['effort']}")
        print(f"   Timeline: {rec['timeline']}")
        print(f"   Benefits: {rec['benefits']}")
        print()

if __name__ == "__main__":
    explore_langchain_splitters()
    explore_tiktoken_integration()
    compare_with_current_implementation()
    recommend_implementation_strategy()