#!/usr/bin/env python3
"""
Test script to verify RAGAS fixes work correctly
"""

import os
import sys
from dotenv import load_dotenv
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def test_ragas_fixes():
    """Test that RAGAS evaluation works with our fixes"""
    
    print("Testing RAGAS fixes...")
    
    # Initialize components
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=100)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create test dataset with 'response' key (not 'answer')
    dataset = Dataset.from_dict({
        'question': ['What is the capital of France?'],
        'response': ['Paris is the capital of France.'],  # Using 'response' key
        'contexts': [['France is a country in Europe. Paris is its capital city.']],
        'ground_truth': ['Paris']
    })
    
    # Run evaluation
    metrics = [answer_relevancy, context_precision]
    
    try:
        print("Running RAGAS evaluation...")
        evaluation_result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings
        )
        
        print("✅ RAGAS evaluation completed successfully")
        
        # Test accessing results using dictionary-like access
        answer_rel_raw = evaluation_result['answer_relevancy']
        context_prec_raw = evaluation_result['context_precision']
        
        # Extract scalar values from lists if needed
        answer_rel = answer_rel_raw[0] if isinstance(answer_rel_raw, list) else answer_rel_raw
        context_prec = context_prec_raw[0] if isinstance(context_prec_raw, list) else context_prec_raw
        
        print(f"✅ Answer Relevancy: {answer_rel}")
        print(f"✅ Context Precision: {context_prec}")
        
        # Verify we got numeric values
        assert answer_rel is not None, "Answer relevancy should not be None"
        assert context_prec is not None, "Context precision should not be None"
        assert isinstance(answer_rel, (int, float)), "Answer relevancy should be numeric"
        assert isinstance(context_prec, (int, float)), "Context precision should be numeric"
        
        print("✅ All tests passed! RAGAS fixes are working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ragas_fixes()
    sys.exit(0 if success else 1)