"""
Test script for ColBERTRAG Pipeline.

This script tests the ColBERTRAGPipeline implementation by executing a sample query
and logging the results.
"""

import logging
import sys
import os
import openai
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import iris_rag

load_dotenv()

# === CONFIGURATION ===
USE_REAL_LLM = True  # Change to False to use dummy_llm
OPENAI_MODEL = "gpt-4.1-mini"  # GPT-4.1 Mini
client = openai.OpenAI()

# Optional: Dummy LLM function
def dummy_llm(prompt: str) -> str:
    print("\n--- Prompt to LLM ---\n")
    print(prompt)
    return "This is a dummy answer generated from the context."

# Real LLM function using OpenAI GPT-4.1 Mini
def openai_llm(prompt: str) -> str:
    print("\n--- Prompt to LLM (OpenAI) ---\n")
    print(prompt)

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant answering based on the context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

def main():
    # Setup logging
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger()

    llm_func = openai_llm if USE_REAL_LLM else dummy_llm

    print("Creating ColBERTRAG Pipeline with Auto-Setup")
    # Create pipeline using iris_rag factory with auto_setup=True
    colbert_rag_pipeline = iris_rag.create_pipeline(
        pipeline_type="colbert",
        llm_func=llm_func,  
        auto_setup=True,     # Crucial: handles schema initialization automatically
        validate_requirements=True
    )
    print("✓ ColBERTRAG Pipeline created successfully")

    # Test validation
    logger.info("Running ColBERT validation...")
    validation_result = colbert_rag_pipeline.validate_setup()
    
    if validation_result:
        logger.info("✅ ColBERT validation passed!")
    else:
        logger.error("❌ ColBERT validation failed")
        return


    print("Running ColBERTRAG Pipeline")
    # Run a sample query
    # Try a simple query
    try:
        response = colbert_rag_pipeline.query("What are the effects of BRCA1 mutations?", top_k=3)
    except Exception as e:
        logger.error(f"❌ ColBERT query failed: {e}")
        return

    # Print final answer
    print("\n========== ColBERTRAG Pipeline Output ==========")
    print(f"Query: {response['query']}")
    print(f"Answer: {response['answer']}")
    print(f"Execution Time: {response['execution_time']:.2f}s")

if __name__ == "__main__":
    main()