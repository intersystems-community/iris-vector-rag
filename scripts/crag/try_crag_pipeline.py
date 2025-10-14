"""
Test script for CRAG Pipeline.

This script tests the CRAGPipeline implementation by executing a sample query
and logging the results.

Please make sure to load your data before running this script.
Run make load-data in the project root to load the data.
"""

import logging
import os
import sys

import openai
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

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
            {
                "role": "system",
                "content": "You are a helpful assistant answering based on the context.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def main():
    # Setup logging
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger()

    llm_func = openai_llm if USE_REAL_LLM else dummy_llm

    print("Creating CRAG Pipeline with Auto-Setup")
    # Create pipeline using iris_rag factory with auto_setup=True
    crag_pipeline = iris_rag.create_pipeline(
        pipeline_type="crag",
        llm_func=llm_func,
        auto_setup=True,
        validate_requirements=True,
    )
    print("✓ CRAG Pipeline created successfully")

    print("Running CRAG Pipeline")
    # Run a sample query
    query = "What demographics are at risk of weight gain?"
    response = crag_pipeline.query(query, top_k=3)

    # Print final answer
    print("\n========== CRAG Pipeline Output ==========")
    print(f"Query: {response['query']}")
    print(f"Answer: {response['answer']}")
    print(f"Execution Time: {response['execution_time']:.2f}s")


if __name__ == "__main__":
    main()
