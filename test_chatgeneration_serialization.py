#!/usr/bin/env python3
"""
Test script to reproduce and verify the ChatGeneration serialization issue.
"""

import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.load import dumpd
from common.llm_cache_manager import LangchainIRISCacheWrapper
from common.llm_cache_iris import create_iris_cache_backend
from common.llm_cache_config import load_cache_config

def test_chatgeneration_serialization():
    """Test serialization of ChatGeneration objects."""
    print("Testing ChatGeneration serialization...")
    
    # Create test ChatGeneration objects
    test_generations = [
        # Basic Generation
        Generation(text="Basic generation text"),
        
        # ChatGeneration with AIMessage
        ChatGeneration(
            text="Chat generation text",
            message=AIMessage(content="AI message content")
        ),
        
        # ChatGeneration with complex message
        ChatGeneration(
            text="Complex chat generation",
            message=AIMessage(
                content="Complex AI message",
                additional_kwargs={"model": "gpt-4", "usage": {"tokens": 100}}
            )
        )
    ]
    
    print(f"Testing {len(test_generations)} generation objects...")
    
    for i, generation in enumerate(test_generations):
        print(f"\nTesting generation {i+1}: {type(generation).__name__}")
        
        # Test current dumpd approach
        try:
            dumped = dumpd(generation)
            print(f"  dumpd success: {type(dumped)}")
            
            # Try to serialize the dumped result
            try:
                json_str = json.dumps(dumped)
                print(f"  JSON serialization success: {len(json_str)} chars")
            except Exception as json_error:
                print(f"  JSON serialization FAILED: {json_error}")
                print(f"  Dumped object type: {type(dumped)}")
                if hasattr(dumped, 'keys'):
                    print(f"  Dumped keys: {list(dumped.keys())}")
                    for key, value in dumped.items():
                        print(f"    {key}: {type(value)}")
                        
        except Exception as dump_error:
            print(f"  dumpd FAILED: {dump_error}")

def test_improved_serialization():
    """Test the improved serialization approach."""
    print("\n" + "="*50)
    print("Testing improved serialization approach...")
    
    from langchain_core.outputs import ChatGeneration, Generation
    from langchain_core.messages import AIMessage, BaseMessage
    from langchain_core.load import dumpd
    
    # Create test ChatGeneration objects
    test_generations = [
        Generation(text="Basic generation text"),
        ChatGeneration(
            text="Chat generation text",
            message=AIMessage(content="AI message content")
        ),
        ChatGeneration(
            text="Complex chat generation",
            message=AIMessage(
                content="Complex AI message",
                additional_kwargs={"model": "gpt-4", "usage": {"tokens": 100}}
            )
        )
    ]
    
    def improved_serialize_generation(generation):
        """Improved serialization for Generation objects."""
        try:
            # First attempt: use dumpd
            serialized_gen = dumpd(generation)
            
            # Special handling for ChatGeneration with BaseMessage
            if hasattr(generation, 'message') and isinstance(generation.message, BaseMessage):
                # Check if the message in serialized form is still a BaseMessage
                if 'message' in serialized_gen and isinstance(serialized_gen['message'], BaseMessage):
                    # Explicitly serialize the message
                    serialized_gen['message'] = dumpd(generation.message)
            
            # Test JSON serialization
            json.dumps(serialized_gen)
            return serialized_gen
            
        except Exception as e:
            print(f"    dumpd failed: {e}")
            # Fallback approach
            try:
                if hasattr(generation, 'dict'):
                    gen_data = generation.dict()
                    # Handle BaseMessage in dict
                    if 'message' in gen_data and hasattr(gen_data['message'], 'dict'):
                        gen_data['message'] = gen_data['message'].dict()
                    json.dumps(gen_data)  # Test serialization
                    return gen_data
                else:
                    # Last resort
                    return {'text': str(generation), 'type': type(generation).__name__}
            except Exception as fallback_error:
                print(f"    Fallback failed: {fallback_error}")
                return {'text': str(generation), 'type': type(generation).__name__, 'error': 'serialization_failed'}
    
    for i, generation in enumerate(test_generations):
        print(f"\nTesting improved serialization {i+1}: {type(generation).__name__}")
        try:
            result = improved_serialize_generation(generation)
            json_str = json.dumps(result)
            print(f"  SUCCESS: {len(json_str)} chars")
        except Exception as e:
            print(f"  FAILED: {e}")

def test_runtime_conditions():
    """Test serialization under actual runtime conditions that might cause issues."""
    print("\n" + "="*50)
    print("Testing runtime conditions that might cause serialization issues...")
    
    from langchain_openai import ChatOpenAI
    from langchain_core.outputs import LLMResult
    
    # Test with actual LLM response that might contain problematic objects
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # This creates the actual Generation objects that would be cached
        result = llm.generate([["What is machine learning?"]])
        
        print(f"LLM Result type: {type(result)}")
        print(f"Generations: {len(result.generations)}")
        
        for i, generation_list in enumerate(result.generations):
            print(f"\nGeneration list {i}: {len(generation_list)} items")
            for j, generation in enumerate(generation_list):
                print(f"  Generation {j}: {type(generation)}")
                
                # Test serialization like the cache would
                try:
                    from langchain_core.load import dumpd
                    dumped = dumpd(generation)
                    json_str = json.dumps(dumped)
                    print(f"    Serialization SUCCESS: {len(json_str)} chars")
                except Exception as e:
                    print(f"    Serialization FAILED: {e}")
                    print(f"    Generation attributes: {dir(generation)}")
                    if hasattr(generation, 'message'):
                        print(f"    Message type: {type(generation.message)}")
                        print(f"    Message attributes: {dir(generation.message)}")
                    
    except Exception as e:
        print(f"Could not test with actual LLM (API key issue?): {e}")

def test_edge_cases():
    """Test edge cases that might cause serialization issues."""
    print("\n" + "="*50)
    print("Testing edge cases...")
    
    from langchain_core.outputs import ChatGeneration, Generation
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_core.load import dumpd
    
    # Edge case: Generation with None values
    edge_cases = [
        Generation(text=None),
        Generation(text="", generation_info=None),
        ChatGeneration(text="test", message=AIMessage(content="")),
        ChatGeneration(text="test", message=AIMessage(content=None)),
    ]
    
    for i, generation in enumerate(edge_cases):
        print(f"\nEdge case {i+1}: {type(generation).__name__}")
        try:
            dumped = dumpd(generation)
            json_str = json.dumps(dumped)
            print(f"  SUCCESS: {len(json_str)} chars")
        except Exception as e:
            print(f"  FAILED: {e}")
            print(f"  Generation: {generation}")

if __name__ == "__main__":
    test_chatgeneration_serialization()
    test_improved_serialization()
    test_runtime_conditions()
    test_edge_cases()