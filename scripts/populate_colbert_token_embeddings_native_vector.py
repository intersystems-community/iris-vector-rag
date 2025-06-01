#!/usr/bin/env python3
"""
Populate ColBERT Token Embeddings - Native VECTOR Version
Properly populate the DocumentTokenEmbeddings table with native VECTOR(DOUBLE, 128)
"""

import os
import sys
import logging
import json
import numpy as np
from typing import List, Dict, Any
import argparse

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.iris_connector import get_iris_connection
import torch
from transformers import AutoTokenizer, AutoModel
from common.utils import get_config_value
import common.utils # Import the module itself to access its global _config_cache

# Force a re-read of the config file by clearing common.utils._config_cache
common.utils._config_cache = None
logger_init_temp = logging.getLogger(__name__) # temp logger for this line
logger_init_temp.info("Forcing common.utils._config_cache to None to ensure fresh config load.")


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_hf_model_cache = {} # Cache for tokenizer and model
def get_real_token_embeddings(text: str, max_length: int = 512) -> tuple[List[str], List[List[float]]]:
    """
    Generates real token embeddings for a given text using a HuggingFace model.
    Returns a tuple of (tokens, token_embeddings).
    """
    global _hf_model_cache
    
    # Always get the model name from config to ensure it's the latest desired one.
    current_model_name = get_config_value("colbert.document_encoder_model",
                                          get_config_value("embedding_model.name", "sentence-transformers/all-MiniLM-L6-v2"))

    # Check if the current_model_name is in cache and if the cached model's name_or_path matches.
    # This handles cases where the key might exist but points to an older version if not managed carefully.
    cached_tokenizer_model = _hf_model_cache.get(current_model_name)
    
    if cached_tokenizer_model is None or cached_tokenizer_model[0].name_or_path != current_model_name:
        if cached_tokenizer_model is not None: # Key exists but name_or_path mismatch
            logger.info(f"Model name '{current_model_name}' in config differs from cached model's name_or_path '{cached_tokenizer_model[0].name_or_path}'. Re-loading.")
        else: # Not in cache at all
            logger.info(f"Model '{current_model_name}' not in cache. Loading.")

        # For simplicity and to ensure freshness, clear the entire cache if we need to load/reload.
        # A more sophisticated cache might evict only the specific old entry.
        if _hf_model_cache: # Check if cache is not empty before clearing
            logger.info("Clearing _hf_model_cache to load new/updated model.")
            _hf_model_cache.clear()
        
        logger.info(f"Loading HuggingFace tokenizer and model: {current_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(current_model_name)
        # Load model with trust_remote_code=True to ensure custom ColBERT code/architecture is used
        model = AutoModel.from_pretrained(current_model_name, trust_remote_code=True)
        model.eval() # Set to evaluation mode
        _hf_model_cache[current_model_name] = (tokenizer, model)
    else:
        # Model is in cache and its name_or_path matches, so use it.
        logger.info(f"Using cached HuggingFace tokenizer and model: {current_model_name}")
        tokenizer, model = cached_tokenizer_model

    token_embeddings_tensor = None
    input_ids_list = None

    with torch.no_grad():
        if hasattr(model, 'encode') and callable(getattr(model, 'encode')) and 'is_query' in model.encode.__code__.co_varnames:
            logger.info(f"Attempting to use model.encode() for model {current_model_name}")
            # The model.encode() from PyLate's ColBERT returns a list of numpy arrays (embeddings per document)
            # or a single numpy array if one document string is passed.
            # We are processing one document at a time here.
            # It expects raw text, not tokenized inputs.
            # It handles tokenization and projection internally.
            try:
                # Ensure text is a single string for single document processing
                if isinstance(text, list): # Should not happen if called one doc at a time
                    text_for_encode = text[0] if text else ""
                else:
                    text_for_encode = text

                # PyLate's ColBERT model.encode returns embeddings directly.
                # It might not return input_ids directly in the same way.
                # We might need to tokenize separately just for getting the tokens if model.encode doesn't provide them.
                
                # First, get tokens for mapping
                inputs_for_tokens = tokenizer(text_for_encode, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length") # Use padding="max_length" for consistent token list
                input_ids_list = inputs_for_tokens["input_ids"].squeeze(0).tolist()

                # Then, get embeddings using model.encode
                # model.encode might return a list of arrays if multiple texts are passed, or one array for one text.
                # We expect one document text here.
                encoded_output = model.encode(text_for_encode, is_query=False, batch_size=1, show_progress_bar=False) # Pass as a single string

                if isinstance(encoded_output, list) and len(encoded_output) == 1:
                     token_embeddings_tensor = torch.tensor(encoded_output[0]) # Convert numpy array to tensor
                elif isinstance(encoded_output, np.ndarray):
                     token_embeddings_tensor = torch.tensor(encoded_output)
                else:
                    logger.error(f"Unexpected output type from model.encode: {type(encoded_output)}. Expected numpy array or list of one.")
                    return [], []
                
                logger.info(f"Used model.encode(). Output shape: {token_embeddings_tensor.shape}")
                # Squeeze if it has an unnecessary batch dim of 1 (e.g. if encode was for a list of one doc)
                if token_embeddings_tensor.ndim == 3 and token_embeddings_tensor.shape[0] == 1:
                    token_embeddings_tensor = token_embeddings_tensor.squeeze(0)

            except Exception as e_encode:
                logger.error(f"Error using model.encode(): {e_encode}. Falling back to standard forward pass.", exc_info=True)
                # Fallback logic below will be triggered if token_embeddings_tensor is still None
                token_embeddings_tensor = None # Ensure fallback
        
        if token_embeddings_tensor is None: # Fallback if model.encode() was not available or failed
            logger.info("Using standard model forward pass (model(**inputs)).")
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding=True)
            input_ids_list = inputs["input_ids"].squeeze(0).tolist()
            outputs = model(**inputs)
            
            if hasattr(outputs, 'token_embeddings'): # Expected for ColBERT models with trust_remote_code=True
                token_embeddings_tensor = outputs.token_embeddings.squeeze(0)
                logger.info(f"Using outputs.token_embeddings. Shape: {token_embeddings_tensor.shape}")
            elif hasattr(outputs, 'last_hidden_state'):
                token_embeddings_tensor = outputs.last_hidden_state.squeeze(0)
                logger.warning(f"Using outputs.last_hidden_state as 'token_embeddings' attribute not found. Shape: {token_embeddings_tensor.shape}")
            else:
                logger.error("Could not find 'token_embeddings' or 'last_hidden_state' in model outputs.")
                return [], []

    if token_embeddings_tensor is None or input_ids_list is None:
        logger.error("Failed to obtain token embeddings or input_ids.")
        return [],[]

    tokens = tokenizer.convert_ids_to_tokens(input_ids_list)

    # Filter out padding tokens and their embeddings
    # Also filter out [CLS] and [SEP] if not desired for ColBERT-style tokens,
    # but for now, let's keep them as the model saw them.
    # ColBERT typically uses special query/document markers or relies on all tokens.
    
    valid_tokens = []
    valid_embeddings = []

    attention_mask = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"])).squeeze(0)

    for i, token_str in enumerate(tokens):
        if attention_mask[i].item() == 1: # Only include non-padded tokens
            # Optionally skip [CLS], [SEP] for pure content tokens, but ColBERT might use them.
            # if token_str in [tokenizer.cls_token, tokenizer.sep_token]:
            #     continue
            valid_tokens.append(token_str)
            valid_embeddings.append(token_embeddings_tensor[i].cpu().numpy().tolist())
            
    # Limit to a practical number of tokens if necessary, e.g., first 256-512 tokens
    # This should align with how ColBERT typically handles passage length.
    # The simple_tokenize previously limited to 30. Let's use a higher, more realistic limit.
    # Max_length for tokenizer already handles input length. This is for output.
    # ColBERT often uses fixed length token sequences per passage.
    # For now, let's return all valid tokens from the (potentially truncated) input.
    # The `max_length` to the tokenizer is the primary control here.

    # Ensure we don't exceed a practical limit for storage / processing, e.g. 512 tokens
    # This is a secondary check.
    # MAX_TOKENS_PER_DOC = 512 
    # valid_tokens = valid_tokens[:MAX_TOKENS_PER_DOC]
    # valid_embeddings = valid_embeddings[:MAX_TOKENS_PER_DOC]


    if not valid_tokens:
        logger.warning(f"No valid tokens produced for text (first 50 chars): {text[:50]}...")

    return valid_tokens, valid_embeddings

def populate_token_embeddings_for_document(iris_connector, doc_id: str, text_content: str) -> int:
    """Populate token embeddings for a single document using native VECTOR."""
    try:
        # Get real tokens and their embeddings
        tokens, token_embeddings = get_real_token_embeddings(text_content)

        if not tokens or not token_embeddings or len(tokens) != len(token_embeddings):
            logger.warning(f"No valid tokens or embeddings generated for doc_id {doc_id}. Skipping.")
            return 0
        
        cursor = iris_connector.cursor()
        
        # Check if embeddings already exist for this document (optional, can be removed if re-population is desired)
        # For now, let's keep it to avoid re-processing if script is run multiple times on same data.
        # A more robust check might involve checking a version or timestamp if content can change.
        cursor.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings WHERE doc_id = ?", (doc_id,))
        if cursor.fetchone()[0] > 0:
            logger.info(f"Token embeddings already exist for doc_id {doc_id}. Skipping.")
            cursor.close()
            return 0 # Or return the count of existing tokens if that's more useful
        
        tokens_inserted = 0
        for i, (token_str, embedding_list) in enumerate(zip(tokens, token_embeddings)):
            if not isinstance(embedding_list, list) or not all(isinstance(x, float) for x in embedding_list):
                logger.error(f"Invalid embedding format for token {i} in doc {doc_id}. Skipping token.")
                continue

            # Format numbers to ensure they are treated as doubles by IRIS TO_VECTOR
            embedding_db_str = "[" + ','.join([f"{x:.8f}" for x in embedding_list]) + "]"
            token_record_id = f"{doc_id}_{i}"
            logger.info(f"Attempting to insert token embedding for {token_record_id} with dimension: {len(embedding_list)}. First 3 elements: {embedding_list[:3]}") # DEBUGGING

            cursor.execute("""
                INSERT INTO RAG.DocumentTokenEmbeddings
                (id, doc_id, token_index, token_text, token_embedding)
                VALUES (?, ?, ?, ?, TO_VECTOR(?))
            """, (token_record_id, doc_id, i, token_str, embedding_db_str))
            
            tokens_inserted += 1
        
        if tokens_inserted > 0:
            logger.info(f"Successfully inserted {tokens_inserted} token embeddings for doc_id {doc_id}")
        
        cursor.close()
        return tokens_inserted
        
    except Exception as e:
        logger.error(f"Error populating token embeddings for {doc_id}: {e}", exc_info=True)
        return 0

def populate_all_token_embeddings(iris_connector, max_docs: int = 1000, doc_ids_file: str = None):
    """Populate token embeddings for specified documents or all documents needing them."""
    try:
        cursor = iris_connector.cursor()
        documents = []

        if doc_ids_file:
            logger.info(f"Processing documents from file: {doc_ids_file}")
            try:
                with open(doc_ids_file, 'r') as f:
                    doc_ids_to_process = [line.strip() for line in f if line.strip()]
                
                if not doc_ids_to_process:
                    logger.warning(f"No document IDs found in {doc_ids_file}. Nothing to process.")
                    cursor.close()
                    return 0, 0
                
                logger.info(f"Found {len(doc_ids_to_process)} document IDs in {doc_ids_file}. Fetching their content.")
                
                # Limit the number of doc_ids to process if max_docs is smaller than the list from file
                if len(doc_ids_to_process) > max_docs:
                    logger.info(f"Limiting doc_ids from file to {max_docs} as per --max_docs argument.")
                    doc_ids_to_process = doc_ids_to_process[:max_docs]

                placeholders = ','.join(['?'] * len(doc_ids_to_process))
                # Fetch documents, filtering for NULL content will happen in Python
                sql_query = f"""
                    SELECT doc_id, text_content
                    FROM RAG.SourceDocuments
                    WHERE doc_id IN ({placeholders}) AND text_content IS NOT NULL
                """
                cursor.execute(sql_query, doc_ids_to_process)
                documents_from_db = cursor.fetchall()

                # Create a dictionary for quick lookup of fetched documents
                docs_map = {doc[0]: doc[1] for doc in documents_from_db}
                
                # Preserve order from file and handle missing/empty content
                for doc_id_from_file in doc_ids_to_process:
                    if doc_id_from_file in docs_map:
                        documents.append((doc_id_from_file, docs_map[doc_id_from_file]))
                    else:
                        logger.warning(f"Could not find content for doc_id '{doc_id_from_file}' from file (or it doesn't exist/has NULL/empty content). Skipping.")
                
            except FileNotFoundError:
                logger.error(f"Error: Document ID file not found: {doc_ids_file}")
                cursor.close()
                return 0, 0
            except Exception as e_file:
                logger.error(f"Error reading or processing doc_ids_file {doc_ids_file}: {e_file}", exc_info=True)
                cursor.close()
                return 0, 0
        else:
            logger.info(f"No doc_ids_file provided. Fetching up to {max_docs} documents that need token embeddings and have non-empty content.")
            # Get documents that need token embeddings
            cursor.execute(f"""
                SELECT TOP {max_docs} doc_id, text_content
                FROM RAG.SourceDocuments
                WHERE doc_id NOT IN (
                    SELECT DISTINCT doc_id FROM RAG.DocumentTokenEmbeddings
                )
                AND text_content IS NOT NULL
            """)
            documents = cursor.fetchall()
        
        cursor.close()
        
        logger.info(f"Found {len(documents)} documents to process for token embeddings.")
        
        total_tokens_created_session = 0
        processed_docs_session = 0
        
        for doc_idx, (doc_id, raw_text_content) in enumerate(documents):
            try:
                text_content_str = ""
                if hasattr(raw_text_content, 'read'): # Check if it's an IRISInputStream
                    try:
                        byte_list = []
                        while True:
                            byte_val = raw_text_content.read()
                            if byte_val == -1: # EOF
                                break
                            byte_list.append(byte_val)
                        if byte_list: # Ensure byte_list is not empty before decoding
                           text_content_str = bytes(byte_list).decode('utf-8', errors='replace')
                    except Exception as stream_read_error:
                        logger.error(f"Error reading IRISInputStream for doc_id {doc_id}: {stream_read_error}")
                        continue # Skip this document
                elif isinstance(raw_text_content, str):
                    text_content_str = raw_text_content
                elif isinstance(raw_text_content, bytes): # Handle if content is already bytes
                    try:
                        text_content_str = raw_text_content.decode('utf-8', errors='replace')
                    except UnicodeDecodeError:
                        logger.warning(f"Could not decode bytes content for doc_id {doc_id}. Skipping.")
                        continue
                elif raw_text_content is None: # Handle if content is None
                     text_content_str = ""
                else:
                    logger.warning(f"Unsupported text_content type for doc_id {doc_id}: {type(raw_text_content)}. Skipping.")
                    continue

                # Limit text length for performance, consistent with previous logic
                text_content_str = text_content_str[:2000] if text_content_str else ""
                
                if len(text_content_str.strip()) < 10: # Skip if content is too short
                    logger.info(f"Skipping doc_id {doc_id} due to short content (less than 10 chars after stripping).")
                    continue
                
                # Assuming the helper function is named populate_token_embeddings_for_doc
                # The script uses populate_token_embeddings_for_document
                tokens_for_doc = populate_token_embeddings_for_document(iris_connector, doc_id, text_content_str)
                if tokens_for_doc > 0:
                    total_tokens_created_session += tokens_for_doc
                    processed_docs_session += 1
            
                if (doc_idx + 1) % 10 == 0 or (doc_idx + 1) == len(documents): # Log progress periodically
                    logger.info(f"Processed {doc_idx + 1}/{len(documents)} documents from current list, created {total_tokens_created_session} tokens this session.")
            except Exception as doc_proc_error:
                logger.error(f"Error processing document {doc_id} in populate_all_token_embeddings loop: {doc_proc_error}", exc_info=True)
                continue # Move to the next document

        logger.info(f"✅ Token embeddings population for current batch complete:")
        logger.info(f"   Documents processed in this session: {processed_docs_session}")
        logger.info(f"   Total tokens created in this session: {total_tokens_created_session}")
        return processed_docs_session, total_tokens_created_session
        
    except Exception as e:
        logger.error(f"Error in populate_all_token_embeddings: {e}", exc_info=True)
        # Ensure cursor is closed if an error occurs before its normal close point
        if 'cursor' in locals() and cursor and hasattr(cursor, 'closed') and not cursor.closed:
            try:
                cursor.close()
            except Exception as ce:
                logger.error(f"Failed to close cursor during exception handling: {ce}")
        return 0, 0

def verify_token_embeddings(iris_connector):
    """Verify token embeddings were created successfully."""
    try:
        cursor = iris_connector.cursor()
        
        # Count total tokens
        cursor.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings")
        total_tokens = cursor.fetchone()[0]
        
        # Count documents with tokens
        cursor.execute("SELECT COUNT(DISTINCT doc_id) FROM RAG.DocumentTokenEmbeddings")
        docs_with_tokens = cursor.fetchone()[0]
        
        # Test vector operations work
        vector_test_success = False
        if total_tokens > 0:
            try:
                # Test that we can use VECTOR_COSINE on the embedding
                cursor.execute("""
                    SELECT TOP 1 VECTOR_COSINE(token_embedding, token_embedding) as test_score
                    FROM RAG.DocumentTokenEmbeddings
                """)
                test_result = cursor.fetchone()
                if test_result and test_result[0] is not None:
                    vector_test_success = True
                    logger.info(f"✅ Vector operations test: {test_result[0]}")
            except Exception as e:
                logger.warning(f"Vector operation test failed: {e}")
        
        cursor.close()
        
        result = {
            "total_tokens": total_tokens,
            "documents_with_tokens": docs_with_tokens,
            "vector_operations_working": vector_test_success
        }
        
        logger.info(f"Token embeddings verification: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error verifying token embeddings: {e}")
        return {"error": str(e)}

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Populate ColBERT Token Embeddings in InterSystems IRIS.")
    parser.add_argument(
        "--doc_ids_file",
        type=str,
        default=None, # Explicitly set default to None
        help="Optional path to a file containing document IDs to process (one ID per line)."
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=1000,
        help="Maximum number of documents to process if --doc_ids_file is not provided. Default is 1000."
    )
    args = parser.parse_args()

    global _hf_model_cache # Ensure we're modifying the global cache
    _hf_model_cache = {}   # Clear cache at the start of main
    logger.info("🚀 Starting ColBERT Token Embeddings Population (Native VECTOR)...")
    if args.doc_ids_file:
        logger.info(f"Processing document IDs from file: {args.doc_ids_file}")
    else:
        logger.info(f"Processing up to {args.max_docs} documents needing embeddings (if no doc_ids_file specified).")

    try:
        # Get database connection
        iris_connector = get_iris_connection()
        
        # Check current state
        initial_state = verify_token_embeddings(iris_connector)
        
        # Only ask to continue if not using a doc_ids_file and if there are existing tokens
        if not args.doc_ids_file and initial_state.get("total_tokens", 0) > 0:
            logger.info(f"Found {initial_state['total_tokens']} existing token embeddings in the database.")
            try:
                user_input = input("Continue adding more token embeddings (for documents not yet processed)? (y/N): ")
                if user_input.lower() != 'y':
                    logger.info("Skipping token embeddings population as per user input.")
                    iris_connector.close()
                    return
            except EOFError: # Handle non-interactive environments (e.g., cron job)
                logger.info("No user input detected (EOFError). Proceeding with population if new documents are found or doc_ids_file is specified.")
                # In a non-interactive script, we'd typically proceed unless explicitly told not to.
                # If no new docs and no doc_ids_file, it will do nothing anyway.
        
        # Populate token embeddings
        processed, tokens_created = populate_all_token_embeddings(
            iris_connector,
            max_docs=args.max_docs,
            doc_ids_file=args.doc_ids_file
        )
        
        # Final verification
        final_state = verify_token_embeddings(iris_connector)
        
        logger.info("\n" + "="*60)
        logger.info("COLBERT TOKEN EMBEDDINGS POPULATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Documents processed: {processed}")
        logger.info(f"Tokens created: {tokens_created}")
        logger.info(f"Total token embeddings: {final_state.get('total_tokens', 0)}")
        logger.info(f"Documents with tokens: {final_state.get('documents_with_tokens', 0)}")
        logger.info(f"Vector operations working: {final_state.get('vector_operations_working', False)}")
        
        if final_state.get("total_tokens", 0) > 0 and final_state.get("vector_operations_working", False):
            logger.info("✅ ColBERT token embeddings population successful!")
            logger.info("🎯 ColBERT pipeline is now ready for enterprise-scale evaluation!")
        else:
            logger.warning("⚠️ Token embeddings created but vector operations may not be working")
        
        iris_connector.close()
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)

if __name__ == "__main__":
    main()