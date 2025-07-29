#!/usr/bin/env python3
"""
Post-processing script to fix RAGAS results by converting 'answer' keys to 'response' keys.

This script allows us to fix the KeyError: 'response' issue without re-running
the expensive RAGAS evaluation by transforming the saved results.
"""

import json
import os
import shutil
from datetime import datetime
from typing import Dict, Any, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fix_pipeline_result_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fix a single pipeline result item by converting 'answer' to 'response'.
    
    Args:
        item: Dictionary containing a single pipeline result
        
    Returns:
        Fixed item with 'response' key instead of 'answer'
    """
    if 'answer' in item and 'response' not in item:
        item['response'] = item['answer']
        del item['answer']
        logger.debug("Converted 'answer' key to 'response' key in pipeline result item")
    
    return item


def fix_pipeline_results(pipeline_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Fix pipeline results by updating all items to use 'response' instead of 'answer'.
    
    Args:
        pipeline_results: List of pipeline result dictionaries
        
    Returns:
        Fixed pipeline results
    """
    fixed_results = []
    for item in pipeline_results:
        fixed_results.append(fix_pipeline_result_item(item.copy()))
    
    return fixed_results


def fix_ragas_results_file(input_file: str, output_file: str = None) -> str:
    """
    Fix RAGAS results file by converting 'answer' keys to 'response' keys.
    
    Args:
        input_file: Path to the input RAGAS results JSON file
        output_file: Path to the output file (if None, creates a _fixed version)
        
    Returns:
        Path to the fixed results file
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Create output filename if not provided
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_fixed.json"
    
    # Create backup of original file
    backup_file = f"{input_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(input_file, backup_file)
    logger.info(f"Created backup: {backup_file}")
    
    # Load the results
    logger.info(f"Loading results from: {input_file}")
    with open(input_file, 'r') as f:
        results = json.load(f)
    
    # Fix the results based on structure
    if 'pipeline_results' in results:
        # Comprehensive results format
        logger.info("Processing comprehensive results format")
        fixed_results = results.copy()
        for pipeline_name, pipeline_data in results['pipeline_results'].items():
            logger.info(f"Processing pipeline: {pipeline_name}")
            fixed_results['pipeline_results'][pipeline_name] = fix_pipeline_results(pipeline_data)
    else:
        # Simple results format (pipeline_name -> list of results)
        logger.info("Processing simple results format")
        fixed_results = {}
        for pipeline_name, pipeline_data in results.items():
            logger.info(f"Processing pipeline: {pipeline_name}")
            if isinstance(pipeline_data, list):
                fixed_results[pipeline_name] = fix_pipeline_results(pipeline_data)
            else:
                fixed_results[pipeline_name] = pipeline_data
    
    # Save the fixed results
    logger.info(f"Saving fixed results to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(fixed_results, f, indent=2)
    
    logger.info(f"Successfully fixed RAGAS results. Fixed file: {output_file}")
    return output_file


def fix_comprehensive_results_directory(results_dir: str) -> str:
    """
    Fix all RAGAS results in a comprehensive results directory.
    
    Args:
        results_dir: Path to the comprehensive results directory
        
    Returns:
        Path to the fixed results directory
    """
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    # Create fixed results directory
    fixed_dir = f"{results_dir}_fixed"
    if os.path.exists(fixed_dir):
        shutil.rmtree(fixed_dir)
    shutil.copytree(results_dir, fixed_dir)
    
    # Fix raw_results.json if it exists
    raw_results_file = os.path.join(fixed_dir, 'raw_results.json')
    if os.path.exists(raw_results_file):
        fix_ragas_results_file(raw_results_file, raw_results_file)
    
    logger.info(f"Fixed comprehensive results directory: {fixed_dir}")
    return fixed_dir


def main():
    """Main function to fix RAGAS results."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix RAGAS results by converting answer keys to response keys')
    parser.add_argument('input_path', help='Path to RAGAS results file or directory')
    parser.add_argument('--output', '-o', help='Output path (optional)')
    parser.add_argument('--in-place', action='store_true', help='Fix files in place (creates backup)')
    
    args = parser.parse_args()
    
    try:
        if os.path.isfile(args.input_path):
            # Fix single file
            output_file = args.input_path if args.in_place else args.output
            output_file = fix_ragas_results_file(args.input_path, output_file)
            print(f"Fixed results saved to: {output_file}")
        elif os.path.isdir(args.input_path):
            # Fix comprehensive results directory
            if args.in_place:
                # Fix in place by creating backup and replacing
                backup_dir = f"{args.input_path}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.copytree(args.input_path, backup_dir)
                logger.info(f"Created backup directory: {backup_dir}")
                
                raw_results_file = os.path.join(args.input_path, 'raw_results.json')
                if os.path.exists(raw_results_file):
                    fix_ragas_results_file(raw_results_file, raw_results_file)
                print(f"Fixed results directory in place: {args.input_path}")
            else:
                output_dir = fix_comprehensive_results_directory(args.input_path)
                print(f"Fixed results directory: {output_dir}")
        else:
            raise FileNotFoundError(f"Path not found: {args.input_path}")
            
    except Exception as e:
        logger.error(f"Error fixing RAGAS results: {e}")
        raise


if __name__ == "__main__":
    main()