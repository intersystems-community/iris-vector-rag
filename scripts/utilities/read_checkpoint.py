#!/usr/bin/env python3
"""
Read ingestion checkpoint to understand the structure.
"""

import pickle
import os
from pathlib import Path

checkpoint_file = Path("ingestion_checkpoint.pkl")

if checkpoint_file.exists():
    try:
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
        
        print("Checkpoint structure:")
        print(f"Type: {type(checkpoint)}")
        
        if hasattr(checkpoint, '__dict__'):
            print("Attributes:")
            for attr, value in checkpoint.__dict__.items():
                print(f"  {attr}: {value} ({type(value)})")
        elif isinstance(checkpoint, dict):
            print("Dictionary contents:")
            for key, value in checkpoint.items():
                print(f"  {key}: {value} ({type(value)})")
        else:
            print(f"Content: {checkpoint}")
            
    except Exception as e:
        print(f"Error reading checkpoint: {e}")
else:
    print("No checkpoint file found")