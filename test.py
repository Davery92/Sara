#!/usr/bin/env python3
"""
Test script to verify loading of the local cross-encoder model in SafeTensors format.
This script specifically checks for issues with the SafeTensors format.
"""

import os
import sys
import torch
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from safetensors.torch import load_file
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("safetensors-test")

# Path to the local model
MODEL_PATH = "/home/david/Sara/ms-marco-MiniLM-L-6-v2"

def inspect_safetensors_files():
    """Inspect the SafeTensors files in the model directory"""
    safetensors_path = os.path.join(MODEL_PATH, "model.safetensors")
    if not os.path.exists(safetensors_path):
        logger.warning(f"No model.safetensors file found at: {safetensors_path}")
        
        # Look for any .safetensors files
        safetensors_files = list(Path(MODEL_PATH).glob("*.safetensors"))
        if safetensors_files:
            logger.info(f"Found {len(safetensors_files)} SafeTensors files:")
            for sf in safetensors_files:
                logger.info(f"  - {sf.name} (size: {sf.stat().st_size/1024/1024:.2f} MB)")
        else:
            logger.warning("No SafeTensors files found in the model directory")
            return False
    else:
        logger.info(f"Found model.safetensors (size: {os.path.getsize(safetensors_path)/1024/1024:.2f} MB)")
    
    # Try to directly load each SafeTensors file
    logger.info("Attempting to load SafeTensors files directly:")
    safetensors_files = list(Path(MODEL_PATH).glob("*.safetensors"))
    
    for sf in safetensors_files:
        try:
            logger.info(f"Loading {sf.name}...")
            tensors = load_file(sf)
            logger.info(f"Successfully loaded {sf.name} - contains {len(tensors)} tensors")
        except Exception as e:
            logger.error(f"Error loading {sf.name}: {e}")
    
    return True

def test_model_loading():
    """Test if the model can be loaded correctly"""
    logger.info(f"Testing model loading from: {MODEL_PATH}")
    
    # Check if the directory exists
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model directory does not exist: {MODEL_PATH}")
        return False
    
    # First inspect the directory contents
    logger.info("Inspecting directory contents:")
    files = os.listdir(MODEL_PATH)
    for file in sorted(files):
        file_path = os.path.join(MODEL_PATH, file)
        if os.path.isfile(file_path):
            logger.info(f"  - {file} (size: {os.path.getsize(file_path)/1024/1024:.2f} MB)")
    
    # Inspect SafeTensors files
    inspect_safetensors_files()
    
    # Try loading with explicit safetensors option
    try:
        logger.info("Attempting to load with explicit use_safetensors=True...")
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            use_safetensors=True
        )
        logger.info("Model loaded successfully with SafeTensors!")
        return True
    except Exception as e:
        logger.error(f"Failed to load with use_safetensors=True: {e}")
    
    # Try loading with explicit pytorch option
    try:
        logger.info("Attempting to load with explicit use_safetensors=False...")
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            use_safetensors=False
        )
        logger.info("Model loaded successfully with PyTorch format!")
        return True
    except Exception as e:
        logger.error(f"Failed to load with use_safetensors=False: {e}")
    
    return False

if __name__ == "__main__":
    print("\n" + "="*50)
    print("CROSS-ENCODER SAFETENSORS TEST")
    print("="*50 + "\n")
    
    if test_model_loading():
        print("\n✅ Model loaded successfully!")
    else:
        print("\n❌ Failed to load the model. Check the errors above.")
    
    print("\n" + "="*50)