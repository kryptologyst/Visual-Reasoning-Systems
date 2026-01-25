#!/usr/bin/env python3
"""
Project 568: Visual Reasoning Systems - Modernized Implementation

This is a simple example script demonstrating the modernized Visual Reasoning Systems framework.
For full functionality, use the main training script or interactive demo.

Usage:
    python 0568.py

For training:
    python scripts/train.py --config configs/config.yaml

For interactive demo:
    streamlit run demo/streamlit_demo.py
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Import from the modernized framework
from src.models import ModelFactory
from src.data.vqa_dataset import VQADataset
from src.utils.device import get_device, set_seed
from src.utils.config import load_config


def main():
    """Simple example demonstrating the Visual Reasoning Systems framework."""
    print("Visual Reasoning Systems - Modernized Implementation")
    print("=" * 60)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Load configuration
    try:
        config = load_config("configs/config.yaml")
        print("✓ Configuration loaded successfully")
    except FileNotFoundError:
        print("⚠ Configuration file not found, using defaults")
        config = None
    
    # Setup device
    device = get_device("auto")
    print(f"✓ Using device: {device}")
    
    # Create a simple dataset for demonstration
    print("\nCreating synthetic dataset...")
    dataset = VQADataset(
        data_dir="data",
        split="train",
        image_size=(224, 224),
        max_question_length=50,
        max_answer_length=20
    )
    print(f"✓ Dataset created with {len(dataset)} samples")
    
    # Test dataset item
    sample = dataset[0]
    print(f"✓ Sample question: '{sample['question']}'")
    print(f"✓ Sample answer: '{sample['answer']}'")
    print(f"✓ Image shape: {sample['image'].shape}")
    
    # Create model (if configuration is available)
    if config:
        try:
            print("\nCreating model...")
            model = ModelFactory.create_model(config.model)
            model.to(device)
            model.eval()
            
            # Get model info
            from src.utils.device import get_model_size
            model_info = get_model_size(model)
            print(f"✓ Model created: {model_info['total_parameters_millions']:.1f}M parameters")
            
            # Test model with sample data
            print("\nTesting model...")
            with torch.no_grad():
                # Prepare inputs
                image = sample["image"].unsqueeze(0).to(device)
                question_input_ids = sample["question_input_ids"].unsqueeze(0).to(device)
                question_attention_mask = sample["question_attention_mask"].unsqueeze(0).to(device)
                
                # Forward pass
                outputs = model(
                    images=image,
                    question_input_ids=question_input_ids,
                    question_attention_mask=question_attention_mask
                )
                
                # Get prediction
                logits = outputs["logits"]
                predicted_class = torch.argmax(logits, dim=-1).item()
                confidence = torch.softmax(logits, dim=-1)[0][predicted_class].item()
                
                answer_map = {0: "No", 1: "Yes"}
                predicted_answer = answer_map.get(predicted_class, "Unknown")
                
                print(f"✓ Question: '{sample['question']}'")
                print(f"✓ Predicted answer: {predicted_answer}")
                print(f"✓ Confidence: {confidence:.2%}")
                
        except Exception as e:
            print(f"⚠ Model creation failed: {e}")
            print("This is expected if the model dependencies are not installed.")
    
    print("\n" + "=" * 60)
    print("Framework demonstration complete!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run training: python scripts/train.py")
    print("3. Launch demo: streamlit run demo/streamlit_demo.py")
    print("4. Read documentation: README.md")


if __name__ == "__main__":
    main()
