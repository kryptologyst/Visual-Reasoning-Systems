"""Evaluation script for Visual Reasoning Systems."""

import argparse
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from src.data.vqa_dataset import VQADataset
from src.data import create_dataloader, collate_fn
from src.models import ModelFactory
from src.eval.evaluator import Evaluator
from src.utils.device import get_device, set_seed
from src.utils.config import load_config, create_directories


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Visual Reasoning System")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.device != "auto":
        config.device = args.device
    if args.seed != 42:
        config.seed = args.seed
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Create directories
    create_directories(config)
    
    # Setup device
    device = get_device(config.device)
    config.device = str(device)
    
    print(f"Using device: {device}")
    print(f"Configuration: {OmegaConf.to_yaml(config)}")
    
    # Create dataset
    test_dataset = VQADataset(
        data_dir=config.data_dir,
        split="test",
        image_size=tuple(config.data.image_size),
        max_question_length=config.data.max_question_length,
        max_answer_length=config.data.max_answer_length,
        augmentation=False,  # No augmentation for evaluation
        normalize=config.data.get("normalize", True)
    )
    
    # Create data loader
    test_dataloader = create_dataloader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn
    )
    
    # Create model
    model = ModelFactory.create_model(config.model)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        dataloader=test_dataloader,
        device=device,
        output_dir=config.evaluation.output_dir,
        metrics=config.evaluation.metrics
    )
    
    # Run evaluation
    results = evaluator.evaluate()
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 50)
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    print(f"\nResults saved to: {config.evaluation.output_dir}")


if __name__ == "__main__":
    main()
