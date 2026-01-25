"""Main training script for Visual Reasoning Systems."""

import argparse
import os
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from src.data.vqa_dataset import VQADataset
from src.data import create_dataloader, collate_fn
from src.models import ModelFactory
from src.train.trainer import Trainer
from src.eval.evaluator import Evaluator
from src.utils.device import get_device, set_seed
from src.utils.config import load_config, create_directories


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Visual Reasoning System")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
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
    
    # Create datasets
    train_dataset = VQADataset(
        data_dir=config.data_dir,
        split="train",
        image_size=tuple(config.data.image_size),
        max_question_length=config.data.max_question_length,
        max_answer_length=config.data.max_answer_length,
        augmentation=config.data.get("augmentation", True),
        normalize=config.data.get("normalize", True)
    )
    
    val_dataset = VQADataset(
        data_dir=config.data_dir,
        split="val",
        image_size=tuple(config.data.image_size),
        max_question_length=config.data.max_question_length,
        max_answer_length=config.data.max_answer_length,
        augmentation=False,  # No augmentation for validation
        normalize=config.data.get("normalize", True)
    )
    
    # Create data loaders
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn
    )
    
    val_dataloader = create_dataloader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn
    )
    
    # Create model
    model = ModelFactory.create_model(config.model)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        checkpoint_dir=config.checkpoint_dir,
        log_dir=config.log_dir,
        experiment_name=config.experiment_name
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train model
    training_history = trainer.train(
        epochs=config.training.epochs,
        save_every=config.training.save_every,
        validate_every=config.training.validate_every
    )
    
    # Evaluate final model
    evaluator = Evaluator(
        model=model,
        dataloader=val_dataloader,
        device=device,
        output_dir=config.evaluation.output_dir,
        metrics=config.evaluation.metrics
    )
    
    final_results = evaluator.evaluate()
    print(f"Final evaluation results: {final_results}")


if __name__ == "__main__":
    main()
