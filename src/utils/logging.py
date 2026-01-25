"""Logging utilities."""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch


def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    experiment_name: Optional[str] = None
) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        log_dir: Directory to save logs
        log_level: Logging level
        experiment_name: Name of the experiment
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("visual_reasoning")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if experiment_name:
        log_file = os.path.join(log_dir, f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    else:
        log_file = os.path.join(log_dir, f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def log_model_info(logger: logging.Logger, model: torch.nn.Module) -> None:
    """Log model information.
    
    Args:
        logger: Logger instance
        model: PyTorch model
    """
    from .device import get_model_size
    
    model_info = get_model_size(model)
    
    logger.info("Model Information:")
    logger.info(f"  Total parameters: {model_info['total_parameters']:,}")
    logger.info(f"  Trainable parameters: {model_info['trainable_parameters']:,}")
    logger.info(f"  Non-trainable parameters: {model_info['non_trainable_parameters']:,}")
    logger.info(f"  Total parameters (M): {model_info['total_parameters_millions']:.2f}")
    logger.info(f"  Trainable parameters (M): {model_info['trainable_parameters_millions']:.2f}")


def log_training_step(
    logger: logging.Logger,
    epoch: int,
    step: int,
    loss: float,
    learning_rate: float,
    **kwargs
) -> None:
    """Log training step information.
    
    Args:
        logger: Logger instance
        epoch: Current epoch
        step: Current step
        loss: Current loss
        learning_rate: Current learning rate
        **kwargs: Additional metrics to log
    """
    log_msg = f"Epoch {epoch}, Step {step}: Loss={loss:.4f}, LR={learning_rate:.6f}"
    
    for key, value in kwargs.items():
        if isinstance(value, (int, float)):
            log_msg += f", {key}={value:.4f}"
        else:
            log_msg += f", {key}={value}"
    
    logger.info(log_msg)


def log_evaluation_results(
    logger: logging.Logger,
    metrics: dict,
    epoch: Optional[int] = None
) -> None:
    """Log evaluation results.
    
    Args:
        logger: Logger instance
        metrics: Dictionary of metrics
        epoch: Current epoch (optional)
    """
    if epoch is not None:
        logger.info(f"Evaluation Results (Epoch {epoch}):")
    else:
        logger.info("Evaluation Results:")
    
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (int, float)):
            logger.info(f"  {metric_name}: {metric_value:.4f}")
        else:
            logger.info(f"  {metric_name}: {metric_value}")
