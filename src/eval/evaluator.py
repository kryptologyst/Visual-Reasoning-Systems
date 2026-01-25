"""Evaluation utilities and evaluator class."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.device import get_device
from ..utils.logging import setup_logging, log_evaluation_results


class Evaluator:
    """Evaluator class for VQA models."""
    
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: Optional[str] = None,
        output_dir: str = "assets/evaluation",
        metrics: Optional[List[str]] = None,
        **kwargs
    ):
        """Initialize evaluator.
        
        Args:
            model: Model to evaluate
            dataloader: Data loader for evaluation
            device: Device to use
            output_dir: Directory to save results
            metrics: List of metrics to compute
            **kwargs: Additional arguments
        """
        self.model = model
        self.dataloader = dataloader
        self.device = get_device(device)
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup metrics
        self.metrics = metrics or ["accuracy", "f1"]
        
        # Setup logging
        self.logger = setup_logging("logs", experiment_name="evaluation")
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the model.
        
        Returns:
            Dict: Evaluation results
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_losses = []
        
        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Evaluating"):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                outputs = self.model(**batch)
                
                # Get predictions
                logits = outputs["logits"]
                predictions = torch.argmax(logits, dim=-1)
                
                # Create dummy labels for demonstration
                labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
                
                # Compute loss
                loss = nn.CrossEntropyLoss()(logits, labels)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_losses.append(loss.item())
        
        # Compute metrics
        results = self._compute_metrics(all_predictions, all_labels, all_losses)
        
        # Log results
        log_evaluation_results(self.logger, results)
        
        # Save results
        self._save_results(results, all_predictions, all_labels)
        
        return results
    
    def _compute_metrics(
        self,
        predictions: List[int],
        labels: List[int],
        losses: List[float]
    ) -> Dict[str, float]:
        """Compute evaluation metrics.
        
        Args:
            predictions: Model predictions
            labels: Ground truth labels
            losses: Loss values
            
        Returns:
            Dict: Computed metrics
        """
        import numpy as np
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        results = {
            "loss": np.mean(losses),
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average="weighted"),
            "precision": precision_score(labels, predictions, average="weighted"),
            "recall": recall_score(labels, predictions, average="weighted")
        }
        
        return results
    
    def _save_results(
        self,
        results: Dict[str, float],
        predictions: List[int],
        labels: List[int]
    ) -> None:
        """Save evaluation results.
        
        Args:
            results: Evaluation results
            predictions: Model predictions
            labels: Ground truth labels
        """
        # Save metrics
        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save predictions
        predictions_path = self.output_dir / "predictions.json"
        predictions_data = {
            "predictions": predictions,
            "labels": labels
        }
        with open(predictions_path, 'w') as f:
            json.dump(predictions_data, f, indent=2)
        
        self.logger.info(f"Results saved to {self.output_dir}")
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device.
        
        Args:
            batch: Input batch
            
        Returns:
            Dict: Batch moved to device
        """
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        
        return device_batch
