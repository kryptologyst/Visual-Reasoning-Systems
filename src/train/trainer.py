"""Training utilities and trainer class."""

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.device import get_device, format_time
from ..utils.logging import setup_logging, log_training_step, log_model_info


class Trainer:
    """Trainer class for VQA models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[str] = None,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        experiment_name: Optional[str] = None,
        **kwargs
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to use
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
            **kwargs: Additional arguments
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = get_device(device)
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=1e-4,
                weight_decay=1e-4
            )
        else:
            self.optimizer = optimizer
        
        # Setup scheduler
        self.scheduler = scheduler
        
        # Setup logging
        self.logger = setup_logging(log_dir, experiment_name=experiment_name)
        
        # Setup directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Log model info
        log_model_info(self.logger, self.model)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dict: Training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = len(self.train_dataloader)
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.get("loss")
            
            if loss is None:
                # If no loss is returned, compute it manually
                # This is a simplified approach for demonstration
                logits = outputs["logits"]
                # For binary classification, create dummy labels
                labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
                loss = nn.CrossEntropyLoss()(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss / (batch_idx + 1):.4f}"
            })
            
            # Log training step
            if self.global_step % 100 == 0:
                log_training_step(
                    self.logger,
                    epoch=epoch,
                    step=self.global_step,
                    loss=loss.item(),
                    learning_rate=self.optimizer.param_groups[0]['lr']
                )
        
        avg_loss = total_loss / num_batches
        return {"train_loss": avg_loss}
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dict: Validation metrics
        """
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation", leave=False):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.get("loss")
                
                if loss is None:
                    # Compute loss manually
                    logits = outputs["logits"]
                    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
                    loss = nn.CrossEntropyLoss()(logits, labels)
                
                total_loss += loss.item()
                
                # Compute accuracy
                predictions = torch.argmax(outputs["logits"], dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)
        
        avg_loss = total_loss / len(self.val_dataloader)
        accuracy = correct_predictions / total_predictions
        
        return {
            "val_loss": avg_loss,
            "val_accuracy": accuracy
        }
    
    def train(
        self,
        epochs: int,
        save_every: int = 1000,
        validate_every: int = 500,
        **kwargs
    ) -> Dict[str, Any]:
        """Train the model.
        
        Args:
            epochs: Number of epochs to train
            save_every: Save checkpoint every N steps
            validate_every: Validate every N steps
            **kwargs: Additional arguments
            
        Returns:
            Dict: Training history
        """
        self.logger.info(f"Starting training for {epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Training samples: {len(self.train_dataloader.dataset)}")
        
        if self.val_dataloader is not None:
            self.logger.info(f"Validation samples: {len(self.val_dataloader.dataset)}")
        
        start_time = time.time()
        training_history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": []
        }
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            training_history["train_loss"].append(train_metrics["train_loss"])
            
            # Validate
            if epoch % validate_every == 0 or epoch == epochs - 1:
                val_metrics = self.validate(epoch)
                training_history["val_loss"].append(val_metrics.get("val_loss", 0.0))
                training_history["val_accuracy"].append(val_metrics.get("val_accuracy", 0.0))
                
                # Save best model
                if val_metrics.get("val_loss", float('inf')) < self.best_val_loss:
                    self.best_val_loss = val_metrics["val_loss"]
                    self.save_checkpoint("best_model.pt")
            
            # Save checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
        
        # Save final checkpoint
        self.save_checkpoint("final_model.pt")
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {format_time(total_time)}")
        
        return training_history
    
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint.
        
        Args:
            filename: Checkpoint filename
        """
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
    
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
