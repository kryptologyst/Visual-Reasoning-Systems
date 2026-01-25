"""Model factory and base classes."""

from typing import Any, Dict, Optional, Type

import torch
import torch.nn as nn
from omegaconf import DictConfig


class BaseVQAModel(nn.Module):
    """Base class for VQA models."""
    
    def __init__(self, config: DictConfig):
        """Initialize base model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
    
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass - to be implemented by subclasses."""
        raise NotImplementedError
    
    def generate_answer(self, *args, **kwargs) -> torch.Tensor:
        """Generate answer - to be implemented by subclasses."""
        raise NotImplementedError


class ModelFactory:
    """Factory class for creating VQA models."""
    
    _models = {}
    
    @classmethod
    def register_model(cls, name: str, model_class: Type[BaseVQAModel]):
        """Register a model class.
        
        Args:
            name: Model name
            model_class: Model class
        """
        cls._models[name] = model_class
    
    @classmethod
    def create_model(cls, config: DictConfig) -> BaseVQAModel:
        """Create a model instance.
        
        Args:
            config: Model configuration
            
        Returns:
            BaseVQAModel: Model instance
        """
        model_name = config.get("name", "visualbert")
        
        if model_name not in cls._models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_class = cls._models[model_name]
        return model_class(config)
    
    @classmethod
    def list_models(cls) -> list:
        """List available models.
        
        Returns:
            list: List of available model names
        """
        return list(cls._models.keys())


# Register models
from .visualbert import VisualBERTModel
from .clip_vqa import CLIPVQAModel

ModelFactory.register_model("visualbert", VisualBERTModel)
ModelFactory.register_model("clip_vqa", CLIPVQAModel)
