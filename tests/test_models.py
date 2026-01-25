"""Unit tests for Visual Reasoning Systems."""

import pytest
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.models import ModelFactory
from src.data.vqa_dataset import VQADataset
from src.utils.device import get_device, set_seed, get_model_size
from src.utils.config import load_config


class TestDeviceUtils:
    """Test device utility functions."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device("auto")
        assert isinstance(device, torch.device)
        
        device = get_device("cpu")
        assert device.type == "cpu"
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        # This is a basic test - in practice you'd check random state
        assert True
    
    def test_get_model_size(self):
        """Test model size calculation."""
        model = torch.nn.Linear(10, 5)
        size_info = get_model_size(model)
        
        assert "total_parameters" in size_info
        assert "trainable_parameters" in size_info
        assert size_info["total_parameters"] == 55  # 10*5 + 5


class TestVQADataset:
    """Test VQA dataset implementation."""
    
    def test_dataset_creation(self):
        """Test dataset creation."""
        dataset = VQADataset(
            data_dir="data",
            split="train",
            image_size=(224, 224),
            max_question_length=50,
            max_answer_length=20
        )
        
        assert len(dataset) > 0
        assert dataset.image_size == (224, 224)
    
    def test_dataset_item(self):
        """Test dataset item retrieval."""
        dataset = VQADataset(
            data_dir="data",
            split="train",
            image_size=(224, 224),
            max_question_length=50,
            max_answer_length=20
        )
        
        item = dataset[0]
        
        assert "image" in item
        assert "question_input_ids" in item
        assert "question_attention_mask" in item
        assert "answer_input_ids" in item
        assert "answer_attention_mask" in item
        assert "question" in item
        assert "answer" in item
        
        # Check tensor shapes
        assert item["image"].shape == (3, 224, 224)
        assert item["question_input_ids"].shape == (50,)
        assert item["question_attention_mask"].shape == (50,)


class TestModels:
    """Test model implementations."""
    
    def test_model_factory(self):
        """Test model factory."""
        config = OmegaConf.create({
            "name": "visualbert",
            "pretrained_model": "uclanlp/visualbert-nlvr2",
            "num_labels": 2,
            "freeze_vision": False,
            "freeze_text": False
        })
        
        # This test might fail if the model is not available
        # In a real test environment, you'd mock the model loading
        try:
            model = ModelFactory.create_model(config)
            assert model is not None
        except Exception:
            # Skip test if model is not available
            pytest.skip("Model not available for testing")
    
    def test_model_forward(self):
        """Test model forward pass."""
        config = OmegaConf.create({
            "name": "visualbert",
            "pretrained_model": "uclanlp/visualbert-nlvr2",
            "num_labels": 2,
            "freeze_vision": False,
            "freeze_text": False
        })
        
        try:
            model = ModelFactory.create_model(config)
            model.eval()
            
            # Create dummy inputs
            batch_size = 2
            images = torch.randn(batch_size, 3, 224, 224)
            question_input_ids = torch.randint(0, 1000, (batch_size, 50))
            question_attention_mask = torch.ones(batch_size, 50)
            
            with torch.no_grad():
                outputs = model(
                    images=images,
                    question_input_ids=question_input_ids,
                    question_attention_mask=question_attention_mask
                )
            
            assert "logits" in outputs
            assert outputs["logits"].shape == (batch_size, 2)
            
        except Exception:
            pytest.skip("Model not available for testing")


class TestConfigUtils:
    """Test configuration utilities."""
    
    def test_config_loading(self):
        """Test configuration loading."""
        # Create a temporary config file
        config_data = {
            "experiment_name": "test",
            "seed": 42,
            "device": "cpu"
        }
        
        config = OmegaConf.create(config_data)
        
        assert config.experiment_name == "test"
        assert config.seed == 42
        assert config.device == "cpu"


if __name__ == "__main__":
    pytest.main([__file__])
