"""Configuration management utilities."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        DictConfig: Loaded configuration
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, save_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration to save
        save_path: Path to save configuration
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        OmegaConf.save(config, f)


def merge_configs(base_config: DictConfig, override_config: DictConfig) -> DictConfig:
    """Merge two configurations.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override with
        
    Returns:
        DictConfig: Merged configuration
    """
    return OmegaConf.merge(base_config, override_config)


def resolve_config(config: DictConfig) -> DictConfig:
    """Resolve configuration variables.
    
    Args:
        config: Configuration to resolve
        
    Returns:
        DictConfig: Resolved configuration
    """
    return OmegaConf.to_container(config, resolve=True)


def get_config_value(config: DictConfig, key: str, default: Any = None) -> Any:
    """Get configuration value with default fallback.
    
    Args:
        config: Configuration object
        key: Configuration key (supports dot notation)
        default: Default value if key not found
        
    Returns:
        Any: Configuration value or default
    """
    try:
        return OmegaConf.select(config, key)
    except:
        return default


def create_directories(config: DictConfig) -> None:
    """Create necessary directories from configuration.
    
    Args:
        config: Configuration object
    """
    directories = [
        config.get("data_dir", "data"),
        config.get("checkpoint_dir", "checkpoints"),
        config.get("log_dir", "logs"),
        config.get("assets_dir", "assets"),
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
