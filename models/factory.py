"""
Model factory for VTT models.
"""

from typing import Any
from .cosmos import CosmosModel
from .smolvlm2 import SmolVLM2Model
from . import BaseVTTModel


# Registry of available models
MODEL_REGISTRY: dict[str, type[BaseVTTModel]] = {
    "cosmos": CosmosModel,
    "smolvlm2": SmolVLM2Model,
}


def create_model(config: dict[str, Any]) -> BaseVTTModel:
    """
    Create a model instance from config.
    
    Args:
        config: Model configuration dict with 'name' key
        
    Returns:
        Instantiated model (not yet loaded)
        
    Raises:
        ValueError: If model name is not in registry
    """
    name = config.get("name", "").lower()
    
    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {name}. Available: {available}")
    
    model_class = MODEL_REGISTRY[name]
    return model_class(config)


def get_active_model_config(models: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Find the active model config from a list.
    
    Args:
        models: List of model configurations
        
    Returns:
        The config dict with active=True
        
    Raises:
        ValueError: If no active model or multiple active models
    """
    active_models = [m for m in models if m.get("active", False)]
    
    if len(active_models) == 0:
        raise ValueError("No active model found in config")
    
    if len(active_models) > 1:
        names = [m.get("name", "?") for m in active_models]
        raise ValueError(f"Multiple active models found: {names}. Only one should be active.")
    
    return active_models[0]
