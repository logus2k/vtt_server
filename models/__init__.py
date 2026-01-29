"""
Base class for VTT models.
"""

from abc import ABC, abstractmethod
from typing import Any
from PIL import Image


class BaseVTTModel(ABC):
    """Abstract base class for Video-to-Text models."""
    
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model = None
        self.processor = None
    
    @abstractmethod
    def load(self) -> None:
        """Load the model and processor."""
        pass
    
    @abstractmethod
    def generate(self, frames: list[Image.Image], prompt: str, fps: int) -> str:
        """Generate text from frames."""
        pass
    
    @abstractmethod
    def chat(
        self,
        message: str,
        image: Image.Image | None = None,
        history: str | None = None
    ) -> str:
        """Chat with optional image and conversation history."""
        pass
    
    @property
    def device(self):
        """Return model device."""
        if self.model is None:
            return None
        return self.model.device
