"""
SmolVLM2 model implementation.
"""

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from typing import Any
from PIL import Image

from . import BaseVTTModel


class SmolVLM2Model(BaseVTTModel):
    """SmolVLM2 model wrapper."""
    
    def load(self) -> None:
        """Load SmolVLM2 model and processor."""
        model_path = self.config["path"]
        dtype = getattr(torch, self.config.get("dtype", "bfloat16"))
        attn_impl = self.config.get("attn_implementation", "flash_attention_2")
        
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True
        )
        
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=dtype,
            _attn_implementation=attn_impl,
            local_files_only=True
        ).to("cuda")
    
    def generate(self, frames: list[Image.Image], prompt: str, fps: int) -> str:
        """Generate text from video frames."""
        # SmolVLM2 handles multiple images
        content = []
        for frame in frames:
            content.append({"type": "image", "image": frame})
        content.append({"type": "text", "text": prompt})
        
        messages = [
            {"role": "user", "content": content}
        ]
        
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)
        
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.config.get("max_new_tokens", 256),
            )
        
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        
        # Extract assistant response
        full_text = generated_texts[0]
        if "Assistant:" in full_text:
            return full_text.split("Assistant:")[-1].strip()
        return full_text.strip()
    
    def chat(
        self,
        message: str,
        image: Image.Image | None = None,
        history: str | None = None
    ) -> str:
        """Chat with optional image and conversation history."""
        content = []
        if image:
            content.append({"type": "image", "image": image})
        
        # Prepend history to message if provided
        if history:
            full_message = f"Conversation history:\n{history}\n\nCurrent message: {message}"
        else:
            full_message = message
        
        content.append({"type": "text", "text": full_message})
        
        messages = [
            {"role": "user", "content": content}
        ]
        
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)
        
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.config.get("max_new_tokens", 256),
            )
        
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        
        # Extract assistant response
        full_text = generated_texts[0]
        if "Assistant:" in full_text:
            return full_text.split("Assistant:")[-1].strip()
        return full_text.strip()
