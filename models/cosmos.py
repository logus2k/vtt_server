"""
Cosmos Reason 2 (Qwen3-VL) model implementation.
"""

import torch
import transformers
from typing import Any
from PIL import Image

from . import BaseVTTModel

# Constants
PIXELS_PER_TOKEN = 32**2
MIN_VISION_TOKENS = 256
MAX_VISION_TOKENS = 8192


class CosmosModel(BaseVTTModel):
    """Cosmos Reason 2 model wrapper."""
    
    def load(self) -> None:
        """Load Cosmos model and processor."""
        model_path = self.config["path"]
        dtype = getattr(torch, self.config.get("dtype", "float16"))
        attn_impl = self.config.get("attn_implementation", "sdpa")
        
        self.model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=dtype,
            device_map="auto",
            attn_implementation=attn_impl,
            local_files_only=True
        )
        
        self.processor = transformers.Qwen3VLProcessor.from_pretrained(
            model_path,
            local_files_only=True
        )
        
        # Vision token limits
        self.processor.image_processor.size = {
            "shortest_edge": MIN_VISION_TOKENS * PIXELS_PER_TOKEN,
            "longest_edge": MAX_VISION_TOKENS * PIXELS_PER_TOKEN,
        }
        self.processor.video_processor.size = {
            "shortest_edge": MIN_VISION_TOKENS * PIXELS_PER_TOKEN,
            "longest_edge": MAX_VISION_TOKENS * PIXELS_PER_TOKEN,
        }
    
    def generate(self, frames: list[Image.Image], prompt: str, fps: int) -> str:
        """Generate text from video frames."""
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": prompt}
            ]}
        ]
        
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            fps=fps,
        )
        inputs = inputs.to(self.model.device)
        
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.get("max_new_tokens", 512),
                use_cache=True
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        
        return output_text[0]
    
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
        content.append({"type": "text", "text": message})
        
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        ]
        
        # Add history as context if provided
        if history:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": f"Conversation history:\n{history}"}]
            })
        
        messages.append({"role": "user", "content": content})
        
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.get("max_new_tokens", 512),
                use_cache=True
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        
        return output_text[0]
