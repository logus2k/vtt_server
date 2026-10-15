"""
Video-to-Text Inference Server
Socket.IO server with binary frame streaming and continuous inference.
Supports multiple models via factory pattern.
"""

import asyncio
import io
import json
import time
from collections import deque
from pathlib import Path

import socketio
import uvicorn
from PIL import Image

from models.factory import create_model, get_active_model_config
from models import BaseVTTModel
from models.memory import SessionMemory, create_memory_from_config

# Load configuration
CONFIG_PATH = Path(__file__).parent / "config.json"
with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)


class FrameBuffer:
    """Thread-safe rolling buffer for camera frames."""
    
    def __init__(self, maxlen: int):
        self.buffer = deque(maxlen=maxlen)
        self.lock = asyncio.Lock()
    
    async def add(self, frame_bytes: bytes, timestamp: float):
        async with self.lock:
            self.buffer.append((frame_bytes, timestamp))
    
    async def get_frames(self) -> list[Image.Image]:
        async with self.lock:
            frames = []
            for frame_bytes, _ in self.buffer:
                img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
                frames.append(img)
            return frames
    
    async def count(self) -> int:
        async with self.lock:
            return len(self.buffer)
    
    async def clear(self):
        async with self.lock:
            self.buffer.clear()


class VTTServer:
    """Video-to-Text inference server."""
    
    def __init__(self):
        self.sio = socketio.AsyncServer(
            async_mode="asgi",
            cors_allowed_origins="*",
            max_http_buffer_size=10 * 1024 * 1024,
            ping_timeout=120,
            ping_interval=30
        )
        self.app = socketio.ASGIApp(self.sio)
        
        # Frame buffers per camera
        buffer_size = CONFIG["inference"]["buffer_size"]
        self.buffers = {
            "front": FrameBuffer(buffer_size),
            "rear": FrameBuffer(buffer_size)
        }

        # Continuous inference state
        self.inference_tasks: dict[str, asyncio.Task | None] = {
            "front": None,
            "rear": None
        }
        self.inference_prompts = {
            "front": CONFIG["inference"]["default_prompt"],
            "rear": CONFIG["inference"]["default_prompt"]
        }
        
        # Model (loaded on startup)
        self.model: BaseVTTModel | None = None
        self.model_config: dict | None = None
        
        # Memory (initialized after model load)
        self.memory: SessionMemory | None = None
        
        self._register_events()
    
    def _register_events(self):
        @self.sio.event
        async def connect(sid, environ):
            print(f"[CONNECT] {sid}")
            # Send current inference state and model info
            memory_enabled = self.memory.config.enabled if self.memory else False
            await self.sio.emit("status", {
                "front": self.inference_tasks["front"] is not None,
                "rear": self.inference_tasks["rear"] is not None,
                "model": self.model_config.get("name") if self.model_config else None,
                "memory": memory_enabled
            }, to=sid)
        
        @self.sio.event
        async def disconnect(sid):
            print(f"[DISCONNECT] {sid}")
            # Clear session memory
            if self.memory:
                await self.memory.clear_session(sid)
        
        @self.sio.event
        async def frame(sid, data):
            """
            Receive binary frame.
            Expected format: {camera: "front"|"rear", image: <binary JPEG>}
            """
            camera = data.get("camera")
            image_bytes = data.get("image")
            
            if camera not in self.buffers:
                await self.sio.emit("error", {"message": f"Unknown camera: {camera}"}, to=sid)
                return
            
            if not isinstance(image_bytes, bytes):
                await self.sio.emit("error", {"message": "Image must be binary"}, to=sid)
                return
            
            await self.buffers[camera].add(image_bytes, time.time())
        
        @self.sio.event
        async def start_inference(sid, data):
            """
            Start continuous inference loop.
            Expected format: {camera: "front"|"rear", prompt?: string}
            """
            camera = data.get("camera")
            prompt = data.get("prompt", CONFIG["inference"]["default_prompt"])
            
            if camera not in self.buffers:
                await self.sio.emit("error", {"message": f"Unknown camera: {camera}"}, to=sid)
                return
            
            # Stop existing task if running
            task = self.inference_tasks[camera]
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            # Store prompt and start loop
            self.inference_prompts[camera] = prompt
            self.inference_tasks[camera] = asyncio.create_task(
                self._continuous_inference_loop(camera)
            )
            
            print(f"[INFERENCE] Started for {camera}: {prompt}")
            await self.sio.emit("inference_started", {"camera": camera, "prompt": prompt})
        
        @self.sio.event
        async def stop_inference(sid, data):
            """
            Stop continuous inference loop.
            Expected format: {camera: "front"|"rear"}
            """
            camera = data.get("camera")
            
            if camera not in self.buffers:
                await self.sio.emit("error", {"message": f"Unknown camera: {camera}"}, to=sid)
                return
            
            task = self.inference_tasks[camera]
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                self.inference_tasks[camera] = None
                
                print(f"[INFERENCE] Stopped for {camera}")
                await self.sio.emit("inference_stopped", {"camera": camera})
            else:
                await self.sio.emit("error", {"message": f"Inference not running for {camera}"}, to=sid)
        
        @self.sio.event
        async def chat(sid, data):
            """
            Chat with the model, optionally with an image.
            Expected format: {message: string, image?: bytes}
            """
            message = data.get("message")
            image_bytes = data.get("image")
            
            if not message:
                await self.sio.emit("error", {"message": "Message is required"}, to=sid)
                return
            
            try:
                start_time = time.perf_counter()
                
                # Parse image if provided
                image = None
                if image_bytes and isinstance(image_bytes, bytes):
                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                
                # Get conversation history
                history = None
                if self.memory:
                    history = await self.memory.get_preamble(sid)
                    await self.memory.add_user_message(sid, message)
                
                # Run chat inference in executor
                loop = asyncio.get_event_loop()
                if self.model is None:
                    raise RuntimeError("Model not loaded")
                _model = self.model  # capture for lambda
                text = await loop.run_in_executor(
                    None,
                    lambda m=_model: m.chat(message, image, history)
                )
                
                # Store assistant response
                if self.memory:
                    await self.memory.add_assistant_message(sid, text)
                
                latency_ms = (time.perf_counter() - start_time) * 1000
                
                await self.sio.emit("chat_response", {
                    "text": text,
                    "latency_ms": round(latency_ms, 1)
                }, to=sid)
                
                print(f"[CHAT] {latency_ms:.0f}ms | {'with image' if image else 'text only'} | memory={'on' if history else 'off'}")
                
            except Exception as e:
                print(f"[ERROR] Chat: {e}")
                await self.sio.emit("error", {"message": str(e)}, to=sid)
        
        @self.sio.event
        async def clear_memory(sid, data):
            """Clear chat memory for this session."""
            if self.memory:
                await self.memory.clear_session(sid)
                await self.sio.emit("memory_cleared", {}, to=sid)
                print(f"[MEMORY] Cleared for {sid}")
    
    async def _continuous_inference_loop(self, camera: str):
        """Run inference continuously until cancelled."""
        fps = CONFIG["inference"].get("fps", 4)
        
        while True:
            try:
                # Wait for buffer to have frames
                frame_count = await self.buffers[camera].count()
                if frame_count == 0:
                    await asyncio.sleep(0.1)
                    continue
                
                # Get frames and run inference
                start_time = time.perf_counter()
                frames = await self.buffers[camera].get_frames()
                prompt = self.inference_prompts[camera]
                
                # Run inference in executor
                loop = asyncio.get_event_loop()
                if self.model is None:
                    raise RuntimeError("Model not loaded")
                _model = self.model  # capture for lambda
                text = await loop.run_in_executor(
                    None,
                    lambda m=_model: m.generate(frames, prompt, fps)
                )
                
                latency_ms = (time.perf_counter() - start_time) * 1000
                
                # Broadcast to all clients
                await self.sio.emit("description", {
                    "camera": camera,
                    "text": text,
                    "latency_ms": round(latency_ms, 1),
                    "frame_count": len(frames)
                })
                
                print(f"[{camera.upper()}] {latency_ms:.0f}ms | {len(frames)} frames")
                
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"[ERROR] Inference loop {camera}: {e}")
                await asyncio.sleep(1)
    
    def load_model(self):
        """Load the active model from config."""
        # Find active model config
        self.model_config = get_active_model_config(CONFIG["models"])
        model_name = self.model_config.get("name", "unknown")
        
        print(f"[MODEL] Loading {model_name} from {self.model_config['path']}...")
        start = time.perf_counter()
        
        # Create and load model
        self.model = create_model(self.model_config)
        self.model.load()
        
        # Initialize memory from model config
        self.memory = create_memory_from_config(self.model_config)
        memory_cfg = self.memory.config
        memory_status = "enabled" if memory_cfg.enabled else "disabled"
        
        print(f"[MODEL] {model_name} loaded in {time.perf_counter() - start:.1f}s")
        print(f"[MEMORY] {memory_status} (max_tokens={memory_cfg.max_context_tokens})")
    
    def run(self):
        """Start the server."""
        self.load_model()
        
        host = CONFIG["server"]["host"]
        port = CONFIG["server"]["port"]
        print(f"[SERVER] Starting on {host}:{port}")
        
        uvicorn.run(self.app, host=host, port=port)


if __name__ == "__main__":
    server = VTTServer()
    server.run()
