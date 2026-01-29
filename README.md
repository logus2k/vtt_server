# VTT Server

A real-time Video-to-Text inference server using Socket.IO for streaming camera frames and generating text descriptions via multimodal vision-language models.

## Features

- **Real-time video inference** — Continuous frame streaming with on-demand or continuous inference
- **Multi-model support** — Factory pattern supporting multiple VLM backends (Cosmos Reason 2, SmolVLM2)
- **Chat with vision** — Send images with text queries for visual Q&A
- **Session memory** — Configurable conversation history for multi-turn chat
- **Dual camera support** — Front and back camera buffers for robotics applications
- **Web client included** — Browser-based test interface

## Supported Models

| Model | Description | VRAM | Precision |
|-------|-------------|------|-----------|
| [Cosmos Reason 2](https://huggingface.co/nvidia/Cosmos-Reason1-2B) | NVIDIA's Qwen3-VL based model | ~8GB | float16 |
| [SmolVLM2 2.2B](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) | HuggingFace's lightweight VLM | ~5GB | bfloat16 |

## Installation

### Requirements

- Python 3.10+
- CUDA 12.0+ with compatible GPU
- ~8GB VRAM (model dependent)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/vtt_server.git
cd vtt_server

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate python-socketio uvicorn pillow

# For SmolVLM2 with flash-attention (optional, faster inference)
pip install flash-attn --no-build-isolation
```

### Download Models

Download your preferred model to the `models/` directory:

```bash
# Cosmos Reason 2
huggingface-cli download nvidia/Cosmos-Reason1-2B --local-dir ./models/cosmos_reason2

# SmolVLM2
huggingface-cli download HuggingFaceTB/SmolVLM2-2.2B-Instruct --local-dir ./models/smolvlm2
```

## Configuration

Edit `config.json` to select the active model and configure settings:

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8200
  },
  "models": [
    {
      "name": "cosmos",
      "active": true,
      "path": "./models/cosmos_reason2",
      "max_new_tokens": 512,
      "dtype": "float16",
      "attn_implementation": "sdpa",
      "memory": {
        "enabled": true,
        "max_context_tokens": 1024
      }
    },
    {
      "name": "smolvlm2",
      "active": false,
      "path": "./models/smolvlm2",
      "max_new_tokens": 256,
      "dtype": "bfloat16",
      "attn_implementation": "flash_attention_2",
      "memory": {
        "enabled": true,
        "max_context_tokens": 2048
      }
    }
  ],
  "inference": {
    "buffer_size": 1,
    "fps": 4,
    "default_prompt": "What do you see?"
  }
}
```

Set `"active": true` on the model you want to use (only one at a time).

## Usage

### Start the Server

```bash
python vtt_server.py
```

### Web Client

Serve the included HTML client:

```bash
python -m http.server 8080
```

Open `http://localhost:8080/vtt_client.html` in your browser.

### Client Features

- **Start Camera** — Begin streaming webcam frames to server
- **Start/Stop Inference** — Toggle continuous video description
- **Chat** — Send text queries with optional image attachment
- **Send with Current Frame** — Chat using the latest camera frame
- **Clear Memory** — Reset conversation history

## Socket.IO API

### Events (Client → Server)

| Event | Payload | Description |
|-------|---------|-------------|
| `frame` | `{camera: "front"\|"back", image: bytes}` | Send a JPEG frame |
| `start_inference` | `{camera: string, prompt?: string}` | Start continuous inference |
| `stop_inference` | `{camera: string}` | Stop continuous inference |
| `chat` | `{message: string, image?: bytes}` | Send chat message |
| `clear_memory` | `{}` | Clear session conversation history |

### Events (Server → Client)

| Event | Payload | Description |
|-------|---------|-------------|
| `status` | `{front: bool, back: bool, model: string, memory: bool}` | Connection status |
| `inference_started` | `{camera: string, prompt: string}` | Inference loop started |
| `inference_stopped` | `{camera: string}` | Inference loop stopped |
| `description` | `{camera: string, text: string, latency_ms: number, frame_count: number}` | Inference result |
| `chat_response` | `{text: string, latency_ms: number}` | Chat response |
| `memory_cleared` | `{}` | Memory cleared confirmation |
| `error` | `{message: string}` | Error message |

## Docker

### Build

```bash
# Build base image (includes flash-attention, takes ~15 min)
docker build -f vtt_server_server.Dockerfile -t vtt_server.base:latest .

# Build application image
docker build -f vtt_server.Dockerfile -t vtt_server:latest .
```

### Run

```bash
docker run --gpus all -p 8200:8200 \
  -v /path/to/models:/app/models \
  vtt_server:latest
```

## Project Structure

```
vtt_server/
├── vtt_server.py              # Main server
├── config.json                # Configuration
├── vtt_client.html            # Web test client
├── models/
│   ├── __init__.py            # BaseVTTModel abstract class
│   ├── factory.py             # Model factory and registry
│   ├── memory.py              # Session memory management
│   ├── cosmos.py              # Cosmos Reason 2 implementation
│   └── smolvlm2.py            # SmolVLM2 implementation
├── vtt_server.Dockerfile      # Application Dockerfile
└── vtt_server_server.Dockerfile  # Base image Dockerfile
```

## Adding New Models

1. Create a new file in `models/` (e.g., `models/newmodel.py`)
2. Implement `BaseVTTModel` interface:

```python
from . import BaseVTTModel

class NewModel(BaseVTTModel):
    def load(self) -> None:
        # Load model and processor
        pass
    
    def generate(self, frames: list[Image.Image], prompt: str, fps: int) -> str:
        # Generate text from video frames
        pass
    
    def chat(self, message: str, image: Image.Image | None = None, history: str | None = None) -> str:
        # Chat with optional image and history
        pass
```

3. Register in `models/factory.py`:

```python
from .newmodel import NewModel

MODEL_REGISTRY: dict[str, type[BaseVTTModel]] = {
    "cosmos": CosmosModel,
    "smolvlm2": SmolVLM2Model,
    "newmodel": NewModel,  # Add here
}
```

4. Add configuration entry in `config.json`

## Performance

Typical inference latency on RTX 4090:

| Model | Single Frame | 4 Frames |
|-------|--------------|----------|
| Cosmos Reason 2 | ~1.8s | ~2.5s |
| SmolVLM2 2.2B | ~1.1s | ~1.5s |

## License

MIT License

## Acknowledgments

- [NVIDIA Cosmos](https://github.com/NVIDIA/Cosmos) for Cosmos Reason 2
- [HuggingFace](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) for SmolVLM2
- [Qwen](https://github.com/QwenLM/Qwen2.5-VL) for the underlying VL architecture
