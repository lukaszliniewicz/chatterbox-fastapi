# Chatterbox FastAPI

Chatterbox FastAPI is an HTTP API wrapper for Resemble AI's [Chatterbox](https://github.com/resemble-ai/chatterbox) text-to-speech (TTS) library. It provides an OpenAI-compatible speech generation endpoint and was developed to serve as a backend engine for the [Pandrator](https://github.com/lukaszliniewicz/Pandrator) application.

## Features

- **OpenAI-Compatible Speech API**: Exposes standard speech endpoints (`/v1/audio/speech`) for drop-in integration.
- **Dynamic Parameter Mapping**: Uses signature introspection to dynamically filter and apply model-specific parameters (`temperature`, `repetition_penalty`, `min_p`, `top_p`, `top_k`, `exaggeration`, `cfg_weight`, `norm_loudness`) depending on whether a turbo, english, or multilingual model is loaded.
- **Model VRAM Unloading**: Automatically unloads inactive models from memory (CUDA VRAM/RAM) when switching models to prevent out-of-memory errors on consumer GPUs.
- **CPU Mode Support**: Can be executed entirely on host CPU memory via configuration.
- **Cloned Voice Management**: Supports uploading reference voice samples dynamically to create custom speakers.

## Prerequisites

- [Pixi](https://pixi.sh) package manager
- FFmpeg installed on your system (or managed automatically via conda-forge)
- NVIDIA GPU (optional; falls back to CPU if configured)

## Installation & Running

Initialize the Pixi environment and run the server (by default listening on port `8040`):

```bash
# Start using python directly
pixi run python run.py

# Or run the batch helper on Windows
run.bat
```

To run the server in CPU-only mode, set the device environment variable:

```bash
set CHATTERBOX_DEVICE=cpu
pixi run python run.py
```

## API Endpoints

### 1. Generate Speech
- **URL**: `POST /v1/audio/speech` (or `POST /v1/speech`)
- **Format**: JSON payload compatible with OpenAI's audio request schema.
- **Supported parameters**: `model`, `input`, `voice` (speaker name), `speed`, `language`, `temperature`, `repetition_penalty`, `min_p`, `top_p`, `top_k`, `exaggeration`, `cfg_weight`, `norm_loudness`.

### 2. List Models
- **URL**: `GET /v1/models`
- **Output**: JSON list of available Chatterbox model architectures:
  - `chatterbox-turbo` (Default)
  - `chatterbox-multilingual`
  - `chatterbox-en`

### 3. List Voices
- **URL**: `GET /v1/voices`
- **Output**: JSON list of default and uploaded reference voices.

### 4. Upload Reference Voice
- **URL**: `POST /v1/voices/upload`
- **Format**: Multipart form data with file `file` (WAV audio format), `name` (speaker identifier), and optional `prompt_text`.
