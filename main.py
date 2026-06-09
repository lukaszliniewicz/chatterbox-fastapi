import os
import io
import sys
import shutil
import threading
import logging
import types
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("chatterbox-api")

# Monkey-patch Perth Implicit Watermarker for Windows compatibility
# Perth watermarking binaries/libraries may fail to load or be None on Windows environments.
class DummyWatermarker:
    def __init__(self, *args, **kwargs):
        pass
    def apply_watermark(self, wav, *args, **kwargs):
        # Pass the audio tensor through unmodified
        return wav
    def encode(self, wav, *args, **kwargs):
        # Pass the audio tensor through unmodified
        return wav
    def decode(self, wav, *args, **kwargs):
        return None
    def __call__(self, *args, **kwargs):
        # Handle cases where the class instance or class is called directly
        if len(args) > 0:
            return args[0]
        return self

try:
    import perth
    if getattr(perth, "PerthImplicitWatermarker", None) is None:
        perth.PerthImplicitWatermarker = DummyWatermarker
        logger.info("Monkey-patched perth.PerthImplicitWatermarker with DummyWatermarker")
except ImportError:
    # Mock the entire perth module if not installed
    perth_mock = types.ModuleType("perth")
    perth_mock.PerthImplicitWatermarker = DummyWatermarker
    sys.modules["perth"] = perth_mock
    logger.info("Mocked missing perth module with DummyWatermarker")


# Setup FastAPI application
app = FastAPI(title="Chatterbox API Wrapper", description="OpenAI-compatible TTS API wrapper for Resemble AI's Chatterbox")

VOICES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices")
os.makedirs(VOICES_DIR, exist_ok=True)

# Lazy import check helper
try:
    import torch
    import torchaudio
    from pydub import AudioSegment
    from pydub.effects import speedup
    HAS_LIBS = True
except ImportError as e:
    logger.warning("Required ML or audio libraries not fully installed yet: %s. Run run.bat first.", e)
    HAS_LIBS = False


class ModelLoader:
    def __init__(self):
        self._models = {}
        self._lock = threading.Lock()

    def get_model(self, model_id: str, device: str = "cuda"):
        if not HAS_LIBS:
            raise HTTPException(status_code=500, detail="ML libraries are not installed. Run run.bat.")

        with self._lock:
            # Normalize model names
            normalized_id = model_id.lower().strip()
            if normalized_id in ("chatterbox", "chatterbox-en", "chatterbox_en", "en"):
                normalized_id = "chatterbox-en"
            elif "turbo" in normalized_id:
                normalized_id = "chatterbox-turbo"
            elif "multilingual" in normalized_id or "mtl" in normalized_id:
                normalized_id = "chatterbox-multilingual"
            else:
                logger.warning("Unknown model_id '%s', defaulting to chatterbox-turbo", model_id)
                normalized_id = "chatterbox-turbo"

            if normalized_id in self._models:
                return self._models[normalized_id]

            # Unload any previously loaded models to free memory (VRAM/RAM)
            if self._models:
                logger.info("Unloading existing models to free memory: %s", list(self._models.keys()))
                self._models.clear()
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Resolve device
            target_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
            logger.info("Initializing model '%s' on device '%s'...", normalized_id, target_device)

            try:
                if normalized_id == "chatterbox-turbo":
                    from chatterbox.tts_turbo import ChatterboxTurboTTS
                    model = ChatterboxTurboTTS.from_pretrained(device=target_device)
                elif normalized_id == "chatterbox-multilingual":
                    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                    model = ChatterboxMultilingualTTS.from_pretrained(device=target_device)
                else:
                    from chatterbox.tts import ChatterboxTTS
                    model = ChatterboxTTS.from_pretrained(device=target_device)

                self._models[normalized_id] = model
                logger.info("Successfully loaded model '%s'", normalized_id)
                return model
            except Exception as e:
                logger.error("Failed to load model '%s': %s", normalized_id, e, exc_info=True)
                raise HTTPException(status_code=500, detail=f"Failed to load model {normalized_id}: {str(e)}")


model_loader = ModelLoader()


class SpeechRequest(BaseModel):
    model: str
    input: str
    voice: Optional[str] = None
    language: Optional[str] = "en"
    speed: Optional[float] = 1.0
    temperature: Optional[float] = 0.8
    exaggeration: Optional[float] = 0.5
    cfg_weight: Optional[float] = 0.5
    response_format: Optional[str] = "wav"


@app.get("/health")
@app.get("/")
async def health_check():
    cuda_available = False
    cuda_device_name = ""
    if HAS_LIBS:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_device_name = torch.cuda.get_device_name(0)

    return {
        "status": "ok",
        "cuda_available": cuda_available,
        "cuda_device_name": cuda_device_name,
        "loaded_models": list(model_loader._models.keys()),
        "voices_count": len([name for name in os.listdir(VOICES_DIR) if name.endswith(".wav")]),
    }


@app.get("/v1/models")
@app.get("/v1/audio/models")
async def list_models():
    return {
        "data": [
            {"id": "chatterbox-turbo", "object": "model", "owned_by": "resemble-ai"},
            {"id": "chatterbox-multilingual", "object": "model", "owned_by": "resemble-ai"},
            {"id": "chatterbox-en", "object": "model", "owned_by": "resemble-ai"}
        ]
    }


@app.get("/v1/audio/voices")
@app.get("/v1/voices")
@app.get("/v1/files")
async def list_voices():
    voices = []
    if os.path.exists(VOICES_DIR):
        for name in os.listdir(VOICES_DIR):
            if name.endswith(".wav"):
                voice_id = os.path.splitext(name)[0]
                voices.append({
                    "id": voice_id,
                    "voice_id": voice_id,
                    "name": voice_id
                })
    return {"data": voices}


@app.post("/v1/audio/voices")
@app.post("/v1/voices")
@app.post("/v1/files")
async def upload_voice(
    files: Optional[List[UploadFile]] = File(None),
    audio_sample: Optional[UploadFile] = File(None),
    voice_id: Optional[str] = Form(None),
    name: Optional[str] = Form(None),
    purpose: Optional[str] = Form(None)
):
    if not HAS_LIBS:
        raise HTTPException(status_code=500, detail="Audio processing libraries are not installed.")

    resolved_id = voice_id or name
    target_file = None

    if audio_sample:
        target_file = audio_sample
    elif files and len(files) > 0:
        target_file = files[0]

    if not target_file:
        raise HTTPException(status_code=400, detail="No audio file uploaded.")

    if not resolved_id:
        filename = target_file.filename
        resolved_id = os.path.splitext(filename)[0]

    # Sanitize voice ID
    resolved_id = "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in resolved_id])
    target_path = os.path.join(VOICES_DIR, f"{resolved_id}.wav")
    temp_path = target_path + ".tmp"

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(target_file.file, buffer)

        # Convert to standard PCM WAV format using pydub
        try:
            sound = AudioSegment.from_file(temp_path)
            sound.export(target_path, format="wav")
            os.remove(temp_path)
            logger.info("Successfully uploaded and converted voice: %s", resolved_id)
        except Exception as e:
            logger.warning("Pydub conversion failed: %s. Saving uploaded file directly.", e)
            if os.path.exists(target_path):
                os.remove(target_path)
            os.rename(temp_path, target_path)

        return {
            "id": resolved_id,
            "voice_id": resolved_id,
            "name": resolved_id,
            "purpose": purpose or "user_data"
        }
    except Exception as e:
        logger.error("Failed to save voice file: %s", e)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Failed to upload voice: {str(e)}")


@app.post("/v1/audio/speech")
@app.post("/audio/speech")
async def generate_speech(request: SpeechRequest):
    if not HAS_LIBS:
        raise HTTPException(status_code=500, detail="Required libraries not installed.")

    logger.info("Speech request: model=%s, text_len=%d, voice=%s", request.model, len(request.input), request.voice)

    # Resolve voice prompt path if specified
    audio_prompt_path = None
    if request.voice and request.voice != "default":
        voice_file = os.path.join(VOICES_DIR, f"{request.voice}.wav")
        if os.path.exists(voice_file):
            audio_prompt_path = voice_file
            logger.info("Using reference voice: %s", voice_file)
        else:
            logger.warning("Voice '%s' not found, defaulting to standard synthesis.", request.voice)

    # Use CPU if requested via CHATTERBOX_DEVICE, otherwise use CUDA if available
    backend = os.environ.get("CHATTERBOX_DEVICE", "cuda").lower()
    device = "cpu" if backend == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu")
    model = model_loader.get_model(request.model, device=device)

    try:
        kwargs = {}
        if "multilingual" in request.model.lower():
            kwargs["language_id"] = request.language or "en"

        wav = model.generate(
            text=request.input,
            audio_prompt_path=audio_prompt_path,
            exaggeration=request.exaggeration or 0.5,
            cfg_weight=request.cfg_weight or 0.5,
            temperature=request.temperature or 0.8,
            **kwargs
        )

        # Ensure shape is 2D
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)

        # Save to buffer
        sr = getattr(model, "sr", 24000)
        buffer = io.BytesIO()
        torchaudio.save(buffer, wav.cpu(), sr, format="wav")
        buffer.seek(0)

        # Apply speed adjustment if needed
        if request.speed and request.speed != 1.0:
            try:
                sound = AudioSegment.from_wav(buffer)
                sound = speedup(sound, request.speed)
                buffer = io.BytesIO()
                sound.export(buffer, format="wav")
                buffer.seek(0)
                logger.info("Applied speed adjustment: %fx", request.speed)
            except Exception as speed_err:
                logger.error("Failed to adjust speed: %s", speed_err)

        return StreamingResponse(buffer, media_type="audio/wav")
    except Exception as e:
        logger.error("TTS generation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"TTS Generation failed: {str(e)}")
