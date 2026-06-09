#!/usr/bin/env python3
import sys
import os
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log = logging.getLogger("run")

def main():
    parser = argparse.ArgumentParser(description="Chatterbox FastAPI Bootstrapper")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface")
    parser.add_argument("--port", type=int, default=8040, help="Port number")
    parser.add_argument("--backend", choices=["cuda", "cpu"], default="cuda", help="Backend target")
    parser.add_argument("--skip-gpu-check", action="store_true", help="Skip NVIDIA GPU check")
    
    args = parser.parse_args()
    
    # Check GPU if cuda requested
    if args.backend == "cuda" and not args.skip_gpu_check:
        try:
            import torch
            if not torch.cuda.is_available():
                log.error("CUDA is not available in PyTorch, but backend is set to cuda. Fallback to CPU.")
                args.backend = "cpu"
            else:
                log.info("CUDA is available! Using GPU for inference: %s", torch.cuda.get_device_name(0))
        except ImportError:
            log.warning("PyTorch not installed yet or import failed.")

    os.environ["CHATTERBOX_DEVICE"] = args.backend

    # Configure model caching directory
    # Align with Pandrator portable model cache directory structure
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_cache_root = os.path.join(parent_dir, 'cache')
    os.makedirs(local_cache_root, exist_ok=True)
    
    os.environ.setdefault('XDG_CACHE_HOME', local_cache_root)
    os.environ.setdefault('HF_HOME', os.path.join(local_cache_root, 'huggingface'))
    os.environ.setdefault('HF_HUB_CACHE', os.path.join(local_cache_root, 'huggingface', 'hub'))
    os.environ.setdefault('HUGGINGFACE_HUB_CACHE', os.path.join(local_cache_root, 'huggingface', 'hub'))
    os.environ.setdefault('TRANSFORMERS_CACHE', os.path.join(local_cache_root, 'huggingface', 'transformers'))
    os.environ.setdefault('TORCH_HOME', os.path.join(local_cache_root, 'torch'))
    os.environ.setdefault('TTS_HOME', os.path.join(local_cache_root, 'tts'))
    
    log.info("Starting uvicorn server on %s:%d...", args.host, args.port)
    
    import uvicorn
    uvicorn.run("main:app", host=args.host, port=args.port, log_level="info", access_log=False)

if __name__ == "__main__":
    main()
