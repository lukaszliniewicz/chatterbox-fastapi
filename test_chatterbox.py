import sys
import os
import requests
import io

BASE_URL = "http://127.0.0.1:8040"

def test_health():
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()
        data = response.json()
        print(f"Health check response: {data}")
        assert data["status"] == "ok"
        print("Health check PASSED.\n")
        return data["cuda_available"]
    except Exception as e:
        print(f"Health check FAILED: {e}")
        sys.exit(1)

def test_list_models():
    print("Testing list models endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/v1/models")
        response.raise_for_status()
        data = response.json()
        print(f"Models: {data}")
        models = [m["id"] for m in data["data"]]
        assert "chatterbox-turbo" in models
        print("List models PASSED.\n")
    except Exception as e:
        print(f"List models FAILED: {e}")
        sys.exit(1)

def test_generate_speech():
    print("Testing standard speech generation...")
    payload = {
        "model": "chatterbox-turbo",
        "input": "Hello, this is a test of the Chatterbox API server. [laugh]",
        "speed": 1.0,
        "temperature": 0.8,
        "exaggeration": 0.5,
        "cfg_weight": 0.5
    }
    try:
        response = requests.post(f"{BASE_URL}/v1/audio/speech", json=payload)
        response.raise_for_status()
        audio_content = response.content
        print(f"Audio content length: {len(audio_content)} bytes")
        # Assert it's a valid WAV by checking header bytes 'RIFF'
        assert audio_content[:4] == b'RIFF'
        print("Standard speech generation PASSED.\n")
        
        output_file = "test_output.wav"
        with open(output_file, "wb") as f:
            f.write(audio_content)
        print(f"Saved generated audio to: {os.path.abspath(output_file)}")
    except Exception as e:
        print(f"Speech generation FAILED: {e}")
        sys.exit(1)

def test_upload_voice():
    print("Testing voice upload and cloning...")
    # 1. Create a dummy wav file content or use a small sample
    # We will generate a minimal valid WAV file header + data (1 second of silence)
    # 44100Hz, 16bit, mono: 44 bytes header + 88200 bytes silence (0)
    import struct
    sample_rate = 44100
    num_samples = 44100 * 6
    data = struct.pack("<" + "h" * num_samples, *([0] * num_samples))
    
    wav_io = io.BytesIO()
    wav_io.write(b'RIFF')
    wav_io.write(struct.pack('<L', 36 + len(data)))
    wav_io.write(b'WAVEfmt ')
    wav_io.write(struct.pack('<L', 16))
    wav_io.write(struct.pack('<HHLLHH', 1, 1, sample_rate, sample_rate * 2, 2, 16))
    wav_io.write(b'data')
    wav_io.write(struct.pack('<L', len(data)))
    wav_io.write(data)
    wav_io.seek(0)
    
    # 2. Upload the dummy WAV
    try:
        files = {
            "audio_sample": ("test_ref_voice.wav", wav_io, "audio/wav")
        }
        data = {
            "voice_id": "test_cloned_voice",
            "purpose": "user_data"
        }
        response = requests.post(f"{BASE_URL}/v1/audio/voices", files=files, data=data)
        response.raise_for_status()
        res_data = response.json()
        print(f"Upload response: {res_data}")
        assert res_data["id"] == "test_cloned_voice"
        print("Voice upload PASSED.\n")
    except Exception as e:
        print(f"Voice upload FAILED: {e}")
        sys.exit(1)

    # 3. List voices to verify
    try:
        response = requests.get(f"{BASE_URL}/v1/audio/voices")
        response.raise_for_status()
        voices_list = response.json()["data"]
        voice_ids = [v["id"] for v in voices_list]
        print(f"Discovered voices: {voice_ids}")
        assert "test_cloned_voice" in voice_ids
        print("List voices verification PASSED.\n")
    except Exception as e:
        print(f"List voices verification FAILED: {e}")
        sys.exit(1)

    # 4. Generate cloned speech
    print("Testing speech generation with cloned voice...")
    payload = {
        "model": "chatterbox-turbo",
        "input": "This speech is generated using a cloned voice.",
        "voice": "test_cloned_voice"
    }
    try:
        response = requests.post(f"{BASE_URL}/v1/audio/speech", json=payload)
        response.raise_for_status()
        audio_content = response.content
        print(f"Cloned audio content length: {len(audio_content)} bytes")
        assert audio_content[:4] == b'RIFF'
        print("Cloned speech generation PASSED.\n")
        
        output_file = "test_output_cloned.wav"
        with open(output_file, "wb") as f:
            f.write(audio_content)
        print(f"Saved cloned audio to: {os.path.abspath(output_file)}")
    except Exception as e:
        print(f"Cloned speech generation FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("===================================================")
    print("Chatterbox FastAPI Server Wrapper Verification")
    print("===================================================\n")
    cuda = test_health()
    test_list_models()
    test_generate_speech()
    test_upload_voice()
    print("===================================================")
    print("All tests completed successfully!")
    print("===================================================")
