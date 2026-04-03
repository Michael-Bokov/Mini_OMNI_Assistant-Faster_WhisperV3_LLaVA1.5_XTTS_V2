from fastapi import FastAPI, UploadFile, File, HTTPException
from faster_whisper import WhisperModel
import tempfile
import os
import torch

app = FastAPI(title="ASR Service", version="2.0")

class FasterWhisperASR:
    def __init__(self, model_size: str = "large-v3", device: str = "cuda"):
        """
        model_size: "tiny", "base", "small", "medium", "large-v3"
        device: "cuda" или "cpu"
        """
        compute_type = "float16" if device == "cuda" else "int8"
        
        if device == "cuda" and torch.cuda.is_available():
            free_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"Available VRAM: {free_vram:.1f} GB")
            
            if free_vram < 8:
                compute_type = "int8_float16"
                print(f"Low VRAM detected, using {compute_type}")
        
        self.model = WhisperModel(
            model_size, 
            device=device,
            compute_type=compute_type,
            cpu_threads=8,
            num_workers=2
        )
        
        print(f"✅ Faster-Whisper loaded: {model_size} on {device} with {compute_type}")
        
        # Вывод использованной памяти
        if device == "cuda":
            used_vram = torch.cuda.memory_allocated() / 1e9
            print(f"VRAM used by ASR: {used_vram:.2f} GB")
    
    def transcribe(self, audio_path: str, language: str = "ru") -> str:
        """Транскрибация аудио с автоматической фильтрацией тишины"""
        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            #best_of=2,
            vad_filter=True,  # отключаем тишину
            vad_parameters=dict(
                min_silence_duration_ms=300,
                threshold=0.3,
                speech_pad_ms=400
            )
        )
        
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text)
        
        result =  " ".join(text_parts).strip() #"".join(segment.text for segment in segments).strip() 
        return result

asr = FasterWhisperASR(model_size="large-v3", device="cuda")

@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: str = "ru"  
):
    """Преобразование аудио в текст"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp.flush()
        
        try:
            text = asr.transcribe(tmp.name, language=language)
            return {
                "text": text,
                "language": language,
                "model": "faster-whisper-large-v3"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"ASR failed: {str(e)}")
        finally:
            os.unlink(tmp.name)

@app.get("/health")
async def health():
    """Проверка статуса сервиса"""
    return {
        "status": "healthy",
        "device": "cuda",
        "model": "large-v3",
        "vram_used_gb": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)