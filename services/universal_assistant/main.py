from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import httpx
import os
from typing import Optional
import uuid
from pathlib import Path

app = FastAPI(title="Universal Assistant", version="2.0")

OUTPUT_DIR = Path("/app/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
STATIC_DIR = Path("/app/static")

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

ASR_URL = os.getenv("ASR_URL", "http://asr:8001")
VLM_URL = os.getenv("VLM_URL", "http://vlm:8002")
TTS_URL = os.getenv("TTS_URL", "http://tts:8003")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/chat")
async def chat(
    prompt: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    language: str = Form("ru"),
):
    user_prompt = (prompt or "").strip()
    transcript = None
    user_text = user_prompt

    try:
        async with httpx.AsyncClient(timeout=280.0) as client:
            # 1) Текст запроса: ASR при наличии аудио, иначе prompt.
            if audio:
                audio_bytes = await audio.read()
                if not audio_bytes:
                    raise HTTPException(
                        status_code=400,
                        detail="Пустой аудиофайл. Запишите голос ещё раз.",
                    )
                asr_resp = await client.post(
                    f"{ASR_URL}/transcribe",
                    files={
                        "audio": (
                            "audio.webm",
                            audio_bytes,
                            "audio/webm",
                        )
                    },
                    params={"language": language},
                )
                if asr_resp.status_code != 200:
                    raise HTTPException(
                        status_code=502,
                        detail=f"ASR failed ({asr_resp.status_code}): {asr_resp.text[:800]}",
                    )
                try:
                    asr_data = asr_resp.json()
                except Exception:
                    raise HTTPException(
                        status_code=502,
                        detail=f"ASR вернул не JSON: {asr_resp.text[:500]}",
                    )
                transcript = (asr_data.get("text") or "").strip()
                user_text = transcript or user_text

            if not user_text:
                raise HTTPException(
                    status_code=400,
                    detail="Введите текст или запишите голосовое сообщение.",
                )

            # 2) VLM: текст и опционально изображение.
            vlm_files = {}
            if image:
                image_bytes = await image.read()
                content_type = image.content_type or "image/jpeg"
                filename = image.filename or "image.jpg"
                vlm_files["image"] = (filename, image_bytes, content_type)

            vlm_resp = await client.post(
                f"{VLM_URL}/generate",
                data={"text": user_text},
                files=vlm_files if vlm_files else None,
            )
            if vlm_resp.status_code != 200:
                raise HTTPException(
                    status_code=502,
                    detail=f"VLM failed ({vlm_resp.status_code}): {vlm_resp.text[:800]}",
                )
            try:
                vlm_data = vlm_resp.json()
            except Exception:
                raise HTTPException(
                    status_code=502,
                    detail=f"VLM вернул не JSON: {vlm_resp.text[:500]}",
                )
            answer = (vlm_data.get("answer") or "").strip()
            if not answer:
                raise HTTPException(
                    status_code=502, detail="VLM returned empty answer."
                )

            # 3) TTS
            tts_resp = await client.post(
                f"{TTS_URL}/synthesize",
                json={"text": answer, "language": language},
            )
            if tts_resp.status_code != 200:
                raise HTTPException(
                    status_code=502,
                    detail=f"TTS failed ({tts_resp.status_code}): {tts_resp.text[:800]}",
                )

            audio_id = uuid.uuid4().hex
            audio_path = OUTPUT_DIR / f"{audio_id}.wav"
            with open(audio_path, "wb") as f:
                f.write(tts_resp.content)

            return {
                "user_text": user_text,
                "transcript": transcript,
                "assistant_text": answer,
                "audio_url": f"/outputs/{audio_id}.wav",
            }
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Сервис недоступен (сеть): {exc}",
        ) from exc


@app.post("/process")
async def process_compat(
    prompt: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    language: str = Form("ru"),
):
    return await chat(prompt=prompt, audio=audio, image=image, language=language)
