from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import torch
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    BitsAndBytesConfig
)
from PIL import Image
import tempfile
import os

app = FastAPI(title="VLM Service")

class LocalLlavaVLM:
    def __init__(self, model_id: str = "llava-hf/llava-1.5-7b-hf", device: str = "cuda"):
        use_cuda = device == "cuda" and torch.cuda.is_available()
        quant_config = BitsAndBytesConfig(
            load_in_4bit=use_cuda,
            bnb_4bit_compute_dtype=torch.float16 if use_cuda else torch.float32,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=False)  # <-- важно
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quant_config if use_cuda else None,
            torch_dtype=torch.float16 if use_cuda else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto" if use_cuda else None,
        )
        if not use_cuda:
            self.model.to("cpu")
        self.model.eval()

    def generate(self, prompt: str, image_path: str | None = None) -> str:
        system_prompt = "Ты -  Мини Алиса, умный голосовой ассистент. Тебя зовут Аня. Отвечай кратко."
        user_prompt = prompt or "Опиши изображение кратко и точно"
        if image_path:
            img = Image.open(image_path).convert("RGB")
            full_prompt = f"<image>\nUSER: <td>\n{system_prompt} {user_prompt}\nASSISTANT: "
            inputs = self.processor(text=full_prompt, images=img, return_tensors="pt")
        else:
            full_prompt = f"USER: {system_prompt} {user_prompt}\nASSISTANT: "
            inputs = self.processor(text=full_prompt, return_tensors="pt")

        # Фильтруем None значения
        inputs = {k: v.to(self.model.device) for k, v in inputs.items() if v is not None}

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        decoded = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        if "ASSISTANT:" in decoded:
            return decoded.split("ASSISTANT:", maxsplit=1)[-1].strip()
        return decoded.strip()
vlm = LocalLlavaVLM()

@app.post("/generate")
async def generate(
    text: str = Form(...),
    image: UploadFile = File(None)
):
    image_path = None
    if image:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            content = await image.read()
            tmp.write(content)
            tmp.flush()
            image_path = tmp.name

    answer = vlm.generate(text, image_path)

    if image_path:
        os.unlink(image_path)

    return {"answer": answer}