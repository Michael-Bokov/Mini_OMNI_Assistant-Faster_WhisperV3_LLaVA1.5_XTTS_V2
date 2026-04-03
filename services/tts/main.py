from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from TTS.api import TTS
import tempfile
import os
from fastapi.responses import FileResponse

app = FastAPI(title="TTS Service")

class TTSRequest(BaseModel):
    text: str
    language: str = "ru"

class CoquiTTS:
    def __init__(self):
        os.environ["COQUI_TOS_AGREED"] = "1"
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        
        print(f"Загрузка модели {model_name}... Это займет время (2.5 ГБ)")
        self.tts = TTS(
            model_name=model_name, 
            progress_bar=True, 
            gpu=False
        )
        os.environ["COQUI_TOS_AGREED"] = "1"

        self.tts.to("cpu") 
        print("XTTS-v2 успешно загружена на CPU.")
        # self.predefined_speakers = ['Claribel Dervla', 'Daisy Studious', 'Gracie Wise', 'Tammie Ema', 'Alison Dietlinde', 'Ana Florence',
        #                              'Annmarie Nele', 'Asya Anara', 'Brenda Stern', 'Gitta Nikolina', 'Henriette Usha', 'Sofia Hellen', 
        #                              'Tammy Grit', 'Tanja Adelina', 'Vjollca Johnnie', 'Andrew Chipper', 'Badr Odhiambo', 
        #                              'Dionisio Schuyler', 'Royston Min', 'Viktor Eka', 'Abrahan Mack', 'Adde Michal', 
        #                              'Baldur Sanjin', 'Craig Gutsy', 'Damien Black', 'Gilberto Mathias', 
        #                              'Ilkin Urbano', 'Kazuhiko Atallah', 'Ludvig Milivoj', 'Suad Qasim', 
        #                              'Torcull Diarmuid', 'Viktor Menelaos', 'Zacharie Aimilios', 'Nova Hogarth', 'Maja Ruoho', 
        #                              'Uta Obando', 'Lidiya Szekeres', 'Chandra MacFarland', 'Szofi Granger', 'Camilla Holmström', 
        #                              'Lilya Stainthorpe', 'Zofija Kendrick', 'Narelle Moon', 'Barbora MacLean', 'Alexandra Hisakawa', 
        #                              'Alma María', 'Rosemary Okafor', 'Ige Behringer', 'Filip Traverse', 'Damjan Chapman', 
        #                              'Wulf Carlevaro', 'Aaron Dreschner', 'Kumar Dahl', 'Eugenio Mataracı', 
        #                              'Ferran Simen', 'Xavier Hayasaka', 'Luis Moray', 'Marcos Rudaski']
    
    def synthesize(self, text: str, language: str) -> str:
        
        import uuid
        tmp_path = f"/tmp/{uuid.uuid4()}.wav"
        
        try:
            self.tts.tts_to_file(
                text=text, 
                file_path=tmp_path,
                speaker="Ana Florence", 
                language=language
            )
            
            if os.path.exists(tmp_path):
                return tmp_path
            else:
                raise Exception("Файл не был создан или пуст")
                
        except Exception as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise e
tts = CoquiTTS()

@app.post("/synthesize")
async def synthesize_endpoint(request: TTSRequest):
    try:
        audio_path = tts.synthesize(request.text, request.language)
        return FileResponse(audio_path, media_type="audio/wav", filename="response.wav")
    except Exception as e:
        print(f"Ошибка синтеза: {e}")
        raise HTTPException(status_code=500, detail=str(e))