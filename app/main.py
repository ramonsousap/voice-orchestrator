from fastapi import FastAPI, UploadFile, File, HTTPException
import requests
import uuid
import os
import shutil

STT_URL = "http://192.168.0.205:10300/transcribe"
LLM_URL = "http://192.168.0.205:11434/api/generate"
TTS_URL = "http://192.168.0.205:5002/tts"


AUDIO_IN = "/audio/in"
AUDIO_OUT = "/audio/out"

os.makedirs(AUDIO_IN, exist_ok=True)
os.makedirs(AUDIO_OUT, exist_ok=True)

app = FastAPI()

@app.post("/process")
async def process_audio(file: UploadFile = File(...)):
    # 1) salvar áudio de entrada
    audio_id = str(uuid.uuid4())
    in_path = f"{AUDIO_IN}/{audio_id}_{file.filename}"
    with open(in_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 2) STT (Whisper)
    with open(in_path, "rb") as f:
        stt_resp = requests.post(
            STT_URL,
            files={"file": f},
            data={"language": "pt"}
        )
    if stt_resp.status_code != 200:
        raise HTTPException(500, "STT failed")

    text_in = stt_resp.json().get("text", "").strip()
    if not text_in:
        raise HTTPException(500, "Empty transcription")

    # 3) LLM (Ollama)
    llm_payload = {
        "model": "phi3:mini",
        "prompt": text_in,
        "stream": False
    }
    llm_resp = requests.post(LLM_URL, json=llm_payload)
    if llm_resp.status_code != 200:
        raise HTTPException(500, "LLM failed")

    text_out = llm_resp.json().get("response", "").strip()
    if not text_out:
        raise HTTPException(500, "Empty LLM response")

    # 4) TTS (XTTS)
tts_resp = requests.get(
    TTS_URL,
    params={"text": text_out},
    stream=True
)

if tts_resp.status_code != 200:
    raise HTTPException(500, "TTS failed")

final_path = f"{AUDIO_OUT}/{audio_id}.wav"

with open(final_path, "wb") as f:
    for chunk in tts_resp.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)

return {
    "transcription": text_in,
    "response": text_out,
    "audio": final_path
}

