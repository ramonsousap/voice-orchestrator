from fastapi import FastAPI, UploadFile, File, HTTPException
import requests
import uuid
import os
import shutil

WHISPER_URL = "http://172.19.0.2:10300/transcribe"
OLLAMA_URL  = "http://ollama:11434/api/generate"
TTS_URL     = "http://tts:5002/tts"


AUDIO_IN = "/audio/in"
AUDIO_OUT = "/audio/out"

os.makedirs(AUDIO_IN, exist_ok=True)
os.makedirs(AUDIO_OUT, exist_ok=True)

app = FastAPI()

@app.post("/process")
async def process_audio(file: UploadFile = File(...)):

    audio_id = str(uuid.uuid4())
    input_path = f"{AUDIO_IN}/{audio_id}_{file.filename}"

    with open(input_path, "wb") as f:
        f.write(await file.read())

    # 1) STT
    stt_resp = requests.post(
        WHISPER_URL,
        files={"file": open(input_path, "rb")}
    )
    text_in = stt_resp.json().get("text")

    # 2) LLM
    llm_resp = requests.post(
        OLLAMA_URL,
        json={"model": "phi3:mini", "prompt": text_in}
    )
    text_out = llm_resp.json().get("response")

    # 3) TTS (CORRIGIDO)
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

