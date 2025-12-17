from fastapi import FastAPI, UploadFile, File, HTTPException
import requests
import uuid
import os
import json

WHISPER_URL = "http://whisper-stt:10300/transcribe"
OLLAMA_URL  = "http://ollama:11434/api/generate"
TTS_URL = "http://tts:5002/api/tts"

AUDIO_IN = "/audio/in"
AUDIO_OUT = "/audio/out"

os.makedirs(AUDIO_IN, exist_ok=True)
os.makedirs(AUDIO_OUT, exist_ok=True)

app = FastAPI()

@app.post("/process")
async def process_audio(file: UploadFile = File(...)):
    audio_id = str(uuid.uuid4())
    input_path = f"{AUDIO_IN}/{audio_id}_{file.filename}"

    # salvar áudio recebido
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # 1) STT (Whisper)
    stt_resp = requests.post(
        WHISPER_URL,
        files={"file": open(input_path, "rb")}
    )
    if stt_resp.status_code != 200:
        raise HTTPException(500, "STT failed")

    text_in = stt_resp.json().get("text")
    if not text_in:
        raise HTTPException(500, "Empty transcription")

    # 2) LLM (Ollama - streaming)
    llm_resp = requests.post(
        OLLAMA_URL,
        json={
            "model": "phi3:mini",
            "prompt": text_in,
            "stream": True
        },
        stream=True
    )

    if llm_resp.status_code != 200:
        raise HTTPException(500, "LLM failed")

    text_out = ""
    for line in llm_resp.iter_lines():
        if line:
            data = json.loads(line.decode("utf-8"))
            if "response" in data:
                text_out += data["response"]

    if not text_out:
        raise HTTPException(500, "Empty LLM response")

    # 3) TTS (XTTS - file based)
tts_resp = requests.get(
    TTS_URL,
    params={
        "text": text_out,
        "speaker": "default",
        "language": "pt-br"
    }
)

if tts_resp.status_code != 200:
    raise HTTPException(500, f"TTS failed: {tts_resp.text}")

tts_file = tts_resp.json().get("file")
if not tts_file:
    raise HTTPException(500, "XTTS did not return file path")

# Ex: /output/uuid.wav  →  /tts-output/uuid.wav
filename = os.path.basename(tts_file)
source_path = f"/tts-output/{filename}"

final_path = f"{AUDIO_OUT}/{audio_id}.wav"

if not os.path.exists(source_path):
    raise HTTPException(500, "Generated WAV not found")

shutil.copy(source_path, final_path)

