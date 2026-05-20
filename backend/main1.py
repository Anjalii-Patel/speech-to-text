import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import asyncio
import aiohttp
from io import BytesIO
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Request
from faster_whisper import WhisperModel
from fastapi.middleware.cors import CORSMiddleware
import json
import time
import torch
import librosa
import soundfile as sf
import logging
from logging.config import dictConfig
import sys
from fastapi.responses import JSONResponse
import shutil
from datetime import datetime
import csv
import noisereduce as nr
from asyncio import Lock, Queue

# Logging
log_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"default": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"}},
    "handlers": {
        "file":    {"class": "logging.FileHandler",   "filename": "server.log", "formatter": "default", "level": "INFO"},
        "console": {"class": "logging.StreamHandler", "formatter": "default",   "level": "INFO"},
    },
    "root": {"handlers": ["file", "console"], "level": "INFO"},
}
dictConfig(log_config)
logger = logging.getLogger(__name__)

# Ollama endpoints
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_API_URL_2 = "http://localhost:11434/api/chat"
OLLAMA_TIMEOUT = aiohttp.ClientTimeout(total=60)

# Whisper model
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
kwargs = {"cpu_threads": os.cpu_count()} if device == "cpu" else {}

whisper_lock = Lock()          # prevents concurrent Whisper calls on same model

try:
    logger.info("Loading Whisper model...")
    model = WhisperModel(
        "large-v3",            # v3 > v2 for Indic languages
        local_files_only=False,
        device=device,
        compute_type=compute_type,
        **kwargs,
    )
    logger.info(f"Whisper loaded on {device.upper()} / {compute_type}")
except Exception as e:
    logger.error("Whisper load failed: %s", e)

# Shared aiohttp session (app-level singleton)
http_session: aiohttp.ClientSession | None = None

# FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    global http_session
    http_session = aiohttp.ClientSession(timeout=OLLAMA_TIMEOUT)

@app.on_event("shutdown")
async def shutdown():
    if http_session:
        await http_session.close()

# Audio pre-processing
def preprocess_audio(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Noise-reduce → normalize. Returns float32."""
    # Spectral gating noise reduction
    audio = nr.reduce_noise(y=audio, sr=sr, stationary=False, prop_decrease=0.85)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio.astype(np.float32)

# ASR
RMS_THRESHOLD = 0.01
LANG_PROB_THRESHOLD = 0.4

def ASR(audio: np.ndarray) -> str:
    if np.sqrt(np.mean(audio**2)) < RMS_THRESHOLD:
        return ""

    segments, info = model.transcribe(
        audio,
        beam_size=3,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 700},
        task="translate",
        language=None,
        condition_on_previous_text=False,
        word_timestamps=True,
    )

    if info.language_probability < LANG_PROB_THRESHOLD:
        return ""
    return " ".join(seg.text.strip() for seg in segments)

async def transcribe_async(audio: np.ndarray) -> str:
    """Thread-safe Whisper call with model lock."""
    async with whisper_lock:
        return await asyncio.to_thread(ASR, audio)

# LLM: structured summary
async def ask_llama(context: str = " ") -> dict:
    PROMPT_REQUESTS = {
        "brief_medical_history": {
            "prompt": "Summarize the Brief Patient medical History from entire conversation in one sentence.",
            "format": {"type": "string"}
        },
        "chief_complaints": {
            "prompt": "List patient's chief complaints with duration and description.",
            "format": {
                "type": "object",
                "properties": {
                    "Complaint":    {"type": "string"},
                    "Duration":     {"type": "string"},
                    "Description":  {"type": "string"},
                },
                "required": ["Complaint", "Duration", "Description"],
            },
        },
        "current_symptoms_and_medical_background": {
            "prompt": "Explain ODP/HPI, current symptoms and medicine history of patient.",
            "format": {"type": "string"},
        },
        "past_medical_history": {
            "prompt": "List patient's past medical history with diagnosis type.",
            "format": {
                "type": "object",
                "properties": {
                    "Diagnosis_Type": {"type": "string", "enum": ["Clinical","Differential","Final","Provisional","Suspected"]},
                    "Disease":        {"type": "string"},
                },
                "required": ["Diagnosis_Type", "Disease"],
            },
        },
        "hospitalization_and_surgical_history": {
            "prompt": "Mention patient's past hospitalization or surgeries with diagnosis, treatment, and admission time.",
            "format": {
                "type": "object",
                "properties": {
                    "Diagnosis":      {"type": "string"},
                    "Treatment":      {"type": "string"},
                    "Admission_Time": {"type": "string"},
                },
                "required": ["Diagnosis", "Treatment", "Admission_Time"],
            },
        },
        "gynecological_history": {
            "prompt": "Provide patient's gynecological history if available, if not then respond with None.",
            "format": {"type": "string"},
        },
        "lifestyle_and_social_activity": {
            "prompt": "Describe patient's physical activity, time and status.",
            "format": {
                "type": "object",
                "properties": {
                    "Physical_Activity": {"type": "string"},
                    "Time":              {"type": "string"},
                    "Status":            {"type": "string"},
                },
                "required": ["Physical_Activity", "Time", "Status"],
            },
        },
        "family_history": {
            "prompt": "Provide patient's relevant family medical history including relation, disease name and age.",
            "format": {
                "type": "object",
                "properties": {
                    "Relation":     {"type": "string"},
                    "Disease_Name": {"type": "string"},
                    "Age":          {"type": "string"},
                },
                "required": ["Relation", "Disease_Name", "Age"],
            },
        },
        "allergies_and_hypersensitivities": {
            "prompt": "Mention patient's allergies with allergen, reaction type, severity and status.",
            "format": {
                "type": "object",
                "properties": {
                    "Allergy":          {"type": "string"},
                    "Allergen":         {"type": "string"},
                    "Type_of_Reaction": {"type": "string"},
                    "Status":           {"type": "string", "enum": ["active","passive"]},
                    "Severity":         {"type": "string"},
                },
                "required": ["Allergy","Allergen","Type_of_Reaction","Status","Severity"],
            },
        },
    }

    async def query_ollama(field: str, details: dict):
        payload = {
            "model":   "llama3.2",
            "prompt":  f"While extracting information if you do not find the data answer strictly with None. Context: {context}\n\n{details['prompt']}",
            "format":  details["format"],
            "options": {"temperature": 0.0},
            "stream":  False,
        }
        for attempt in range(3):                       # retry up to 3×
            try:
                async with http_session.post(OLLAMA_API_URL, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return field, json.loads(data.get("response", "{}"))
                    else:
                        raw = await resp.text()
                        return field, {"error": f"HTTP {resp.status}", "raw": raw}
            except asyncio.TimeoutError:
                if attempt == 2:
                    return field, {"error": "timeout"}
                await asyncio.sleep(1)
            except Exception as e:
                return field, {"error": str(e)}

    start = time.perf_counter()
    tasks = [query_ollama(f, d) for f, d in PROMPT_REQUESTS.items()]
    results = await asyncio.gather(*tasks)
    logger.info(f"ask_llama total: {time.perf_counter()-start:.2f}s")
    return {f: r for f, r in results}

# LLM: chat
async def ask_llama1(context: str, query: str = " ") -> str:
    payload = {
        "model": "llama3.2",
        "messages": [
            {"role": "system", "content": "You are an expert doctor assistant made by Artem Health. Keep answers short and precise."},
            {"role": "user",   "content": f"Conversation context:\n{context}"},
            {"role": "user",   "content": query},
        ],
        "options": {"temperature": 0.0},
        "stream":  False,
    }
    try:
        async with http_session.post(OLLAMA_API_URL_2, json=payload) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("message", {}).get("content", "No response")
            return f"Error: {resp.status}"
    except asyncio.TimeoutError:
        return "Error: Ollama timeout"
    except Exception as e:
        return f"Exception: {e}"

# WebSocket: live transcription
@app.websocket("/transcribe/")
async def transcribe_audio(websocket: WebSocket):
    await websocket.accept()

    SAMPLE_RATE = 16000
    CHUNK_SECONDS = 6                              # increased from 4→6
    OVERLAP_SEC = 1
    CHUNK_SIZE = SAMPLE_RATE * CHUNK_SECONDS
    OVERLAP_SIZE = SAMPLE_RATE * OVERLAP_SEC
    MAX_QUEUE = SAMPLE_RATE * 30               # backpressure: ~30s max buffered

    audio_queue: list[np.ndarray] = []
    total_buffered = 0

    async def send_result(task: asyncio.Task):
        try:
            text = await task
            if text.strip():
                await websocket.send_text(text)
        except Exception as e:
            logger.error(f"send_result error: {e}")

    try:
        while True:
            # heartbeat / receive with timeout to detect dead connections
            try:
                audio_bytes = await asyncio.wait_for(websocket.receive_bytes(), timeout=30)
            except asyncio.TimeoutError:
                await websocket.send_text("")     # ping-keep-alive
                continue

            if len(audio_bytes) % 2 != 0:
                audio_bytes += b'\x00'

            chunk = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            audio_queue.append(chunk)
            total_buffered += len(chunk)

            # backpressure: drop oldest if too far behind
            while total_buffered > MAX_QUEUE and audio_queue:
                dropped = audio_queue.pop(0)
                total_buffered -= len(dropped)
                logger.warning("Backpressure: dropped old audio chunk")

            if total_buffered >= CHUNK_SIZE:
                combined = np.concatenate(audio_queue)
                # carry-forward overlap to avoid word cutoff
                leftover = combined[-OVERLAP_SIZE:]
                audio_queue.clear()
                audio_queue.append(leftover)
                total_buffered = len(leftover)

                # noise-reduce before ASR
                cleaned = preprocess_audio(combined)
                task = asyncio.create_task(transcribe_async(cleaned))
                asyncio.create_task(send_result(task))

    except WebSocketDisconnect:
        logger.info("Transcribe client disconnected")

# WebSocket: summary
@app.websocket("/generate_summary/")
async def generate_summary_api(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            transcription = await websocket.receive_text()
            summary = await ask_llama(transcription)
            await websocket.send_text(json.dumps(summary))
    except WebSocketDisconnect:
        logger.info("Summary client disconnected")

# WebSocket: AI chat
@app.websocket("/talk_with_ai/")
async def talk_with_ai_api(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            transcription = data.get("transcription", "")
            query = data.get("query", "")
            response = await ask_llama1(transcription, query)
            await websocket.send_text(response)
    except WebSocketDisconnect:
        logger.info("AI chat client disconnected")

# REST: upload & transcribe
def load_audio_bytes(audio_bytes: bytes) -> np.ndarray:
    audio_array, sample_rate = sf.read(BytesIO(audio_bytes))
    if audio_array.ndim > 1:
        audio_array = np.mean(audio_array, axis=1)
    if sample_rate != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
    return preprocess_audio(audio_array)

@app.post("/upload/")
async def upload_audio(file: UploadFile = File(...)):
    try:
        audio_array = load_audio_bytes(await file.read())
        text = await transcribe_async(audio_array)
        return {"filename": file.filename, "transcription": text}
    except Exception as e:
        logger.error(f"upload error: {e}")
        return {"error": str(e)}

@app.post("/upload_and_summary/")
async def upload_and_summary(file: UploadFile = File(...)):
    try:
        audio_array = load_audio_bytes(await file.read())
        text = await transcribe_async(audio_array)
        summary = await ask_llama(text)
        return {"filename": file.filename, "final_transcription": text, "summary": summary}
    except Exception as e:
        logger.error(f"upload_and_summary error: {e}")
        return {"error": str(e)}

# REST: beam size
beamsize = 3

@app.post("/update-beam-size/")
async def update_beam_size(new_beam_size: int = Form(...)):
    global beamsize
    try:
        beamsize = int(new_beam_size)
        return JSONResponse({"status": "success", "beam_size": beamsize})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# REST: save audio
base_dir = (
    os.path.dirname(sys.executable) if getattr(sys, "frozen", False)
    else os.path.dirname(__file__)
)
SAVE_DIR = os.path.join(base_dir, "recordings")
os.makedirs(SAVE_DIR, exist_ok=True)

@app.post("/save-audio/")
async def save_audio(
    audio_file: UploadFile,
    transcription: str = Form(...),
    live_transcription: str = Form(...),
    quality: str = Form(...),
):
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(SAVE_DIR, quality)
        os.makedirs(save_dir, exist_ok=True)

        audio_path = os.path.join(save_dir, f"audio_{ts}.wav")
        text_path = os.path.join(save_dir, f"transcription_{ts}.txt")
        live_path = os.path.join(save_dir, f"live_transcription_{ts}.txt")

        with open(audio_path, "wb") as f:
            shutil.copyfileobj(audio_file.file, f)
        with open(text_path,  "w", encoding="utf-8") as f:
            f.write(transcription)
        with open(live_path,  "w", encoding="utf-8") as f:
            f.write(live_transcription)

        return JSONResponse({"message": "Saved", "audio_path": audio_path, "text_path": text_path})
    except Exception as e:
        logger.error(f"save_audio error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# REST: evaluate summary
EVALUATION_CSV_PATH = os.path.join(base_dir, "evaluation_data.csv")

@app.post("/evaluate-summary/")
async def evaluate_summary(request: Request):
    try:
        data = await request.json()
        # Define CSV headers
        headers = [
            "timestamp","transcription",
            "brief_medical_history_text","brief_medical_history_correct",
            "chief_complaints_text","chief_complaints_correct",
            "current_symptoms_text","current_symptoms_correct",
            "past_medical_history_text","past_medical_history_correct",
            "hospitalization_text","hospitalization_correct",
            "gynecological_history_text","gynecological_history_correct",
            "lifestyle_text","lifestyle_correct",
            "family_history_text","family_history_correct",
            "allergies_text","allergies_correct",
        ]
        row = {
            "timestamp":   data.get("timestamp", ""),
            "transcription": data.get("transcription", ""),
            "brief_medical_history_text":    data.get("brief_medical_history_text", ""),
            "brief_medical_history_correct": data.get("brief_medical_history_correct", ""),
            "chief_complaints_text":    f"Complaint: {data.get('chief_complaints_complaint','')}; Duration: {data.get('chief_complaints_duration','')}; Description: {data.get('chief_complaints_description','')}",
            "chief_complaints_correct": data.get("chief_complaints_correct", ""),
            "current_symptoms_text":    data.get("current_symptoms_and_medical_background", ""),
            "current_symptoms_correct": data.get("current_symptoms_correct", ""),
            "past_medical_history_text":    f"Diagnosis Type: {data.get('past_medical_history_diagnosis_type','')}; Disease: {data.get('past_medical_history_disease','')}",
            "past_medical_history_correct": data.get("past_medical_history_correct", ""),
            "hospitalization_text":    f"Diagnosis: {data.get('hospitalization_diagnosis','')}; Treatment: {data.get('hospitalization_treatment','')}; Admission Time: {data.get('hospitalization_admission_time','')}",
            "hospitalization_correct": data.get("hospitalization_correct", ""),
            "gynecological_history_text":    data.get("gynecological_history", ""),
            "gynecological_history_correct": data.get("gynecological_history_correct", ""),
            "lifestyle_text":    f"Physical Activity: {data.get('lifestyle_physical_activity','')}; Time: {data.get('lifestyle_time','')}; Status: {data.get('lifestyle_status','')}",
            "lifestyle_correct": data.get("lifestyle_correct", ""),
            "family_history_text":    f"Relation: {data.get('family_history_relation','')}; Disease Name: {data.get('family_history_disease_name','')}; Age: {data.get('family_history_age','')}",
            "family_history_correct": data.get("family_history_correct", ""),
            "allergies_text":    f"Allergy: {data.get('allergies_allergy','')}; Allergen: {data.get('allergies_allergen','')}; Reaction: {data.get('allergies_reaction_type','')}; Status: {data.get('allergies_status','')}; Severity: {data.get('allergies_severity','')}",
            "allergies_correct": data.get("allergies_correct", ""),
        }
        file_exists = os.path.isfile(EVALUATION_CSV_PATH)
        with open(EVALUATION_CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        return JSONResponse({"status": "success"})
    except Exception as e:
        logger.error(f"evaluate_summary error: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# Entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        forwarded_allow_ips="*",
        ssl_certfile="cert.pem",
        ssl_keyfile="key.pem",
        log_config=log_config,
    )
