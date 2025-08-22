# main1.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import asyncio
import aiohttp
from io import BytesIO
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
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

log_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        },
    },
    "handlers": {
        "file": {
            "class": "logging.FileHandler",
            "filename": "server.log",
            "formatter": "default",
            "level": "INFO",
        },
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "INFO",
        },
    },
    "root": {
        "handlers": ["file", "console"],
        "level": "INFO",
    },
}

dictConfig(log_config)
logger = logging.getLogger(__name__)

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_API_URL_2 = "http://localhost:11434/api/chat"

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set compute type
compute_type = "float16" if device == "cuda" else "int8"

kwargs = {}
if device == "cpu":
    kwargs["cpu_threads"] = os.cpu_count()

try:
    logger.info("Whisper model attempt to load.")

    model = WhisperModel(
        "large-v2",
        local_files_only=True,
        device=device,
        compute_type=compute_type,
        **kwargs
    )

    logger.info(f"Whisper model loaded successfully on {device.upper()} with compute type {compute_type}.")
except Exception as e:
    logger.error("Error loading model: %s", str(e))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def ASR(audio):
    """Transcribe the audio into text."""
    segments, _ = model.transcribe(audio, beam_size=5,vad_filter=True,task="translate")# word_timestamps=False)
    text = " ".join(segment.text.strip() for segment in segments)
    return text

async def ask_llama(context=" "):
    PROMPT_REQUESTS = {
        "brief_medical_history": {
            "prompt": "Summarize the Brief Patient medical History from entire conversation in one sentence.",
            "format": {
                "type": "string"
            }
        },
        "chief_complaints": {
            "prompt": "List patient's chief complaints with duration and description.",
            "format": {
                "type": "object",
                "properties": {
                    "Complaint": {"type": "string"},
                    "Duration": {"type": "string"},
                    "Description": {"type": "string"}
                },
                "required": ["Complaint", "Duration", "Description"]
            }
        },
        "current_symptoms_and_medical_background": {
            "prompt": "Explain ODP/HPI, current symptoms and medicine history of patient.",
            "format": {
                "type": "string"
            }
        },
        "past_medical_history": {
            "prompt": "List patient's past medical history with diagnosis type.",
            "format": {
                "type": "object",
                "properties": {
                    "Diagnosis_Type": {
                        "type": "string",
                        "enum": ["Clinical", "Differential", "Final", "Provisional", "Suspected"]
                    },
                    "Disease": {"type": "string"}
                },
                "required": ["Diagnosis_Type", "Disease"]
            }
        },
        "hospitalization_and_surgical_history": {
            "prompt": "Mention patient's past hospitalization or surgeries with diagnosis, treatment, and admission time.",
            "format": {
                "type": "object",
                "properties": {
                    "Diagnosis": {"type": "string"},
                    "Treatment": {"type": "string"},
                    "Admission_Time": {"type": "string"}
                },
                "required": ["Diagnosis", "Treatment", "Admission_Time"]
            }
        },
        "gynecological_history": {
            "prompt": "Provide patient's gynecological history if available, if not then respond with None.",
            "format": {
                "type": "string"
            }
        },
        "lifestyle_and_social_activity": {
            "prompt": "Describe patient's physical activity, time and status.",
            "format": {
                "type": "object",
                "properties": {
                    "Physical_Activity": {"type": "string"},
                    "Time": {"type": "string"},
                    "Status": {"type": "string"}
                },
                "required": ["Physical_Activity", "Time", "Status"]
            }
        },
        "family_history": {
            "prompt": "Provide patient's relevant family medical history including relation, disease name and age from the conversation if available.",
            "format": {
                "type": "object",
                "properties": {
                    "Relation": {"type": "string"},
                    "Disease_Name": {"type": "string"},
                    "Age": {"type": "string"}
                },
                "required": ["Relation", "Disease_Name", "Age"]
            }
        },
        "allergies_and_hypersensitivities": {
            "prompt": "Mention patient's allergies with allergen, reaction type, severity and status.",
            "format": {
                "type": "object",
                "properties": {
                    "Allergy": {"type": "string"},
                    "Allergen": {"type": "string"},
                    "Type_of_Reaction": {"type": "string"},
                    "Status": {"type": "string", "enum": ["active", "passive"]},
                    "Severity": {"type": "string"}
                },
                "required": ["Allergy", "Allergen", "Type_of_Reaction", "Status", "Severity"]
            }
        }
    }

    async def query_ollama(session, field, details):
        if not isinstance(details["format"], dict):
            return field, {"error": "Invalid schema format"}

        formatted_schema = details["format"]

        payload = {
            "model": "llama3.2",
            "prompt": f"While extracting information if you do not find the data answer strictly with None. Context: {context}\n\n{details['prompt']}",
            "format": formatted_schema,
            "options": {"temperature": 0.0},
            "stream": False
        }
        start_time = time.perf_counter()
        try:
            async with session.post(OLLAMA_API_URL, json=payload) as resp:
                raw = await resp.text()
                print(f"[{field}] Raw Response:\n{raw}")
                elapsed = time.perf_counter() - start_time
                print(f"[{field}] Time taken: {elapsed:.2f}s")

                if resp.status == 200:
                    try:
                        response = await resp.json()
                        return field, json.loads(response.get("response", "{}"))
                    except Exception:
                        return field, {"error": "Invalid JSON", "raw": raw}
                else:
                    return field, {"error": f"HTTP {resp.status}", "raw": raw}
        except Exception as e:
            return field, {"error": str(e)}

    async with aiohttp.ClientSession() as session:
        start = time.perf_counter()
        tasks = [query_ollama(session, field, details) for field, details in PROMPT_REQUESTS.items()]
        results = await asyncio.gather(*tasks)
        print(f"Total time for all fields: {time.perf_counter() - start:.2f}s")

    return {field: result for field, result in results}
    
async def ask_llama1(context, query=" "):
    headers = {'Content-Type': 'application/json'}
    payload = {
        "model": "llama3.2",
        "messages": [
            {"role": "system", "content": "Assume you are an expert doctor Assitant with no name made by Artem Health. Just talk to the patient and keep it short and precise and you can answer anything."},
            {"role": "user", "content": context},
            {"role": "user", "content": query},
        ],

        "option": {
            "temperature": 0.0
        },
        "stream": False
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(OLLAMA_API_URL_2, json=payload, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("message", {}).get("content", "No response from Ollama")
                else:
                    return f"Error: {response.status}"
    except Exception as e:
        return f"Exception: {str(e)}"

import numpy as np

async def transcribe_async(audio_array):
    """Run Faster-Whisper in a background task"""
    return await asyncio.to_thread(ASR, audio_array.astype(np.float32))  # Run ASR in a separate thread

@app.websocket("/transcribe/")
async def transcribe_audio(websocket: WebSocket):
    await websocket.accept()
    audio_queue = []
    sample_rate = 16000
    buffer_size = sample_rate * 4
    try:
        while True:
            audio_bytes = await websocket.receive_bytes()

            # Ensure correct PCM chunk size
            if len(audio_bytes) % 2 != 0:
                audio_bytes += b'\x00'

            # Convert PCM to float32
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            audio_queue.append(audio_array)

            # Process when buffer is full
            if sum(len(chunk) for chunk in audio_queue) >= buffer_size:
                combined_audio = np.concatenate(audio_queue)
                audio_queue.clear()

                # Run transcription in background
                transcription_task = asyncio.create_task(transcribe_async(combined_audio))
                asyncio.create_task(send_transcription(websocket, transcription_task))

    except WebSocketDisconnect:
        print("Client disconnected")

async def send_transcription(websocket, transcription_task):
    transcription = await transcription_task  # Wait for ASR to finish
    await websocket.send_text(transcription)  # Send back to client
    
@app.websocket("/generate_summary/")
async def generate_summary_api(websocket: WebSocket):
    """WebSocket for generating summary in real-time."""
    await websocket.accept()
    
    try:
        while True:
            transcription = await websocket.receive_text()
            summary = await ask_llama(transcription)
            if isinstance(summary, dict):
                await websocket.send_text(json.dumps(summary))
            else:
                await websocket.send_text(str(summary))
    
    except WebSocketDisconnect:
        print("Client disconnected")
        await websocket.close()

@app.websocket("/talk_with_ai/")
async def talk_with_ai_api(websocket: WebSocket):
    """WebSocket for interacting with AI based on transcription."""
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            transcription = data.get("transcription")
            query = data.get("query")
            response = await ask_llama1(transcription, query)
            print(response)
            await websocket.send_text(response)
    
    except WebSocketDisconnect:
        print("Client disconnected")
        await websocket.close()

@app.post("/upload/")
async def upload_audio(file: UploadFile = File(...)):

    audio_bytes = await file.read()
    logger.info(f"Received type: {type(audio_bytes)}, Length: {len(audio_bytes)}")
    logger.info(f"File content type: {file.content_type}, filename: {file.filename}")

    try:
        audio_array, sample_rate = sf.read(BytesIO(audio_bytes))

        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=1)

        if sample_rate != 16000:
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000

        max_val = np.max(np.abs(audio_array))
        if max_val > 0:
            audio_array = audio_array / max_val
        else:
            logger.warning("Silent audio detected, skipping normalization.")

        audio_array = audio_array.astype(np.float32)

        text = ASR(audio_array)

        return {"filename": file.filename, "transcription": text}

    except Exception as e:
        logger.error(f"Failed to load/process audio: {e}")
        return {"error": str(e)}

@app.post("/upload_and_summary/")
async def upload_and_summary(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    logger.info(f"Received type: {type(audio_bytes)}, Length: {len(audio_bytes)}")
    logger.info(f"File content type: {file.content_type}, filename: {file.filename}")

    try:
        audio_array, sample_rate = sf.read(BytesIO(audio_bytes))

        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=1)

        if sample_rate != 16000:
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000

        max_val = np.max(np.abs(audio_array))
        if max_val > 0:
            audio_array = audio_array / max_val
        else:
            logger.warning("Silent audio detected, skipping normalization.")

        audio_array = audio_array.astype(np.float32)

        text = ASR(audio_array)
        summary = await ask_llama(text)
        print("Summary generated successfully.")
        return {
            "filename": file.filename,
            "final_transcription": text,
            "summary": summary
        }

    except Exception as e:
        logger.error(f"Failed to load/process audio: {e}")
        return {"error": str(e)}

beamsize = 2

@app.post("/update-beam-size/")
async def update_beam_size(new_beam_size: int = Form(...)):
    global beamsize
    try:
        beamsize = int(new_beam_size)
        return JSONResponse({"status": "success", "beam_size": beamsize, "message": f"Beam size updated to {beamsize}."})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# Create safe save path that works in .py or bundled EXE
base_dir = (
    os.path.dirname(sys.executable)
    if getattr(sys, 'frozen', False)
    else os.path.dirname(__file__)
)
SAVE_DIR = os.path.join(base_dir, "recordings")
os.makedirs(SAVE_DIR, exist_ok=True)

@app.post("/save-audio/")
async def save_audio(audio_file: UploadFile, transcription: str = Form(...),
                     live_transcription: str = Form(...), quality: str = Form(...)):
    logger.info("save_audio function called...")

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        audio_filename = f"audio_{timestamp}.wav"
        text_filename = f"transcription_{timestamp}.txt"
        live_transcription_filename = f"live_transcription_{timestamp}.txt"

        save_dir = os.path.join(SAVE_DIR, quality)
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Saving files under directory: {save_dir}")
        
        audio_path = os.path.join(save_dir, audio_filename)
        text_path = os.path.join(save_dir, text_filename)
        live_transcription_path = os.path.join(save_dir, live_transcription_filename)

        # Save audio file
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        
        # Save transcription text
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(transcription)

        with open(live_transcription_path, "w", encoding="utf-8") as f:
            f.write(live_transcription)

        return JSONResponse({
            "message": "Files saved successfully",
            "audio_path": audio_path,
            "text_path": text_path,
            "live_transcription_path": live_transcription_path,
        })
        
    except Exception as e:
        logger.error(f"Error saving files: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Failed to save files",
                "details": str(e)
            }
        )

if __name__ == "__main__":
    import uvicorn
    import logging
    from logging.config import dictConfig

    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            },
        },
        "handlers": {
            "file": {
                "class": "logging.FileHandler",
                "filename": "server.log",
                "formatter": "default",
                "level": "INFO",
            },
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": "INFO",
            },
        },
        "root": {
            "handlers": ["file", "console"],
            "level": "INFO",
        },
    }

    dictConfig(log_config)

    uvicorn.run(
        app, 
        host="0.0.0.0",
        port=8001,
        forwarded_allow_ips="*",
        ssl_certfile="cert.pem",
        ssl_keyfile="key.pem",
        log_config=log_config
    )