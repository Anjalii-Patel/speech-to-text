# main1.py
import torchaudio
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
import os
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
# OLLAMA_API_URL = "http://192.168.3.41:11434/api/generate"
# OLLAMA_API_URL = "http://ollama:11434/api/generate"

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set compute type
compute_type = "float16" if device == "cuda" else "int8"

# Optional: use more threads for CPU
kwargs = {}
if device == "cpu":
    kwargs["cpu_threads"] = os.cpu_count()

# # Get EXE's directory
# base_dir = os.path.dirname(os.path.abspath(__file__))

# # Your local model folder
# model_dir = os.path.join(base_dir, "Models")

# # Load from that folder, using model name
# model = WhisperModel(
#     "large-v2",
#     download_root=model_dir,
#     local_files_only=True,
#     device=device,
#     compute_type=compute_type,
#     **kwargs
# )
# import sys

# if getattr(sys, 'frozen', False):
#     BASE_DIR = sys._MEIPASS
# else:
#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# MODEL_DIR = os.path.join(BASE_DIR, "Model")

# # Set environment vars
# os.environ["TRANSFORMERS_CACHE"] = MODEL_DIR
# os.environ["HF_HOME"] = MODEL_DIR
# os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_DIR

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


# # Load model with correct multi-threading args
# model = WhisperModel("large-v2", 
#                      device="cpu", 
#                      compute_type="int8"
#                     #  ,cpu_threads=8
#                     )   # For multi-threading within CUDA core

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
    segments, _ = model.transcribe(audio, beam_size=2,vad_filter=True,task="translate")# word_timestamps=False)
    text = " ".join(segment.text.strip() for segment in segments)
    return text

async def ask_llama(context=" ", query=" "):
    PROMPT_REQUESTS = {
        "brief_medical_history": {
            "prompt": "Summarize the Brief Patient medical History from entire conversation in one sentence.",
            "format": {
                "type": "string"
            }
        },
        "chief_complaints": {
            "prompt": "List chief complaints with duration and description.",
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
            "prompt": "Explain ODP/HPI, current symptoms and medicine history.",
            "format": {
                "type": "string"
            }
        },
        "past_medical_history": {
            "prompt": "List past medical history with diagnosis type.",
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
            "prompt": "Mention past hospitalization or surgeries with diagnosis, treatment, and admission time.",
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
            "prompt": "Provide gynecological history.",
            "format": {
                "type": "string"
            }
        },
        "lifestyle_and_social_activity": {
            "prompt": "Describe physical activity, time and status.",
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
            "prompt": "Provide relevant family medical history including relation, disease name and age.",
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
            "prompt": "Mention allergies with allergen, reaction type, severity and status.",
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
        # print(details["format"])
        if not isinstance(details["format"], dict):
            return field, {"error": "Invalid schema format"}

        formatted_schema = details["format"]

        # formatted_schema = {
        #     "type": "object",
        #     # "properties": {
        #     #     key: {"type": value}
        #     #     for key, value in details["format"].items()
        #     # },
        #     "properties":details["format"],
        #     "required": list(details["format"].keys())
        # }
        payload = {
            "model": "llama3.2",
            "prompt": f"While extracting information if you do not find the data answer strictly with None. Context: {context}\n\n{details['prompt']}",
            "format": formatted_schema,
            "options": {"temperature": 0.0},
            "stream": False
        }
        print("Payload to ollama:\n",payload)
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
            async with session.post(OLLAMA_API_URL, json=payload, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("response", {}).get("content", "No response from Ollama")
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
            summary = await ask_llama(transcription, "")
            if isinstance(summary, dict):
                await websocket.send_text(json.dumps(summary))
            else:
                await websocket.send_text(str(summary))
            # print("Raw Response from ollama:\n",summary)
            # json_data = json.loads(summary)
            # --------await websocket.send_text(summary)
    
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

# Create safe save path that works in .py or bundled EXE
base_dir = (
    os.path.dirname(sys.executable)
    if getattr(sys, 'frozen', False)
    else os.path.dirname(__file__)
)
SAVE_DIR = os.path.join(base_dir, "recordings")
os.makedirs(SAVE_DIR, exist_ok=True)

@app.post("/save-audio/")
async def save_audio(audio_file: UploadFile, transcription: str = Form(...)):
    logger.info("save_audio function called...")
    logger.info(f"Received audio file: {audio_file.filename}, size: {audio_file.size if hasattr(audio_file, 'size') else 'unknown'}")
    logger.info(f"Received transcription length: {len(transcription)}")
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        audio_filename = f"audio_{timestamp}.wav"
        text_filename = f"transcription_{timestamp}.txt"
        
        audio_path = os.path.join(SAVE_DIR, audio_filename)
        text_path = os.path.join(SAVE_DIR, text_filename)
        
        # Save audio file
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        logger.info(f"Audio file saved to {audio_path}")
        
        # Save transcription text
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(transcription)
        logger.info(f"Transcription file saved to {text_path}")
        
        return JSONResponse({
            "message": "Files saved successfully",
            "audio_path": audio_path,
            "text_path": text_path,
            "audio_filename": audio_filename,
            "text_filename": text_filename,
            "transcription_length": len(transcription)
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

# @app.post("/upload/")
# async def upload_audio(file: UploadFile = File(...)):
#     """Upload an audio file and get the transcription."""
#     audio_bytes = await file.read()

#     print(f"Received type: {type(audio_bytes)}, Length: {len(audio_bytes)} bytes")
#     print(f"File content type: {file.content_type}, filename: {file.filename}")

#     audio_tensor, sample_rate = torchaudio.load(BytesIO(audio_bytes), normalize=True)
    
#     if audio_tensor.dim() > 1:
#         audio_tensor = audio_tensor.mean(dim=0)
    
#     if sample_rate != 16000:
#         resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
#         audio_tensor = resampler(audio_tensor)
    
#     waveform_np = audio_tensor.numpy()
#     waveform_np = waveform_np / np.max(np.abs(waveform_np))
    
#     text = ASR(waveform_np)
#     # print(text)
#     return {"filename": file.filename, "transcription": text}


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8001, forwarded_allow_ips='*',ssl_certfile="cert.pem",ssl_keyfile="key.pem")

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
