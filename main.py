import asyncio
import json
import logging
import os
import wave # For handling potential sample rate conversion if needed
from fastapi import Request
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles # Optional: if you have separate CSS/JS
from vosk import Model, KaldiRecognizer, SetLogLevel

# --- Vosk Configuration ---
SetLogLevel(0) # Set to -1 to disable logs, 0 for info, 1 for debug
MODEL_PATH = "model/vosk-model-small-en-us-0.15" # Path to your downloaded Vosk model
SAMPLE_RATE = 16000.0 # Vosk model's expected sample rate

# --- Basic Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Load Vosk Model ---
if not os.path.exists(MODEL_PATH):
    logger.error(f"Vosk model not found at path: {MODEL_PATH}")
    exit(1)

logger.info(f"Loading Vosk model from: {MODEL_PATH}")
# Note: Loading the model can take time and memory.
# Consider loading it lazily or in a separate thread for large models in production.
model = Model(MODEL_PATH)
logger.info("Vosk model loaded successfully.")


# --- FastAPI App ---
app = FastAPI()

# --- Serve Frontend ---
# Option 1: Serve HTML directly from Python string (simple)
# (See index_html content in Step 2)

# Option 2: Serve from templates directory (better practice)
from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="html")

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    from fastapi import Request # Import here if using Jinja
    return templates.TemplateResponse("index.html", {"request": request})

# Optional: Mount static files if you have separate CSS/JS
# app.mount("/static", StaticFiles(directory="static"), name="static")


# --- WebSocket Endpoint for Transcription ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info(f"WebSocket connection established from: {websocket.client.host}")

    # Create a recognizer instance for this connection
    # It's crucial to create a new recognizer for each stream.
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    # recognizer.SetWords(True) # Uncomment if you want word timings

    last_partial = "" # Keep track of the last partial result to avoid duplicates

    try:
        while True:
            # Receive audio data (bytes) from the client
            audio_chunk = await websocket.receive_bytes()

            # --- Process Audio with Vosk ---
            # AcceptWaveform returns True if a partial or final result is available
            if recognizer.AcceptWaveform(audio_chunk):
                result_json = recognizer.Result() # Final result for a segment
                result_dict = json.loads(result_json)
                text = result_dict.get('text', '')
                if text: # Only send if there's actual text
                    logger.info(f"Final Result: {text}")
                    await websocket.send_text(json.dumps({"type": "final", "text": text}))
                    last_partial = "" # Reset partial tracking after final result
            else:
                partial_json = recognizer.PartialResult()
                partial_dict = json.loads(partial_json)
                partial_text = partial_dict.get('partial', '')

                # Send partial result ONLY if it's new and not empty
                if partial_text and partial_text != last_partial:
                    logger.debug(f"Partial Result: {partial_text}")
                    await websocket.send_text(json.dumps({"type": "partial", "text": partial_text}))
                    last_partial = partial_text # Update last sent partial

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected from: {websocket.client.host}")
        # Process any remaining buffered audio for a final result
        final_json = recognizer.FinalResult()
        final_dict = json.loads(final_json)
        final_text = final_dict.get('text', '')
        if final_text:
            logger.info(f"Final Result (on disconnect): {final_text}")
            # We can't send to a disconnected client, but we log it.
    except Exception as e:
        logger.error(f"Error during WebSocket communication: {e}")
    finally:
        # Clean up recognizer if needed (though Python GC usually handles it)
        logger.info(f"Closing connection processing for: {websocket.client.host}")

# --- Add a simple health check endpoint (optional) ---
@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model is not None}

