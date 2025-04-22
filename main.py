import asyncio
import json
import logging
import logging.handlers
import os
import io
import time # Import time for blocking sleep
from multiprocessing import Process, Queue as MpQueue # Use multiprocessing Queue and Process
from contextlib import asynccontextmanager # For lifespan manager
from llm import ds_worker, tts_worker, test_wav
from vosk import Model, KaldiRecognizer, SetLogLevel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
import cv2
# Assuming 'config.py' exists with necessary variables like:
# VOSK_LOG_LEVEL, VOSK_MODEL_PATH, VOSK_SAMPLE_RATE, TEMPLATES_DIR,
# CAMERA_INDEX, VIDEO_FPS, JPEG_QUALITY
import config # Make sure config is imported

# --- Queues (Multiprocessing safe) ---
# These queues are already multiprocessing.Queue, which is correct for IPC.
ask_queue= MpQueue()
answer_queue = MpQueue()
tts_queue = MpQueue()
log_queue = MpQueue()

# --- Vosk Configuration ---
SetLogLevel(config.VOSK_LOG_LEVEL if hasattr(config, 'VOSK_LOG_LEVEL') else -1) # Default if not set

# --- Basic Logging ---
stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
listener = logging.handlers.QueueListener(log_queue, stream_handler)
listener.start()

logger = logging.getLogger(__name__)

# --- Load Vosk Model ---
if not os.path.exists(config.VOSK_MODEL_PATH):
    logger.error(f"Vosk model not found at path: {config.VOSK_MODEL_PATH}")
    exit(1)

logger.info(f"Loading Vosk model from: {config.VOSK_MODEL_PATH}")
# Model loading is blocking, done at startup. Fine for now.
try:
    model = Model(config.VOSK_MODEL_PATH)
    logger.info("Vosk model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Vosk model: {e}", exc_info=True)
    exit(1)


# --- Lifespan Context Manager for Process Management ---
processes = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages the startup and shutdown of background processes."""
    logger.info("Application startup: Starting background processes...")

    # Configure logging for multiprocessing if needed (can be tricky)
    # Consider using QueueHandler for centralized logging from processes

    # Start LLM Process
    llm_process = Process(target=ds_worker, args=(ask_queue, answer_queue, log_queue), daemon=True)
    llm_process.start()
    processes.append(("LLM", llm_process))
    logger.info(f"LLM process started (PID: {llm_process.pid}).")

    # Start TTS Process
    tts_process = Process(target=tts_worker, args=(answer_queue, tts_queue, log_queue), daemon=True)
    tts_process.start()
    processes.append(("TTS", tts_process))
    logger.info(f"TTS process started (PID: {tts_process.pid}).")
    
    # Start wav2lip Process
    wav2lip_process = Process(target=test_wav, args=(tts_queue, log_queue))
    wav2lip_process.start()
    processes.append(("wav2lip", wav2lip_process))

    yield # Application runs here

    logger.info("Application shutdown: Stopping background processes...")
    # Send sentinel value (None) to signal processes to stop
    try:
         ask_queue.put(None)
         answer_queue.put(None)
         tts_queue.put(None)
    except Exception as e:
         logger.error(f"Error putting sentinel values in queues: {e}")

    for name, process in processes:
        try:
            # Wait for a short time for graceful exit
            process.join(timeout=5)
            if process.is_alive():
                logger.warning(f"Process '{name}' did not exit gracefully, terminating.")
                process.terminate() # Force terminate if still alive
                process.join(timeout=1) # Wait briefly for termination
            else:
                 logger.info(f"Process '{name}' (PID: {process.pid}) stopped gracefully.")
        except Exception as e:
             logger.error(f"Error stopping process '{name}': {e}", exc_info=True)

    # Close queues (optional, helps prevent hangs in some scenarios)
    ask_queue.close()
    answer_queue.close()
    ask_queue.join_thread()
    answer_queue.join_thread()
    tts_queue.close()
    tts_queue.join_thread()
    logger.info("Background processes stopped and queues closed.")


# --- FastAPI App ---
# Pass the lifespan manager to the FastAPI constructor
app = FastAPI(lifespan=lifespan)

# --- Serve Frontend ---
templates = Jinja2Templates(directory=config.TEMPLATES_DIR)

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --- WebSocket Endpoint for Transcription ---
@app.websocket("/mic")
async def mic_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_host = websocket.client.host
    logger.info(f"WebSocket connection established from: {client_host}")

    # Create a recognizer instance for this connection
    try:
        recognizer = KaldiRecognizer(model, config.VOSK_SAMPLE_RATE)
        # recognizer.SetWords(True) # Uncomment if you want word timings
    except Exception as e:
        logger.error(f"Failed to create KaldiRecognizer for {client_host}: {e}")
        await websocket.close(code=1011) # Internal error
        return

    last_partial = "" # Keep track of the last partial result to avoid duplicates

    try:
        while True:
            # Receive audio data (bytes) from the client
            audio_chunk = await websocket.receive_bytes()

            # --- Process Audio with Vosk (in thread) ---
            # Run the blocking Vosk functions in a separate thread
            def process_chunk(rec, chunk):
                """Blocking function to process audio chunk."""
                if rec.AcceptWaveform(chunk):
                    res = json.loads(rec.Result())
                    return "final", res.get('text', '')
                else:
                    partial_res = json.loads(rec.PartialResult())
                    return "partial", partial_res.get('partial', '')

            try:
                 # Use asyncio.to_thread to run the blocking function
                 result_type, text = await asyncio.to_thread(process_chunk, recognizer, audio_chunk)

                 if result_type == "final" and text:
                     logger.info(f"Final Result ({client_host}): {text}")
                     await websocket.send_text(json.dumps({"type": "final", "text": text}))
                     # Put final recognized text onto the queue for the LLM process
                     await asyncio.to_thread(ask_queue.put, text) # Use to_thread for queue put
                     logger.info(f"Sent to LLM queue: '{text}'")
                     last_partial = "" # Reset partial tracking

                 elif result_type == "partial" and text and text != last_partial:
                     logger.debug(f"Partial Result ({client_host}): {text}")
                     await websocket.send_text(json.dumps({"type": "partial", "text": text}))
                     last_partial = text # Update last sent partial

            except Exception as thread_e:
                 # Catch errors from within the thread execution
                 logger.error(f"Error during Vosk processing thread for {client_host}: {thread_e}", exc_info=True)
                 # Decide if the connection should be closed or just log
                 # await websocket.close(code=1011)
                 # break


    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected from: {client_host}")
        # Process any remaining buffered audio for a final result (in thread)
        def final_process(rec):
            final_res = json.loads(rec.FinalResult())
            return final_res.get('text', '')

        try:
            # Run final processing in a thread
            final_text = await asyncio.to_thread(final_process, recognizer)
            if final_text:
                logger.info(f"Final Result (on disconnect from {client_host}): {final_text}")
                # Also send this last bit to LLM if needed
                await asyncio.to_thread(ask_queue.put, final_text)
                logger.info(f"Sent final (on disconnect) to LLM queue: '{final_text}'")
        except Exception as final_e:
             logger.error(f"Error during final Vosk processing on disconnect for {client_host}: {final_e}", exc_info=True)

    except Exception as e:
        # Catch other WebSocket errors
        logger.error(f"Error during WebSocket communication with {client_host}: {e}", exc_info=True)
    finally:
        # Clean up recognizer if needed (though Python GC usually handles it)
        logger.info(f"Closing connection processing for: {client_host}")
        # Ensure websocket is closed if not already
        try:
            await websocket.close()
        except RuntimeError: # Can happen if already closed
            pass


# --- 文件上传识别路由 (Modified to put result on queue) ---
@app.post("/upload_audio/", response_class=JSONResponse)
async def handle_audio_upload(audio_file: UploadFile = File(...)):
    """
    Receives uploaded audio, converts with pydub, transcribes with Vosk (non-blocking),
    and puts the result onto the ask_queue.
    """
    logger.info(f"Received uploaded file: {audio_file.filename}, Type: {audio_file.content_type}")

    if not audio_file.content_type or not audio_file.content_type.startswith("audio/"):
        logger.warning(f"Uploaded non-audio file type: {audio_file.content_type}")
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file.")

    # Create a NEW recognizer for this specific request
    # Crucial: Do not reuse recognizers across different audio streams/files
    try:
        recognizer = KaldiRecognizer(model, config.VOSK_SAMPLE_RATE)
    except Exception as e:
        logger.error(f"Failed to create KaldiRecognizer for upload {audio_file.filename}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error: Could not initialize recognizer.")

    full_text = ""

    try:
        contents = await audio_file.read()
        logger.info(f"File read complete, size: {len(contents)} bytes")

        # --- Process Audio in Thread ---
        def process_audio_sync(audio_data, rec):
            try:
                logger.info("Using pydub for audio conversion (in thread)...")
                audio_stream = io.BytesIO(audio_data)
                # Explicitly provide format if possible, otherwise pydub guesses
                # Example: audio = AudioSegment.from_file(audio_stream, format="wav")
                try:
                    audio = AudioSegment.from_file(audio_stream) # Blocking decode
                except CouldntDecodeError as cd_err:
                     logger.error(f"Pydub failed to decode {audio_file.filename}. Ensure ffmpeg/libav is installed and supports the format. Error: {cd_err}")
                     raise HTTPException(status_code=400, detail=f"Could not decode audio file '{audio_file.filename}'. Check format or server setup (ffmpeg/libav).") from cd_err

                # Ensure correct sample rate and channels for Vosk
                target_sr = int(config.VOSK_SAMPLE_RATE)
                if audio.frame_rate != target_sr or audio.channels != 1:
                    audio = audio.set_channels(1).set_frame_rate(target_sr) # Blocking CPU
                    logger.info(f"Audio converted to: Mono, {target_sr} Hz (in thread)")
                else:
                     logger.info(f"Audio already in correct format: Mono, {target_sr} Hz (in thread)")

                raw_audio_data = audio.raw_data # Blocking

                logger.info("Starting Vosk recognition (in thread)...")
                rec.AcceptWaveform(raw_audio_data) # Blocking CPU
                result = json.loads(rec.Result()) # Blocking CPU
                _full_text = result.get('text', '')

                # Get final result (though Result() often contains everything for complete files)
                # final_result = json.loads(rec.FinalResult()) # Blocking CPU
                # _full_text += final_result.get('text', '') # Usually empty if Result() was called after full waveform

                logger.info(f"Vosk recognition complete (in thread). Text found: {bool(_full_text)}")
                return _full_text.strip()

            except HTTPException: # Re-raise HTTP exceptions from inner scope
                 raise
            except Exception as inner_e:
                 logger.error(f"Error processing audio in thread for {audio_file.filename}: {inner_e}", exc_info=True)
                 raise HTTPException(status_code=500, detail=f"Internal error during audio processing: {inner_e}") from inner_e

        # Run the synchronous processing function in a thread
        full_text = await asyncio.to_thread(process_audio_sync, contents, recognizer)

        if full_text:
            # Put the result onto the multiprocessing queue (using to_thread for the put)
            await asyncio.to_thread(ask_queue.put, full_text)
            logger.info(f"File recognition complete: '{full_text}'. Sent to LLM queue.")
        else:
            logger.warning(f"Uploaded file '{audio_file.filename}' did not produce any recognized text.")

        return JSONResponse(content={"transcription": full_text if full_text else "No text recognized."})

    except HTTPException as http_exc:
        # Re-raise HTTP exceptions directly
        raise http_exc
    except Exception as e:
        logger.error(f"Error handling uploaded file '{audio_file.filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing uploaded audio file: {e}")
    finally:
        await audio_file.close() # Ensure file handle is closed

# --- Video Streaming Endpoint (Unchanged conceptually) ---
async def gen_camera():
    # Use camera index from config
    cap = None # Initialize to None
    try:
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        if not cap.isOpened():
            logger.error(f"Cannot open camera index: {config.CAMERA_INDEX}")
            # Yield an error message or image? For now, just stop.
            return
        logger.info(f"Camera {config.CAMERA_INDEX} opened successfully.")
        # Use FPS and quality from config
        frame_delay = 1.0 / config.VIDEO_FPS if config.VIDEO_FPS > 0 else 0.033 # Default to ~30fps if 0

        while True:
            start_time = asyncio.get_event_loop().time()

            # Run blocking cap.read() in a thread
            ret, frame = await asyncio.to_thread(cap.read)

            if not ret or frame is None:
                logger.warning(f"Failed to grab frame from camera {config.CAMERA_INDEX}. Retrying...")
                # Avoid busy-looping if the camera fails continuously
                await asyncio.sleep(0.5)
                # Optionally try reopening the camera here?
                continue

            # Run blocking cv2.imencode() in a thread
            encode_ret, jpeg = await asyncio.to_thread(
                cv2.imencode, '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY]
            )

            if not encode_ret:
                logger.warning("Failed to encode frame to JPEG.")
                continue

            # Yield the frame bytes for the streaming response
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
            )

            # Calculate sleep duration to approximate target FPS
            processing_time = asyncio.get_event_loop().time() - start_time
            sleep_duration = max(0, frame_delay - processing_time)
            await asyncio.sleep(sleep_duration)

    except asyncio.CancelledError:
         logger.info(f"Video stream for camera {config.CAMERA_INDEX} cancelled.")
    except Exception as e:
        logger.error(f"Error in video stream generator for camera {config.CAMERA_INDEX}: {e}", exc_info=True)
    finally:
        if cap and cap.isOpened():
            logger.info(f"Releasing camera {config.CAMERA_INDEX} resources...")
            # Run blocking cap.release() in a thread
            await asyncio.to_thread(cap.release)
            logger.info(f"Camera {config.CAMERA_INDEX} resources released.")
        else:
             logger.info(f"Camera {config.CAMERA_INDEX} was not open or already released.")


@app.get("/video_feed")
async def video_feed():
    logger.info(f"Request received for video stream /video_feed (Source: Camera {config.CAMERA_INDEX})")
    return StreamingResponse(gen_camera(), media_type='multipart/x-mixed-replace; boundary=frame')


# --- Health Check Endpoint (Unchanged) ---
@app.get("/health")
async def health_check():
    # Basic check: app is running and model seems loaded
    model_status = "loaded" if 'model' in globals() and model is not None else "not loaded"
    # Check process health (simple check if they are alive)
    process_status = {}
    for name, process in processes:
         process_status[f"{name}_process_alive"] = process.is_alive() if process else False

    return {"status": "ok", "model_status": model_status, **process_status}

# --- Main execution block (for running with uvicorn directly) ---
# if __name__ == "__main__":
#     import uvicorn
#     # Note: Running directly like this might have issues with multiprocessing
#     # on some OSes (especially Windows). It's generally better to run with
#     # uvicorn command line: uvicorn your_module_name:app --reload
#     logger.info("Starting FastAPI application with uvicorn...")
#     uvicorn.run(app, host="0.0.0.0", port=8000)

