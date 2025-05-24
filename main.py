import asyncio
import json
import logging
import logging.handlers
import os
import fcntl
import io
import time # Import time for blocking sleep
import multiprocessing
from multiprocessing import Process, Queue as MpQueue # Use multiprocessing Queue and Process
from contextlib import asynccontextmanager # For lifespan manager
from llm import ds_worker, tts_worker
from dh.wav2lip.wav2lip import wav2lip
from hls import ffmpeg_worker
from vosk import Model, KaldiRecognizer, SetLogLevel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File, HTTPException, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
import cv2
# Assuming 'config.py' exists with necessary variables like:
# VOSK_LOG_LEVEL, VOSK_MODEL_PATH, VOSK_SAMPLE_RATE, TEMPLATES_DIR,
# CAMERA_INDEX, VIDEO_FPS, JPEG_QUALITY
import config # Make sure config is imported

# --- HLS Video  ---
import mimetypes
mimetypes.add_type('application/vnd.apple.mpegurl', '.m3u8')
mimetypes.add_type('video/mp2t', '.ts')

HLS_OUTPUT_DIR = "./hls_stream"  
os.makedirs(HLS_OUTPUT_DIR, exist_ok=True)

# --- Pipe File Descriptors (will be initialized in lifespan) ---
multiprocessing.set_start_method('fork', force=True)
PIPE_BUF_SIZE = 1048576  # 1MB
video_pipe_r_fd = None
video_pipe_w_fd = None
audio_pipe_r_fd = None
audio_pipe_w_fd = None

# --- Queues (Multiprocessing safe) ---
# These queues are already multiprocessing.Queue, which is correct for IPC.
ask_queue= MpQueue()
answer_queue = MpQueue()
llm_queue = MpQueue()
tts_queue = MpQueue()
log_queue = MpQueue()

active_connections: set[WebSocket] = set()
llm_stream_connections: set[WebSocket] = set()
# --- Vosk Configuration ---
SetLogLevel(config.VOSK_LOG_LEVEL if hasattr(config, 'VOSK_LOG_LEVEL') else -1) # Default if not set

# --- Basic Logging ---
stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
listener = logging.handlers.QueueListener(log_queue, stream_handler)
listener.start()

queue_handler = logging.handlers.QueueHandler(log_queue)
logger = logging.getLogger("Main")
logger.setLevel(logging.INFO)
logger.handlers = []  # 移除默认handlers，防止重复
logger.addHandler(queue_handler)

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

async def broadcast_llm_stream():
    """监控LLM队列并向连接到/llm_stream的客户端广播消息。

    该协程持续从`llm_queue`中获取LLM生成的文本片段，
    并将其序列化为JSON格式，然后通过WebSocket连接发送给所有订阅的客户端。
    它处理客户端断开连接，并允许通过在队列中发送`None`来优雅地停止。

    Raises:
        asyncio.CancelledError: 当任务被取消时引发。
        Exception: 广播过程中发生任何其他意外错误时。
    """
    
    logger.info("LLM Stream广播任务已启动") # Broadcast task for LLM stream started
    while True:
        try:
            # Get data from the LLM queue (non-blocking via asyncio.to_thread)
            message_data = await asyncio.to_thread(llm_queue.get)

            if message_data is None:  # Check for stop signal
                logger.info("LLM Stream广播器收到停止信号") # LLM Stream broadcaster received stop signal
                break

            # Ensure message_data is serializable (e.g., dict or string)
            # If it's just text, wrap it in a dict for consistency
            if isinstance(message_data, str):
                 message_payload = json.dumps({"type": "llm_answer", "text": message_data})
            elif isinstance(message_data, dict):
                 # Assume it already has 'type', 'text', etc.
                 message_payload = json.dumps(message_data)
            else:
                 logger.warning(f"LLM Stream广播器收到未知格式数据: {type(message_data)}") # LLM Stream broadcaster received unknown data format
                 continue # Skip broadcasting this item

            logger.info(f"正在向 {len(llm_stream_connections)} 个 LLM Stream 客户端广播: {message_payload}") # Broadcasting to N LLM Stream clients

            # Use list comprehension for potentially removing dead connections safely
            disconnected_clients = []
            for connection in llm_stream_connections:
                try:
                    await connection.send_text(message_payload)
                except (WebSocketDisconnect, RuntimeError) as e:
                    logger.warning(f"发送到LLM Stream客户端时出错，准备移除: {e}") # Error sending to LLM Stream client, preparing removal
                    disconnected_clients.append(connection)
                except Exception as e:
                    logger.error(f"发送到LLM Stream客户端时发生意外错误: {e}", exc_info=True) # Unexpected error sending to LLM Stream client
                    disconnected_clients.append(connection) # Also remove on unexpected errors

            # Remove dead connections *after* iteration
            for client in disconnected_clients:
                if client in llm_stream_connections:
                     llm_stream_connections.remove(client)
                     logger.info(f"已从 LLM Stream 移除断开的客户端. 当前数量: {len(llm_stream_connections)}") # Removed disconnected client from LLM Stream. Current count: N

        except asyncio.CancelledError:
             logger.info("LLM Stream广播任务被取消") # LLM Stream broadcast task cancelled
             break
        except Exception as e:
            logger.error(f"LLM Stream广播任务出错: {e}", exc_info=True) # LLM Stream broadcast task error
            await asyncio.sleep(1) # Avoid busy-looping on errors
    logger.info("LLM Stream广播任务已完成") # LLM Stream broadcast task finished

# --- Lifespan Context Manager for Process Management ---
processes = []
broadcast_task = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理FastAPI应用的生命周期，包括后台进程的启动和关闭。

    在应用启动时：
    1. 创建用于视频和音频数据传输的操作系统管道。
    2. 启动LLM（大语言模型）、TTS（文本转语音）、Wav2Lip（数字人视频生成）
       和FFmpeg（HLS流编码）等多个独立进程。
    3. 启动一个异步任务以广播LLM的实时回复到WebSocket客户端。
    4. 关闭父进程不再需要的管道端，确保子进程拥有正确的读写权限。

    在应用关闭时：
    1. 向队列发送停止信号，通知所有后台进程优雅地退出。
    2. 逐一等待并终止未能优雅退出的进程。
    3. 关闭所有队列资源。

    Args:
        app (FastAPI): FastAPI应用程序实例。

    Yields:
        None: 在所有启动操作完成后，将控制权交还给应用程序。

    Raises:
        RuntimeError: 如果管道创建失败。
    """
    
    logger.info("Application startup: Starting background processes...")

    # Pipes for DHVG -> FFmpeg communication
    try:
        video_pipe_r_fd, video_pipe_w_fd = os.pipe()
        logger.info(f"Created video pipe: R={video_pipe_r_fd}, W={video_pipe_w_fd}")
        audio_pipe_r_fd, audio_pipe_w_fd = os.pipe()
        logger.info(f"Created audio pipe: R={audio_pipe_r_fd}, W={audio_pipe_w_fd}")
        
        fcntl.fcntl(video_pipe_w_fd, fcntl.F_SETPIPE_SZ, PIPE_BUF_SIZE)
        fcntl.fcntl(video_pipe_r_fd, fcntl.F_SETPIPE_SZ, PIPE_BUF_SIZE)
        fcntl.fcntl(audio_pipe_w_fd, fcntl.F_SETPIPE_SZ, PIPE_BUF_SIZE)
        fcntl.fcntl(audio_pipe_r_fd, fcntl.F_SETPIPE_SZ, PIPE_BUF_SIZE)
        
        os.set_inheritable(video_pipe_r_fd, True)
        os.set_inheritable(video_pipe_w_fd, True)
        os.set_inheritable(audio_pipe_r_fd, True)
        os.set_inheritable(audio_pipe_w_fd, True)
        
    except OSError as e:
        logger.error(f"Failed to create pipes: {e}", exc_info=True)
        # Handle error appropriately - maybe exit?
        raise RuntimeError("Failed to create necessary pipes") from e

    # Start LLM Process
    try:
        llm_process = Process(target=ds_worker, args=(ask_queue, answer_queue, llm_queue, log_queue), daemon=True)
        llm_process.start()
        processes.append(("LLM", llm_process))
        logger.info(f"LLM process started (PID: {llm_process.pid}).")
    except:
        logger.error(f"Failed to start LLM process: {e}", exc_info=True)

    # Start TTS Process
    try:
        tts_process = Process(target=tts_worker, args=(answer_queue, tts_queue, log_queue), daemon=True)
        tts_process.start()
        processes.append(("TTS", tts_process))
        logger.info(f"TTS process started (PID: {tts_process.pid}).")
    except:
        logger.error(f"Failed to start TTS process: {e}", exc_info=True)

    # Start wav2lip Process
    try:
        # os.close(video_pipe_r_fd)c
        # os.close(audio_pipe_r_fd)
        wav2lip_process = Process(target=wav2lip, args=(tts_queue, log_queue, video_pipe_w_fd, audio_pipe_w_fd), daemon=True)
        wav2lip_process.start()
        processes.append(("wav2lip", wav2lip_process))
        os.close(video_pipe_w_fd)
        os.close(audio_pipe_w_fd)
        video_pipe_w_fd = audio_pipe_w_fd = -1
    except Exception as e:
        logger.error(f"Failed to start wav2lip process: {e}", exc_info=True)
        # Close write ends if DHVG failed to start?
        if video_pipe_w_fd is not None: os.close(video_pipe_w_fd)
        if audio_pipe_w_fd is not None: os.close(audio_pipe_w_fd)
        
    try:
        logger.info("Starting LLM answer broadcaster task...")
        broadcast_task = asyncio.create_task(broadcast_llm_stream())
        logger.info("LLM Broadcast task created")
    except:
        logger.error("LLM Broadcast task Failed")
    
    # time.sleep(45)
    # Start ffmpeg Process
    try:
        logger.info(f"PARENT: Passing FDs to ffmpeg_worker: Video={video_pipe_r_fd}, Audio={audio_pipe_r_fd}")
        # Check if these values are valid integers (e.g., >= 0)
        if not isinstance(video_pipe_r_fd, int) or video_pipe_r_fd < 0 or \
        not isinstance(audio_pipe_r_fd, int) or audio_pipe_r_fd < 0:
            logger.error("PARENT: Invalid FD values detected before passing to FFmpeg process!")
            
        output_stream_url = config.OUTPUT_STREAM_URL 
        # Parent closes the WRITE ends of the pipes it gives to FFmpeg
        # logger.debug(f"Parent closed WRITE ends of FFmpeg pipes: Video={video_pipe_w_fd}, Audio={audio_pipe_w_fd}")
        ffmpeg_process = Process(
            target=ffmpeg_worker,
            args=(video_pipe_r_fd, audio_pipe_r_fd, output_stream_url, log_queue),
            name="FFmpegProcess",
            daemon=True # Make daemon so it doesn't block exit? Or manage explicitly.
        )
        ffmpeg_process.start()
        processes.append(("FFmpeg", ffmpeg_process))
        # time.sleep(0.1)
        os.close(video_pipe_r_fd)
        os.close(audio_pipe_r_fd)
        video_pipe_r_fd = audio_pipe_r_fd = -1
    except Exception as e:
        logger.error(f"Failed to start FFmpeg process: {e}", exc_info=True)
        # Close write ends if DHVG failed to start?
        if video_pipe_r_fd is not None: os.close(video_pipe_r_fd)
        if audio_pipe_r_fd is not None: os.close(audio_pipe_r_fd)    
    
    

    yield # Application runs here

    logger.info("Application shutdown: Stopping background processes...")
    # Send sentinel value (None) to signal processes to stop
    
    if broadcast_task and not broadcast_task.done():
        logger.info("正在通知LLM回答广播器停止...")
        try:
            await asyncio.to_thread(llm_queue.put, None) 
            broadcast_task.cancel()
            try:
                await asyncio.wait_for(broadcast_task, timeout=5.0)
                logger.info("LLM回答广播任务已停止")
            except asyncio.TimeoutError:
                logger.warning("LLM回答广播任务未按时完成")
            except asyncio.CancelledError:
                 logger.info("LLM回答广播任务已被取消")
        except Exception as e:
            logger.error(f"停止LLM回答广播任务时出错: {e}", exc_info=True)
    
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

@app.get("/live/stream.m3u8")
async def get_m3u8():
    """
    专门处理 HLS manifest 文件请求，强制设置禁止缓存的头。
    确保每次请求都能获取到最新的HLS切片列表
    """
    m3u8_path = os.path.join(HLS_OUTPUT_DIR, "stream.m3u8")
    logger.debug(f"Request for /live/stream.m3u8, checking path: {m3u8_path}") # 添加 Debug 日志

    if not os.path.exists(m3u8_path):
        logger.warning(f"m3u8 file not found at: {m3u8_path}")
        return Response(status_code=404, content="Not Found")

    try:
        with open(m3u8_path, "rb") as f:
            content = f.read()
        
        # ---> 关键：设置禁止缓存的响应头 <---
        headers = {
            'Cache-Control': 'no-cache, no-store, must-revalidate, max-age=0',
            'Pragma': 'no-cache', # 兼容旧版 HTTP/1.0
            'Expires': '0', # 兼容旧版 HTTP/1.0
            'Content-Type': 'application/vnd.apple.mpegurl' # 正确的 MIME 类型
        }
        logger.debug(f"Serving m3u8 file with no-cache headers. Size: {len(content)} bytes.")
        return Response(content=content, headers=headers, status_code=200)
    except Exception as e:
        logger.error(f"Error reading or serving m3u8 file: {e}", exc_info=True)
        return Response(status_code=500, content="Internal Server Error")
    
app.mount("/live", StaticFiles(directory=HLS_OUTPUT_DIR), name="hls")
# --- Serve Frontend ---
templates = Jinja2Templates(directory=config.TEMPLATES_DIR)

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """渲染并返回主页面的HTML模板。

    Args:
        request (Request): 客户端的HTTP请求对象，用于Jinja2模板上下文。

    Returns:
        fastapi.responses.TemplateResponse: 渲染后的HTML页面响应。
    """
    
    return templates.TemplateResponse("index.html", {"request": request})

# --- WebSocket Endpoint for Transcription ---
@app.websocket("/mic")
async def mic_endpoint(websocket: WebSocket):
    """处理来自客户端的实时麦克风音频WebSocket连接。

    该端点接收客户端发送的原始音频字节流，使用Vosk语音识别模型
    （在单独的线程中运行以避免阻塞）进行实时转录。
    它将部分和最终的转录结果通过WebSocket发送回客户端，
    并将最终的识别文本放入`ask_queue`供LLM进程处理。

    Args:
        websocket (WebSocket): 活跃的WebSocket连接对象。

    Raises:
        WebSocketDisconnect: 客户端断开连接时。
        Exception: 在WebSocket通信或Vosk处理过程中发生任何其他意外错误时。
    """
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

@app.websocket("/llm_stream")
async def llm_stream_endpoint(websocket: WebSocket):
    """处理用于接收LLM实时回复的WebSocket连接。

    此端点主要用于将从`llm_queue`中获取的LLM回复广播给连接的客户端。
    客户端连接后会被添加到`llm_stream_connections`集合中，
    并保持开放状态以接收后续广播的消息。

    Args:
        websocket (WebSocket): 活跃的WebSocket连接对象。

    Raises:
        WebSocketDisconnect: 客户端断开连接时。
        Exception: 在WebSocket端点操作过程中发生任何其他意外错误时。
    """
    
    await websocket.accept()
    client_host = websocket.client.host if websocket.client else "Unknown"
    logger.info(f"LLM Stream WebSocket 连接已建立，来自: {client_host}") # LLM Stream WebSocket connection established from...
    llm_stream_connections.add(websocket) # Add to the *LLM stream* specific set
    logger.info(f"LLM Stream WebSocket {client_host} ADDED. 总计 LLM Stream 连接数: {len(llm_stream_connections)}") # LLM Stream WebSocket ADDED. Total LLM Stream connections...
    try:
        # Keep the connection open, wait for disconnect
        while True:
            # This endpoint is primarily for sending, but we need to await
            # something to detect disconnects. Receiving text is standard.
            # We don't actually process received data here.
            data = await websocket.receive_text()
            logger.debug(f"LLM Stream 端点收到消息 (通常忽略): {data}") # LLM Stream endpoint received message (usually ignored)

    except WebSocketDisconnect:
        logger.info(f"LLM Stream WebSocket 连接已断开，来自: {client_host}") # LLM Stream WebSocket connection disconnected from...
    except Exception as e:
        logger.error(f"LLM Stream WebSocket 端点出错: {e}", exc_info=True) # LLM Stream WebSocket endpoint error
    finally:
        if websocket in llm_stream_connections:
            logger.info(f"正在从 LLM Stream 连接中移除 WebSocket {client_host}. 移除前总数: {len(llm_stream_connections)}") # Removing WebSocket from LLM Stream connections. Total before remove...
            llm_stream_connections.remove(websocket)
            logger.info(f"已从 LLM Stream 连接移除 WebSocket {client_host}. 当前总数: {len(llm_stream_connections)}") # Removed WebSocket from LLM Stream connections. Current total...

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
    """一个异步生成器，用于捕获摄像头视频流并将其编码为MJPEG格式。（已遗弃）

    该函数从指定摄像头（配置中定义）捕获视频帧，
    将其编码为JPEG图像，并以`multipart/x-mixed-replace`格式的字节流形式产出，
    适合作为HTTP视频流响应。它会尝试维持配置中指定的帧率（FPS）。

    Yields:
        bytes: 格式化为MJPEG流部分的JPEG图像字节数据。

    Raises:
        asyncio.CancelledError: 当视频流任务被取消时。
        Exception: 在摄像头操作或图像编码过程中发生任何其他意外错误时。
    """
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
    """提供实时摄像头视频流的HTTP端点。（已遗弃）

    该端点返回一个`StreamingResponse`，其内容由`gen_camera`生成器提供，
    媒体类型设置为`multipart/x-mixed-replace`，适用于在浏览器中显示MJPEG流。

    Returns:
        fastapi.responses.StreamingResponse: 实时摄像头视频流的响应。
    """
    logger.info(f"Request received for video stream /video_feed (Source: Camera {config.CAMERA_INDEX})")
    return StreamingResponse(gen_camera(), media_type='multipart/x-mixed-replace; boundary=frame')


# --- Health Check Endpoint (Unchanged) ---
@app.get("/health")
async def health_check():
    """提供应用程序的健康检查状态。

    此端点返回一个JSON响应，指示应用程序是否正常运行，Vosk模型是否已加载，
    以及所有后台多进程（LLM、TTS、Wav2Lip、FFmpeg）是否仍然存活。

    Returns:
        dict: 包含应用程序状态、Vosk模型加载状态和各个后台进程存活状态的字典。
    """
    
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

