import asyncio
import json
import logging
import os
import io 
from vosk import Model, KaldiRecognizer, SetLogLevel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File, HTTPException # 增加 UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse 
from fastapi.templating import Jinja2Templates
from pydub import AudioSegment 
from pydub.exceptions import CouldntDecodeError
import cv2
# --- Vosk Configuration ---
SetLogLevel(0) # Set to -1 to disable logs, 0 for info, 1 for debug
MODEL_PATH = "model/vosk-model-en-us-0.22" # Path to your downloaded Vosk model
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

# --- 新增：文件上传识别路由 ---
@app.post("/upload_audio/", response_class=JSONResponse)
async def handle_audio_upload(audio_file: UploadFile = File(...)):
    """
    接收上传的音频文件，使用 pydub 进行转换，然后用 Vosk 识别 (非阻塞方式)。
    """
    logger.info(f"接收到上传文件: {audio_file.filename}, 类型: {audio_file.content_type}")

    if not audio_file.content_type or not audio_file.content_type.startswith("audio/"):
        logger.warning(f"上传了非音频类型文件: {audio_file.content_type}")
        raise HTTPException(status_code=400, detail="无效的文件类型，请上传音频文件。")

    # 为本次请求创建一个新的识别器
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    full_text = ""

    try:
        # 1. 读取上传文件的内容 (async)
        contents = await audio_file.read()
        logger.info(f"文件读取完毕，大小: {len(contents)} 字节")

        # --- 将阻塞的 pydub 和 Vosk 操作移至线程 ---
        def process_audio_sync(audio_data):
            nonlocal full_text # Allow modifying the outer scope variable
            try:
                logger.info("使用 pydub 进行音频转换 (in thread)...")
                audio_stream = io.BytesIO(audio_data)
                audio = AudioSegment.from_file(audio_stream) # Blocking decode
                audio = audio.set_channels(1).set_frame_rate(int(SAMPLE_RATE)) # Blocking CPU
                logger.info(f"音频已转换为: 单声道, {SAMPLE_RATE} Hz (in thread)")
                raw_audio_data = audio.raw_data # Blocking

                logger.info("开始 Vosk 识别 (in thread)...")
                # 处理整个音频
                recognizer.AcceptWaveform(raw_audio_data) # Blocking CPU
                result = json.loads(recognizer.Result()) # Blocking CPU
                _full_text = result.get('text', '')

                # 获取可能残留的最后部分
                final_result = json.loads(recognizer.FinalResult()) # Blocking CPU
                _full_text += final_result.get('text', '')
                return _full_text.strip()

            except CouldntDecodeError as cd_err:
                 logger.error(f"无法解码音频文件 (in thread): {audio_file.filename}. {cd_err}")
                 # Re-raise a specific exception type or return an error indicator
                 raise HTTPException(status_code=400, detail="无法解码音频文件。请检查文件格式或确保服务器安装了 ffmpeg/libav。") from cd_err
            except Exception as inner_e:
                 logger.error(f"处理音频线程内部错误: {inner_e}", exc_info=True)
                 raise HTTPException(status_code=500, detail=f"处理音频时发生内部错误: {inner_e}") from inner_e


        # 使用 asyncio.to_thread 运行同步处理函数
        full_text = await asyncio.to_thread(process_audio_sync, contents)

        logger.info(f"文件识别完成: {full_text}")
        return JSONResponse(content={"transcription": full_text})

    except HTTPException as http_exc:
        # Re-raise HTTP exceptions directly
        raise http_exc
    except Exception as e:
        logger.error(f"处理上传文件时发生错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理音频文件时发生外部错误: {e}")
    finally:
        await audio_file.close() # 确保关闭文件句柄
         
async def gen_camera():
    """异步生成器，用于非阻塞地读取和编码摄像头帧"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("无法打开摄像头")
        # 在异步生成器中引发异常可能不会很好地被框架捕获，
        # 更好的方法是 yield 一个错误指示或提前返回。
        # 这里我们先记录错误并退出生成器。
        return # 或者 yield b'error: could not open camera'

    logger.info("摄像头已打开")
    try:
        while True:
            # 在线程中运行阻塞的 cap.read()
            ret, frame = await asyncio.to_thread(cap.read)
            if not ret:
                logger.warning("无法从摄像头读取帧，可能已断开连接")
                await asyncio.sleep(0.5) # 等待一下再试
                continue

            # 在线程中运行阻塞的 cv2.imencode()
            encode_ret, jpeg = await asyncio.to_thread(cv2.imencode, '.jpg', frame)
            if not encode_ret:
                logger.warning("无法将帧编码为 JPEG")
                continue

            # 异步地 yield 帧数据
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
            )
            # 短暂 sleep 以允许其他任务运行，避免 CPU 占用过高
            # 同时也控制了帧率（大约）
            await asyncio.sleep(0.03) # 大约 30fps，可以调整

    except Exception as e:
        logger.error(f"视频流生成过程中发生错误: {e}", exc_info=True)
    finally:
        logger.info("正在释放摄像头资源...")
        # 确保 cap.release() 也在线程中运行（如果它可能阻塞的话）
        # 通常 release 很快，但以防万一
        await asyncio.to_thread(cap.release)
        logger.info("摄像头资源已释放")


@app.get("/video_feed")
async def video_feed():
    """提供视频流的端点"""
    logger.info("请求视频流")
    return StreamingResponse(gen_camera(), media_type='multipart/x-mixed-replace; boundary=frame')


# --- Add a simple health check endpoint (optional) ---
@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model is not None}

