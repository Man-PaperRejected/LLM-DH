import asyncio
import json
import logging
import os
import io # 需要导入 io
from vosk import Model, KaldiRecognizer, SetLogLevel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File, HTTPException # 增加 UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse # 增加 JSONResponse
from fastapi.templating import Jinja2Templates
from pydub import AudioSegment # 导入 pydub
from pydub.exceptions import CouldntDecodeError # 导入 pydub 异常

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
    接收上传的音频文件，使用 pydub 进行转换，然后用 Vosk 识别。
    """
    logger.info(f"接收到上传文件: {audio_file.filename}, 类型: {audio_file.content_type}")

    if not audio_file.content_type or not audio_file.content_type.startswith("audio/"):
         logger.warning(f"上传了非音频类型文件: {audio_file.content_type}")
         raise HTTPException(status_code=400, detail="无效的文件类型，请上传音频文件。")

    # 为本次请求创建一个新的识别器
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    full_text = ""

    try:
        # 1. 读取上传文件的内容
        contents = await audio_file.read()
        logger.info(f"文件读取完毕，大小: {len(contents)} 字节")

        # 2. 使用 pydub 加载音频并转换为所需格式
        logger.info("使用 pydub 进行音频转换...")
        # 将 bytes 包装在 BytesIO 中，以便 pydub 读取
        audio_stream = io.BytesIO(contents)
        # pydub 会尝试自动检测格式 (MP3, WAV, OGG 等)
        # 需要系统安装了 ffmpeg 或 libav
        audio = AudioSegment.from_file(audio_stream)

        # 转换为单声道 (mono) 和目标采样率 (SAMPLE_RATE)
        audio = audio.set_channels(1).set_frame_rate(int(SAMPLE_RATE))
        logger.info(f"音频已转换为: 单声道, {SAMPLE_RATE} Hz")

        # 3. 获取原始 PCM 数据并进行识别
        # pydub 默认导出 16-bit PCM，这通常是 Vosk 所需的
        raw_audio_data = audio.raw_data

        # 分块处理以避免内存问题（对于非常大的文件）
        # chunk_size = 8000 # 可以调整块大小
        # for i in range(0, len(raw_audio_data), chunk_size):
        #     chunk = raw_audio_data[i:i + chunk_size]
        #     if recognizer.AcceptWaveform(chunk):
        #         result = json.loads(recognizer.Result())
        #         full_text += result.get('text', '') + " " # 拼接最终结果
        #     # else:
        #         # partial = json.loads(recognizer.PartialResult()) # 文件处理通常不需要部分结果
        #         # print(partial)

        # --- 或者一次性处理（对于不是特别巨大的文件更简单） ---
        if recognizer.AcceptWaveform(raw_audio_data):
             result = json.loads(recognizer.Result())
             full_text = result.get('text', '')
        # ---------------------------------------------------

        # 获取最后的结果
        final_result = json.loads(recognizer.FinalResult())
        full_text += final_result.get('text', '') # 拼接最后的部分

        logger.info(f"文件识别完成: {full_text.strip()}")
        return JSONResponse(content={"transcription": full_text.strip()})

    except CouldntDecodeError:
        logger.error(f"无法解码音频文件: {audio_file.filename}. 请确保安装了 ffmpeg 并且文件格式受支持。")
        raise HTTPException(status_code=400, detail="无法解码音频文件。请检查文件格式或确保服务器安装了 ffmpeg/libav。")
    except Exception as e:
        logger.error(f"处理上传文件时发生错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理音频文件时发生内部错误: {e}")
    finally:
         await audio_file.close() # 确保关闭文件句柄

# --- Add a simple health check endpoint (optional) ---
@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model is not None}

