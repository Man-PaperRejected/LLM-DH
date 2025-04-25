# config.py
import logging
import os

# --- General ---
# Get the directory where this config file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Set to True for development (e.g., enable --reload)
DEBUG_MODE = True

# --- Logging ---
LOG_LEVEL = logging.INFO  # Or logging.DEBUG for more detail
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# --- Vosk ASR ---
# Assumes 'model' directory is at the same level as this config file or main.py
# Adjust if your structure is different
VOSK_MODEL_DIR = os.path.join(BASE_DIR, "model") # Directory containing language models
VOSK_MODEL_NAME = "vosk-model-en-us-0.22"      # Specific model folder within VOSK_MODEL_DIR
VOSK_MODEL_PATH = os.path.join(VOSK_MODEL_DIR, VOSK_MODEL_NAME)
VOSK_SAMPLE_RATE = 16000.0
VOSK_LOG_LEVEL = 0  # -1 for none, 0 for info, 1 for debug

# --- FastAPI Server ---
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000

# --- Frontend ---
TEMPLATES_DIR = "html" # Directory for Jinja2 templates

# --- Video Streaming ---
CAMERA_INDEX = 0        # 0 for default camera, adjust if needed
VIDEO_FPS = 30          # Target frames per second for the stream
JPEG_QUALITY = 80       # Quality for encoding video frames (0-100)

# --- TTS Setting ---
TTS_VOICE = "zh-CN-YunxiNeural"  
TTS_SPEED = "-20%"                # (-50% to +100%）
TTS_VOLUME = "+10%"               # （-50% to +50%）
TTS_PITCH = "+0Hz"              # （-50Hz to +50Hz）

# --- Audio Processing ---
# Ensure ffmpeg/libav is installed if needed by pydub for specific formats
# Pydub uses these for non-WAV formats primarily

# --- Queues (No configurable values here, just used internally) ---

# --- HLS
OUTPUT_STREAM_URL = "hls_stream/stream.m3u8"
DHVG_VIDEO_WIDTH = 536
DHVG_VIDEO_HEIGHT = 960
DHVG_VIDEO_FRAMERATE = 24
DHVG_VIDEO_PIXFMT = 'bgr24'

DHVG_AUDIO_FORMAT = 's16le'
DHVG_AUDIO_SAMPLE_RATE = 16000
DHVG_AUDIO_CHANNELS = 1

FFMPEG_VCODEC = 'libx264'
FFMPEG_PIXFMT = 'yuv420p'
FFMPEG_PRESET = 'veryfast'
FFMPEG_GOP = 50
FFMPEG_VBITRATE = '2500k'
FFMPEG_ACODEC = 'aac'
FFMPEG_ABITRATE = '128k'
FFMPEG_ARATE = 44100
FFMPEG_ACHANNELS = 2
FFMPEG_BUFSIZE = '5000k'
FFMPEG_FORMAT = 'hls'
