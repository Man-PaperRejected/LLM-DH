from multiprocessing import Process, Queue as MpQueue # Use multiprocessing Queue and Process
import time
import asyncio
import io
import numpy as np
import os 
import logging
import logging.handlers
from edge_tts import Communicate, VoicesManager
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
import wave
# 提取音频并保存为wav文件


def test_wav(tts_q: MpQueue,
             log_q: MpQueue,
             output_filename: str = "output_test.wav", # 输出文件名
             channels: int = 1,      # 声道数 (应与 tts_worker 输出匹配)
             sampwidth: int = 2,     # 样本宽度 (字节, 16位 = 2) (应与 tts_worker 输出匹配)
             framerate: int = 16000  # 采样率 (Hz) (应与 tts_worker 输出匹配)
            ):
    
    queue_handler = logging.handlers.QueueHandler(log_q)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []
    root_logger.addHandler(queue_handler)
    
    logger = logging.getLogger(f"test_wav_{os.getpid()}")
    logger.info("Test Wav Save, read data from tts_q")
    
    while True:
        try:
            audio_bytes = tts_q.get()
            
            if audio_bytes is None:
                logger.warning("Received stop signal (None) from tts_q.")
                break
            
            audio_np_array = np.frombuffer(audio_bytes, dtype=np.int16)
            logger.info("Received WAV data from tts_q")
        except EOFError:
            logger.warning("tts_q connection closed unexpectedly (EOFError). Stopping.")
            break
        except BrokenPipeError:
            logger.error("tts_q connection broke (BrokenPipeError). Stopping.")
            break
        except Exception as e:
            logger.error(f"Error getting data from tts_q: {e}", exc_info=True)
            # 根据需要决定是否继续或停止
            break # 通常在未知错误时停止比较安全
        
        try:
            with wave.open(output_filename, 'wb') as wf:
                wf.setnchannels(channels)      
                wf.setsampwidth(sampwidth)      
                wf.setframerate(framerate)     
                wf.writeframes(audio_bytes)

            logger.info(f"Successfully saved audio to {output_filename}")

        except wave.Error as e:
            logger.error(f"Error writing WAV file using wave module: {e}", exc_info=True)
        except IOError as e:
            logger.error(f"Error opening or writing file {output_filename}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"An unexpected error occurred during WAV file saving: {e}", exc_info=True)
