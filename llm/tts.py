from multiprocessing import Process, Queue as MpQueue # Use multiprocessing Queue and Process
import time
import asyncio
import io
import os 
import logging
import logging.handlers
from edge_tts import Communicate, VoicesManager
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
    
# --- Helper async function for TTS synthesis and initial processing ---
async def _synthesize_and_get_raw_audio(text: str, voice_name: str, logger: logging.Logger) -> tuple[bytes | None, list | None]:
    """
    Synthesizes text using edge-tts, collects all audio chunks and word boundaries.
    Returns the combined raw audio bytes (in original edge-tts format, likely mp3)
    and the list of word boundaries.
    Returns (None, None) on failure during synthesis.
    """
    logger.info(f"Starting TTS synthesis for text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    communicate = Communicate(text, voice_name)
    audio_chunks = []
    word_boundaries = []

    try:
        async for chunk in communicate.stream():
            chunk_type = chunk["type"]
            if chunk_type == "audio":
                chunk_data = chunk["data"]
                if chunk_data:
                    audio_chunks.append(chunk_data)
            elif chunk_type == "WordBoundary":
                word_boundaries.append(chunk)
            # Ignore other chunk types for now

        if not audio_chunks:
            logger.warning("TTS synthesis resulted in no audio data.")
            return None, None

        # Combine all received audio chunks
        full_audio_bytes = b"".join(audio_chunks)
        logger.info(f"TTS synthesis complete. Raw audio size: {len(full_audio_bytes)} bytes. Got {len(word_boundaries)} word boundaries.")
        return full_audio_bytes, word_boundaries

    except Exception as e:
        logger.error(f"Error during edge-tts streaming for text '{text[:30]}...': {e}", exc_info=True)
        return None, None

# --- The tts_worker function ---
def tts_worker(answer_q: MpQueue, tts_q: MpQueue, log_q: MpQueue,
               voice_name: str = "en-US-MichelleNeural", # Example voice
               target_samplerate: int = 16000,
               target_channels: int = 1, # Mono
               target_sample_width: int = 2 # 16-bit (2 bytes)
               ):
    """
    Target function for the TTS consumer process.
    1. Gets text from answer_q.
    2. Synthesizes full audio and word boundaries using edge-tts.
    3. Decodes and converts audio to target PCM format (16k, 16b, mono).
    4. Puts the processed PCM audio bytes into tts_q.
    """
    # --- Logging Setup ---
    queue_handler = logging.handlers.QueueHandler(log_q)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []  # 移除默认handlers，防止重复
    root_logger.addHandler(queue_handler)
    
    logger = logging.getLogger(f"TTS_Worker_{os.getpid()}") # Unique logger name per process
    logger.info(f"TTS worker process started. PID: {os.getpid()}. Voice: {voice_name}")
    logger.info(f"Target audio format: {target_samplerate}Hz, {target_sample_width*8}-bit, {target_channels}-channel PCM")

    # --- Main Processing Loop ---
    while True:
        processed_audio_bytes = None # Reset for each loop iteration
        word_boundaries_data = None # Reset for each loop iteration
        try:
            # --- 1. Get text from answer_q ---
            answer_text = answer_q.get() # Blocking get

            # --- Check for Stop Signal ---
            if answer_text is None:
                logger.warning("Received stop signal (None) from answer_q.")
                tts_q.put(None) # Propagate stop signal
                logger.warning("Propagated stop signal (None) to tts_q. Exiting worker.")
                break

            # --- Validate Input ---
            if not isinstance(answer_text, str) or not answer_text.strip():
                logger.warning(f"Received invalid or empty text from answer_q: {type(answer_text)}. Skipping.")
                continue

            logger.info(f"Received text chunk: '{answer_text[:100]}{'...' if len(answer_text) > 100 else ''}'")

            # --- 2. Synthesize full audio and word boundaries using edge-tts ---
            # Use asyncio.run to execute the async helper function
            try:
                raw_audio_bytes, word_boundaries_data = asyncio.run(
                    _synthesize_and_get_raw_audio(answer_text, voice_name, logger)
                )
            except RuntimeError as e:
                 logger.error(f"RuntimeError calling async TTS function: {e}. Potential event loop issue.", exc_info=True)
                 continue # Skip this item
            except Exception as e:
                 logger.error(f"Unexpected error calling async TTS function: {e}", exc_info=True)
                 continue # Skip this item


            if raw_audio_bytes is None:
                logger.warning(f"Skipping text chunk due to TTS synthesis failure: '{answer_text[:30]}...'")
                continue # Move to the next item in the queue

            # --- 3. Decode and Convert Audio using pydub ---
            try:
                logger.debug("Decoding and converting raw audio bytes...")
                # Load raw bytes (likely MP3) into pydub AudioSegment
                audio_stream = io.BytesIO(raw_audio_bytes)
                # Let pydub detect format (ensure ffmpeg is available!)
                segment = AudioSegment.from_file(audio_stream)

                # Convert to target format
                segment = segment.set_frame_rate(target_samplerate)
                segment = segment.set_channels(target_channels)
                segment = segment.set_sample_width(target_sample_width)

                # Get the raw PCM bytes
                processed_audio_bytes = segment.raw_data
                logger.info(f"Audio successfully converted to PCM. Size: {len(processed_audio_bytes)} bytes.")

                # Optional: Log captured word boundaries (we don't put them in tts_q per requirement)
                if word_boundaries_data:
                     logger.debug(f"Captured {len(word_boundaries_data)} word boundaries (not queued).")


            except CouldntDecodeError as e:
                logger.error(f"pydub/ffmpeg failed to decode audio. Is ffmpeg installed and in PATH? Error: {e}", exc_info=True)
                continue # Skip this chunk if decoding fails
            except Exception as e:
                logger.error(f"Error during audio conversion: {e}", exc_info=True)
                continue # Skip this chunk if conversion fails

            # --- 4. Push processed audio to tts_q ---
            if processed_audio_bytes:
                logger.debug(f"Putting processed PCM audio chunk (size: {len(processed_audio_bytes)}) into tts_q.")
                tts_q.put(processed_audio_bytes)
            else:
                logger.warning("Audio processing resulted in empty data, not putting into queue.")


        except EOFError:
            logger.warning("answer_q connection closed unexpectedly (EOFError). Stopping worker.")
            tts_q.put(None) # Signal downstream
            break
        except BrokenPipeError:
            logger.warning("tts_q connection broke (BrokenPipeError). Cannot send further data. Stopping worker.")
            break # Can't signal downstream if pipe is broken
        except Exception as e:
            logger.error(f"Critical error in TTS worker main loop: {e}", exc_info=True)
            # Consider recovery or exiting gracefully
            time.sleep(1) # Avoid tight loop on persistent error
            # Maybe try to signal downstream before breaking on critical error
            # try:
            #     tts_q.put(None)
            # except Exception:
            #     logger.error("Failed to put None sentinel into tts_q after critical error.")
            # break # Decide if critical errors should stop the worker

    logger.info("TTS Worker process finished.")