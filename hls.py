# ffmpeg_process.py
import os
import subprocess
import logging
import logging.handlers
import threading
import config # Import config to access FFmpeg settings
from multiprocessing import Queue as MpQueue
# --- Logging Configuration Helper ---
# This helper configures logging within the subprocess to use the shared queue.

# --- FFmpeg Process Starter ---
def ffmpeg_worker(
    video_in_fd: int,
    audio_in_fd: int,
    output_url: str, # e.g., RTMP URL or file path
    log_q: MpQueue
):
    """
    Starts and manages the FFmpeg process, reading from pipes and streaming/saving output.
    """
    queue_handler = logging.handlers.QueueHandler(log_q)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers = []  # 移除默认handlers，防止重复
    root_logger.addHandler(queue_handler)
    # Configure logging for this specific FFmpeg process using the shared queue
    ffmpeg_logger = logging.getLogger("FFmpeg") # Get a specific logger instance
    pid = os.getpid()
    ffmpeg_logger.info(f"Process ({pid}) started. Configuring FFmpeg...")
    ffmpeg_logger.info(f"wav2lip received pipe FDs: Video read={video_in_fd}, Audio read={audio_in_fd}")
    try:
        os.fstat(video_in_fd)
        os.fstat(audio_in_fd)
        ffmpeg_logger.info(f"FDs are valid: video_fd={video_in_fd}, audio_fd={audio_in_fd}")
    except OSError as e:
        ffmpeg_logger.error(f"FD invalid in ffmpeg_worker: {e}", exc_info=True)
        return  # 或抛出异常，停止进程
    # --- Get Parameters from Config (or use defaults) ---
    # Video Input
    width = config.DHVG_VIDEO_WIDTH if hasattr(config, 'DHVG_VIDEO_WIDTH') else 640
    height = config.DHVG_VIDEO_HEIGHT if hasattr(config, 'DHVG_VIDEO_HEIGHT') else 480
    framerate = config.DHVG_VIDEO_FRAMERATE if hasattr(config, 'DHVG_VIDEO_FRAMERATE') else 25
    pix_fmt_in = 'bgr24' # Common format from OpenCV/numpy

    # Audio Input
    audio_format_in = config.DHVG_AUDIO_FORMAT if hasattr(config, 'DHVG_AUDIO_FORMAT') else 's16le'
    sample_rate_in = config.DHVG_AUDIO_SAMPLE_RATE if hasattr(config, 'DHVG_AUDIO_SAMPLE_RATE') else 16000
    channels_in = config.DHVG_AUDIO_CHANNELS if hasattr(config, 'DHVG_AUDIO_CHANNELS') else 1

    # Output Settings (Examples - Adjust in config.py ideally)
    vcodec_out = config.FFMPEG_VCODEC if hasattr(config, 'FFMPEG_VCODEC') else 'libx264'
    pix_fmt_out = config.FFMPEG_PIXFMT if hasattr(config, 'FFMPEG_PIXFMT') else 'yuv420p'
    preset = config.FFMPEG_PRESET if hasattr(config, 'FFMPEG_PRESET') else 'veryfast'
    gop_size = config.FFMPEG_GOP if hasattr(config, 'FFMPEG_GOP') else framerate * 2
    video_bitrate = config.FFMPEG_VBITRATE if hasattr(config, 'FFMPEG_VBITRATE') else '2500k'
    acodec_out = config.FFMPEG_ACODEC if hasattr(config, 'FFMPEG_ACODEC') else 'aac'
    audio_bitrate = config.FFMPEG_ABITRATE if hasattr(config, 'FFMPEG_ABITRATE') else '128k'
    sample_rate_out = config.FFMPEG_ARATE if hasattr(config, 'FFMPEG_ARATE') else 44100
    channels_out = config.FFMPEG_ACHANNELS if hasattr(config, 'FFMPEG_ACHANNELS') else 2
    buffer_size = config.FFMPEG_BUFSIZE if hasattr(config, 'FFMPEG_BUFSIZE') else '5000k'
    output_format = config.FFMPEG_FORMAT if hasattr(config, 'FFMPEG_FORMAT') else 'flv' # e.g., 'flv' for RTMP

    # Ensure FDs are used correctly in the command arguments
    video_pipe_str = f"pipe:{video_in_fd}"
    audio_pipe_str = f"pipe:{audio_in_fd}"

    # --- Construct the FFmpeg Command ---
    command = [
        'ffmpeg',
        '-loglevel', 'warning', # Reduce log verbosity, errors will still appear on stderr
        # Global options (can go before inputs)
        # '-re', # Read input at native frame rate (important for streaming)

        # Input Video Options
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', pix_fmt_in,
        '-s', f'{width}x{height}',
        '-r', str(framerate),
        '-i', video_pipe_str,

        # # Input Audio Options
        # '-f', audio_format_in,
        # '-ar', str(sample_rate_in),
        # '-ac', str(channels_in),
        # '-i', audio_pipe_str,
        
        # Audio Encoding (DISABLED)
        '-an', # <--- CRITICAL: Disable audio output stream

        # Output Options
        '-c:v', vcodec_out,
        '-pix_fmt', pix_fmt_out,
        '-preset', preset,
        '-g', str(gop_size),
        '-b:v', video_bitrate,
        '-maxrate', video_bitrate, # Optional: constrain max rate
        '-c:a', acodec_out,
        '-b:a', audio_bitrate,
        '-ar', str(sample_rate_out), # Output sample rate
        '-ac', str(channels_out),   # Output channels
        '-bufsize', buffer_size,
        '-f', output_format,
        '-hls_time', '2',           # <--- 添加：目标分片时长
        '-hls_list_size', '10',      # <--- 添加：播放列表长度
        '-hls_flags', 'delete_segments', # <--- 添加：自动删除旧分片
        # Output destination
        output_url
    ]

    ffmpeg_logger.info(f"Executing FFmpeg command: {' '.join(command)}")
    process = None # Initialize process variable

    try:
        # Start FFmpeg subprocess
        # os.set_inheritable(video_in_fd, True)
        # os.set_inheritable(audio_in_fd, True)
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL, # Don't inherit stdin
            stdout=subprocess.PIPE,   # Capture stdout
            stderr=subprocess.PIPE,   # Capture stderr
            pass_fds=[video_in_fd, audio_in_fd] # IMPORTANT: Pass read FDs to child
        )
        ffmpeg_logger.info(f"FFmpeg process ({pid}) started with PID: {process.pid}. Monitoring output...")

        # --- Monitor FFmpeg's output streams ---
        def log_stream(stream, level):
            """Reads and logs lines from a stream."""
            try:
                # Use iter(readline, b'') for robust line reading
                for line in iter(stream.readline, b''):
                    if not line: # Check if the stream ended
                        break
                    try:
                        # Decode safely, ignoring errors, and strip whitespace
                        ffmpeg_logger.log(level, line.decode('utf-8', errors='ignore').strip())
                    except Exception as log_err:
                         # Log decoding/logging errors themselves if they occur
                         print(f"FFMPEG LOGGING ERROR: {log_err}") # Use print as logger might be problematic
            except Exception as read_err:
                 # Log errors during stream reading
                 print(f"FFMPEG STREAM READ ERROR: {read_err}")
            finally:
                # Ensure stream is closed when done or if error occurs
                try:
                    stream.close()
                except Exception:
                    pass # Ignore errors on close

        # Start threads to read stdout and stderr concurrently
        stdout_thread = threading.Thread(target=log_stream, args=(process.stdout, logging.INFO), daemon=True)
        stderr_thread = threading.Thread(target=log_stream, args=(process.stderr, logging.ERROR), daemon=True)
        stdout_thread.start()
        stderr_thread.start()

        # Wait for FFmpeg process to exit
        # process.wait() can block indefinitely if streams aren't read,
        # but the threads handle that. We still need to wait for completion.
        ret_code = process.wait() # Wait for process termination

        # Wait for logging threads to finish processing any remaining output
        stdout_thread.join(timeout=2.0)
        stderr_thread.join(timeout=2.0)

        if ret_code == 0:
            ffmpeg_logger.info(f"FFmpeg process ({pid}) exited cleanly (code 0).")
        else:
            ffmpeg_logger.error(f"FFmpeg process ({pid}) exited with error code {ret_code}.")

    except FileNotFoundError:
        ffmpeg_logger.error(f"FFmpeg command not found for process {pid}. Make sure FFmpeg is installed and in system PATH.")
    except Exception as e:
        ffmpeg_logger.error(f"Exception in FFmpeg process ({pid}): {e}", exc_info=True)
        # If Popen failed, process might be None
        if process and process.poll() is None: # Check if process started but failed later
             process.terminate() # Try to terminate if it's still running after exception
             process.wait(timeout=2.0)
    finally:
        ffmpeg_logger.info(f"FFmpeg process ({pid}) function finishing. Ensuring resources are released.")
        # --- Resource Cleanup ---
        # Close the pipe FDs passed to FFmpeg (important!)
        try:
            if video_in_fd is not None:
                os.close(video_in_fd)
                ffmpeg_logger.debug(f"({pid}) Closed video input FD {video_in_fd}")
        except OSError as e:
            ffmpeg_logger.error(f"({pid}) Error closing video input FD {video_in_fd}: {e}")
        try:
             if audio_in_fd is not None:
                os.close(audio_in_fd)
                ffmpeg_logger.debug(f"({pid}) Closed audio input FD {audio_in_fd}")
        except OSError as e:
            ffmpeg_logger.error(f"({pid}) Error closing audio input FD {audio_in_fd}: {e}")

        # Ensure process streams are closed (should be handled by log_stream, but double-check)
        if process:
            for stream in [process.stdout, process.stderr]:
                 if stream and not stream.closed:
                     try:
                         stream.close()
                     except Exception: pass
        ffmpeg_logger.info(f"FFmpeg process ({pid}) function finished execution.")

