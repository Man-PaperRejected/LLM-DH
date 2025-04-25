import math
from multiprocessing import Process, Queue as MpQueue # Use multiprocessing Queue and Process
import threading
import time
import asyncio
import io
import numpy as np
import os 
import logging
import logging.handlers
import wave
import torch
from . import face_detection
from .models import Wav2Lip
from . import audio
from tqdm import tqdm
import cv2
from .config import VideoConfig as cfg
import subprocess
import copy
import queue
# 提取音频并保存为wav文件


class FaceDetector:
    def __init__(self, args, device='cuda'):
        """
        :param args: argparse.Namespace, face_det_batch_size, pads, nosmooth
        :param device: 'cuda' or 'cpu'
        """
        self.args = args
        self.device = device
        
    @staticmethod
    def get_smoothened_boxes(boxes, T):
        """
        滑动窗口平滑box序列
        :param boxes: np.array, shape=(N, 4)
        :param T: int, 窗口大小
        :return: np.array, 经平滑的boxes, shape=(N, 4)
        """
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i : i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes

    def detect(self, images):
        """
        :param images: List[np.ndarray]
        :return: List, 每个元素为 [face_crop_img(人脸区), (y1, y2, x1, x2)]
        """
        detector = face_detection.FaceAlignment(
            face_detection.LandmarksType._2D, 
            flip_input=False, 
            device=self.device
        )
        batch_size = self.args.face_det_batch_size

        while True:
            predictions = []
            try:
                for i in tqdm(range(0, len(images), batch_size)):
                    batch_imgs = np.array(images[i:i+batch_size])
                    predictions.extend(detector.get_detections_for_batch(batch_imgs))
            except RuntimeError:
                if batch_size == 1:
                    raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
                batch_size //= 2
                print('Recovering from OOM error; New batch size: {}'.format(batch_size))
                continue
            break

        pady1, pady2, padx1, padx2 = self.args.pads
        results = []
        for rect, image in zip(predictions, images):
            if rect is None:
                cv2.imwrite('temp/faulty_frame.jpg', image)
                raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')
            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)
            results.append([x1, y1, x2, y2])

        boxes = np.array(results)
        if not self.args.nosmooth:
            boxes = self.get_smoothened_boxes(boxes, T=5)
        
        # 得到最终人脸裁剪和坐标
        output = [
            [image[y1: y2, x1:x2], (y1, y2, x1, x2)]
            for image, (x1, y1, x2, y2) in zip(images, boxes)
        ]

        del detector
        return output

class VideoProcessor:
	def __init__(self,args):
		self.args = args
		self.face_batch=[]
		self.frame_batch=[]
		self.coords_batch=[]

	def forward(self):
		face_batch=[]
		frame_batch=[]
		coords_batch=[]
		frames = self.read_video()
		face_det_results = self.face_det(frames)
		for i,f in enumerate(frames):
			face, coords = face_det_results[i]
			face = cv2.resize(face, (self.args.img_size, self.args.img_size))
			frame_batch.append(f)
			face_batch.append(face)
			coords_batch.append(coords)

		face_batch = np.asarray(face_batch)
		img_masked = face_batch.copy()
		img_masked[:, self.args.img_size//2:] = 0
		face_batch = np.concatenate((img_masked, face_batch), axis=3) / 255.
		self.face_batch=np.asarray(face_batch)
		self.frame_batch=np.asarray(frame_batch)
		self.coords_batch=np.asarray(coords_batch)
        
	def read_video(self):
		if not os.path.isfile(self.args.face):
			raise ValueError('--face argument must be a valid path to video/image file')

		elif self.args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
			full_frames = [cv2.imread(self.args.face)]
			self.fps = self.args.fps

		else:
			video_stream = cv2.VideoCapture(self.args.face)
			self.fps = video_stream.get(cv2.CAP_PROP_FPS)

			print('Reading video frames...')

			full_frames = []
			while 1:
				still_reading, frame = video_stream.read()
				if not still_reading:
					video_stream.release()
					break
				if self.args.resize_factor > 1:
					frame = cv2.resize(frame, (frame.shape[1]//self.args.resize_factor, frame.shape[0]//self.args.resize_factor))

				if self.args.rotate:
					frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

				y1, y2, x1, x2 = self.args.crop
				if x2 == -1: x2 = frame.shape[1]
				if y2 == -1: y2 = frame.shape[0]

				frame = frame[y1:y2, x1:x2]

				full_frames.append(frame)
		return full_frames

	def face_det(self,frames):
		face_detector = FaceDetector(self.args)
		if self.args.box[0] == -1:
			if not self.args.static:
				face_det_results = face_detector.detect(frames) # BGR2RGB for CNN face detection
			else:
				face_det_results = face_detector.detect([frames[0]])
		else:
			print('Using the specified bounding box instead of face detection...')
			y1, y2, x1, x2 = self.args.box
			face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]
		return face_det_results

class Audioprocessor:
	def __init__(self, args):
		self.args = args
		self.mel_step_size=16
		self.fpm = 80./25  # 每段音频对应几帧的图像
		

	def read_audio(self):
		if not self.args.audio.endswith('.wav'):
			print('Extracting raw audio...')
			command = 'ffmpeg -y -i {} -strict -2 {}'.format(self.args.audio, 'temp/temp.wav')

			subprocess.call(command, shell=True)
			self.args.audio = 'temp/temp.wav'

		wav = audio.load_wav(self.args.audio, 16000)
		return wav
  
	def forward(self,wav):
		mel = audio.melspectrogram(wav)
		i = 0
		mel_chunks=[]
		while 1:
			start_idx = int(i * self.fpm)
			if start_idx + self.mel_step_size > len(mel[0]):
				mel_chunks.append(mel[:, len(mel[0]) - self.mel_step_size:])
				break
			mel_chunks.append(mel[:, start_idx : start_idx + self.mel_step_size])
			i += 1
		return np.asarray(mel_chunks)

def _load(checkpoint_path,device):
	if device == 'cuda':
		checkpoint = torch.load(checkpoint_path)
	else:
		checkpoint = torch.load(checkpoint_path,
								map_location=lambda storage, loc: storage)
	return checkpoint

def load_model(path):
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	print('Using {} for inference.'.format(device))
	model = Wav2Lip()
	print("Load checkpoint from: {}".format(path))
	checkpoint = _load(path,device)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	model = model.to(device)
	return model.eval()

def lip_sync(args, model, mel_batch, face_batch, frame_batch, coord_batch):
	gendata = []
	bs = args.wav2lip_batch_size
	face_len = len(face_batch)
	mel_len = len(mel_batch)
	for i in range(0, mel_len, bs):
		end_idx = min(mel_len, i+bs)
		mels = mel_batch[i:end_idx]
		mels = np.reshape(mels, [len(mels), mels.shape[1], mels.shape[2], 1])

		seg_len = end_idx - i

        # 判断是否能直接切片
		if i + seg_len <= face_len:
			faces = face_batch[i:i+seg_len]
			frames = frame_batch[i:i+seg_len]
			coords = coord_batch[i:i+seg_len]
		else:
			idxs = [(i+j)%face_len for j in range(seg_len)]
			faces  = [face_batch[k] for k in idxs]
			frames = [frame_batch[k] for k in idxs]
			coords = [coord_batch[k] for k in idxs]

		faces = torch.FloatTensor(np.transpose(faces, (0, 3, 1, 2))).to(next(model.parameters()).device)
		mels = torch.FloatTensor(np.transpose(mels, (0, 3, 1, 2))).to(next(model.parameters()).device)

		with torch.no_grad():
			pred = model(mels, faces)

		pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
		for p, f, c in zip(pred, frames, coords):
			y1, y2, x1, x2 = c
			p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

			f[y1:y2, x1:x2] = p
			gendata.append(f)
	return gendata


def lip_sync_thread_worker(result_q, model, mel_chunks, face_batch, frame_batch, coord_batch):
    """Function to run lip_sync in a separate thread."""
    global logger # Access the logger configured in the main function
    try:
        logger.info("lip_sync thread started.")

        gendata = lip_sync(cfg, model, mel_chunks, face_batch, frame_batch, coord_batch)
        logger.info(f"lip_sync thread produced {len(gendata) if gendata else 0} frames.")
        result_q.put(gendata) # Put the result into the queue
    except Exception as e:
        logger.error(f"Error in lip_sync thread: {e}", exc_info=True)
        result_q.put(None) # Signal error or completion without valid data

def wav2lip(tts_q: MpQueue,
             log_q: MpQueue,
             video_pipe_w_fd: int,
             audio_pipe_w_fd: int,
             channels: int = 1,
             sampwidth: int = 2,
             framerate: int = 16000
            ):

    # --- Logging Setup --- (Keep as is)
    queue_handler = logging.handlers.QueueHandler(log_q)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []
    root_logger.addHandler(queue_handler)
    global logger # Make logger global if thread worker needs it
    logger = logging.getLogger(f"wav2lip_{os.getpid()}")
    logger.info("wav2lip processor Start")
    # --- Validate FDs --- (Keep as is)
    try:
        os.fstat(video_pipe_w_fd)
        os.fstat(audio_pipe_w_fd) # Keep audio check
        logger.info(f"FDs are valid: video_fd={video_pipe_w_fd}, audio_fd={audio_pipe_w_fd}")
    except OSError as e:
        logger.error(f"FD invalid in wav2lip: {e}", exc_info=True)
        return
    # --- Video/Model Initialization --- (Keep as is)
    try:
        video_processor = VideoProcessor(cfg)
        video_processor.forward()
        fps = video_processor.fps # Keep fps for audio calculation
        ori_frame_batch = copy.deepcopy(video_processor.frame_batch)
        original_len = len(ori_frame_batch)
        audio_processor = Audioprocessor(cfg)
        model = load_model(cfg.checkpoint_path)
        logger.info(f"Original frames loaded: {original_len}, FPS: {fps}")
    except Exception as e:
        logger.error(f"Failed to load/prepare initial video frames: {e}", exc_info=True)
        return
    # --- Calculate Audio Bytes --- (Keep as is, still needs fps)
    try:
        bytes_per_sample = channels * sampwidth
        samples_per_frame_float = (framerate * channels) / fps
        samples_per_frame_int = math.ceil(samples_per_frame_float)
        bytes_per_video_frame = samples_per_frame_int * bytes_per_sample
        if bytes_per_video_frame <= 0: raise ValueError("Bytes per frame <= 0")
        SILENT_AUDIO_CHUNK = bytes([0] * bytes_per_video_frame)
        logger.info(f"Calculated {bytes_per_video_frame} audio bytes per frame.")
    except Exception as e:
        logger.error(f"Error calculating audio bytes: {e}", exc_info=True)
        return

    logger.info("wav2lip initialize Done")

    # --- State Variables ---
    original_frame_index = 0
    gendata = None
    gendata_index = 0
    get_timeout = 0.01 # Keep for tts_q? Or use block=False?
    running = True
    result_q = queue.Queue(maxsize=10)
    # frame_interval = 1.0 / fps # No longer needed for rate control
    # last_frame_time = time.monotonic() # No longer needed for rate control

    try:
        while running:
            # now = time.monotonic() # No longer needed for rate control
            # time_since_last_frame = now - last_frame_time # No longer needed for rate control

            # --- Try to start NEW lip_sync task ---
            # (Your existing logic for starting threads based on tts_q)
            try:
                # Using block=False might be better without rate control to avoid waiting
                audio_bytes = tts_q.get(block=True, timeout=0.05)
                logger.info("Received WAV data from tts_q.")
                if audio_bytes is None:
                    logger.warning("Received stop signal (None) from tts_q.")
                    running = False
                    continue

                # Process audio and start thread (Keep your logic)
                audio_np_array = np.frombuffer(audio_bytes, dtype=np.int16)
                mel_chunks = audio_processor.forward(audio_np_array)
                frame_batch_copy, face_batch_copy, coord_batch_copy = copy.deepcopy(video_processor.frame_batch), copy.deepcopy(video_processor.face_batch), copy.deepcopy(video_processor.coords_batch)

                logger.info("Starting lip_sync thread...")
                lip_sync_thread = threading.Thread(
                        target=lip_sync_thread_worker,
                        args=(result_q, model, mel_chunks, face_batch_copy, frame_batch_copy, coord_batch_copy),
                        daemon=True
                    )
                lip_sync_thread.start()

            except queue.Empty:
                pass # No new audio data
            except Exception as e:
                logger.error(f"Error getting data or starting thread: {e}", exc_info=True)
                running = False
                continue

            # --- Check for NEW COMPLETED results ONLY if not currently processing a batch ---
            if gendata is None:
                try:
                    new_data = result_q.get_nowait()
                    if new_data is not None:
                        logger.info(f"Fetched new batch of {len(new_data)} generated frames.")
                        gendata = new_data
                        gendata_index = 0
                    else:
                        logger.warning("Received None from result queue (thread error?).")
                except queue.Empty:
                    pass # No results waiting

            # --- Determine frame to send ---
            frame_to_send = None
            source = "unknown"

            # Priority 1: Use generated frame if available and index is valid
            if gendata is not None:
                if gendata_index < len(gendata):
                    frame_to_send = gendata[gendata_index]
                    gendata_index += 1
                    source = f"generated[{gendata_index}/{len(gendata)}]"
                else:
                    logger.info("Finished sending generated batch.")
                    gendata = None # Clear it

            # Priority 2: Use original frame if no generated frame was selected above
            if frame_to_send is None:
                # logger.info("push original frame")
                frame_to_send = ori_frame_batch[original_frame_index % original_len]
                original_frame_index += 1
                source = f"original[{original_frame_index % original_len}]"

            # --- Send frame (Removed the time check) ---
            # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
            # The 'if time_since_last_frame >= frame_interval:' block is removed.
            # Sending now happens in every loop iteration if a frame is ready.
            if frame_to_send is not None:
                try:
                    # Convert to bytes
                    if not isinstance(frame_to_send, np.ndarray): raise TypeError(f"Frame is not NumPy array ({source})")
                    if frame_to_send.dtype != np.uint8: frame_to_send = frame_to_send.astype(np.uint8)
                    video_frame_bytes = frame_to_send.tobytes()

                    # Write video
                    # logger.debug(f"Writing frame: {source}") # Add debug log if needed
                    written_video = os.write(video_pipe_w_fd, video_frame_bytes)
                    if written_video != len(video_frame_bytes): raise BrokenPipeError("Short video write")

                    # Write silent audio
                    # written_audio = os.write(audio_pipe_w_fd, SILENT_AUDIO_CHUNK)
                    # if written_audio != len(SILENT_AUDIO_CHUNK): raise BrokenPipeError("Short audio write")

                    # last_frame_time = now # No longer needed

                except BrokenPipeError as e:
                    logger.warning(f"Pipe closed by FFmpeg? ({e}). Stopping.")
                    running = False
                except Exception as e:
                    logger.error(f"Error writing frame from {source} to pipes: {e}", exc_info=True)
                    running = False
            else:
                 # This case should ideally not be reached with the current logic
                 logger.info("No frame selected to send this cycle.")

            # Removed the entire rate control sleep logic here
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    except Exception as e:
        logger.error(f"Critical error in main loop: {e}", exc_info=True)
    finally:
        # --- Cleanup --- (Keep your existing finally block)
        logger.info("Closing write pipes.")
        for fd, name in [(video_pipe_w_fd, "video"), (audio_pipe_w_fd, "audio")]:
            if fd is not None:
                try:
                    os.close(fd)
                    logger.debug(f"Closed {name} write pipe FD: {fd}")
                except OSError as e:
                    logger.error(f"Error closing {name} write pipe FD {fd}: {e}")

    logger.info(f"Wav2lip Looping Test Pipe process ({os.getpid()}) finished.")
        # try:
        #     with wave.open(output_filename, 'wb') as wf:
        #         wf.setnchannels(channels)      
        #         wf.setsampwidth(sampwidth)      
        #         wf.setframerate(framerate)     
        #         wf.writeframes(audio_bytes)

        #     logger.info(f"Successfully saved audio to {output_filename}")

        # except wave.Error as e:
        #     logger.error(f"Error writing WAV file using wave module: {e}", exc_info=True)
        # except IOError as e:
        #     logger.error(f"Error opening or writing file {output_filename}: {e}", exc_info=True)
        # except Exception as e:
        #     logger.error(f"An unexpected error occurred during WAV file saving: {e}", exc_info=True)


'''

1 个 mel_chunk (从 mel_chunks 列表中取出)
用于生成 1 帧 视频图像
对应于 (target_samplerate * target_channels * target_sample_width) / fps 字节的原始 PCM 音频流。
'''