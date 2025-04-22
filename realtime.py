from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
from models import Wav2Lip
import platform
import torch.nn.functional as F
import queue
import copy
gendata = queue.Queue()
def get_args():
	parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

	parser.add_argument('--checkpoint_path', type=str, 
						help='Name of saved checkpoint to load weights from', default="checkpoints/wav2lip/wav2lip.pth")

	parser.add_argument('--face', type=str, 
						help='Filepath of video/image that contains faces to use', default="girl.mp4")
	parser.add_argument('--audio', type=str, 
						help='Filepath of video/audio file to use as raw audio source', default="audio1.wav")
	parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.', 
									default='results/result_voice.mp4')

	parser.add_argument('--static', type=bool, 
						help='If True, then use only first video frame for inference', default=False)
	parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', 
						default=25., required=False)

	parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
						help='Padding (top, bottom, left, right). Please adjust to include chin at least')

	parser.add_argument('--face_det_batch_size', type=int, 
						help='Batch size for face detection', default=16)
	parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)

	parser.add_argument('--resize_factor', default=1, type=int, 
				help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

	parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], 
						help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. ' 
						'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

	parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], 
						help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
						'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

	parser.add_argument('--rotate', default=False, action='store_true',
						help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
						'Use if you get a flipped result, despite feeding a normal looking video')

	parser.add_argument('--nosmooth', default=False, action='store_true',
						help='Prevent smoothing face detections over a short temporal window')

	args = parser.parse_args()
	args.img_size = 96
	if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
		args.static = True
	return args



class FaceDetector:
    def __init__(self, args, device='cuda'):
        """
        :param args: argparse.Namespace，包括 face_det_batch_size, pads, nosmooth
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
					frame = cv2.resize(frame, (frame.shape[1]//self.args.resize_factor, frame.shape[0]//args.resize_factor))

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
		self.fpm = 80./30  # 每段音频对应几帧的图像
		self.mel_chunks = []
		

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
		self.mel_chunks = np.asarray(mel_chunks)

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


def video_player(frames, fps=25):
    """
    循环播放图像帧列表，直到按下 'q' 键。

    Args:
        frames (list or numpy.ndarray): 包含图像帧的列表或数组。
                                       每个帧应该是 OpenCV 可以显示的格式 (例如 NumPy 数组 BGR)。
        fps (int, optional): 期望的播放帧率。默认为 25。
    """
    if not isinstance(frames, (list, np.ndarray)) or len(frames) == 0:
        print("错误：输入的帧数据无效或为空。")
        return

    num_frames = len(frames)
    if num_frames == 0:
        print("错误：帧列表为空。")
        return

    delay = max(1, int(1000 / fps))

    frame_idx = 0
    window_name = "循环视频播放器 (按 'q' 退出)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # 创建可调整大小的窗口

    print(f"开始播放... 按 'q' 键停止。")

    while True:
        current_frame = frames[frame_idx % num_frames]

        cv2.imshow(window_name, current_frame)

        key = cv2.waitKey(delay) & 0xFF

        if key == ord('q'):
            print("检测到 'q' 键，停止播放。")
            break

        frame_idx += 1

        try:
             if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("窗口已关闭，停止播放。")
                break
        except cv2.error:
             print("窗口已被销毁，停止播放。")
             break

    cv2.destroyAllWindows()
    print("播放器已关闭。")


def lip_sync(args, model, mel_batch, face_batch, frame_batch, coord_batch):
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
			gendata.put(f)


def main(args):
	video_processor = VideoProcessor(args)
	audio_processor = Audioprocessor(args)
	model = load_model(args.checkpoint_path)
	video_processor.forward()
	wav = audio_processor.read_audio()
	audio_processor.forward(wav)
	mel_batch = copy.deepcopy(audio_processor.mel_chunks)
	frame_batch,face_batch,coord_batch = copy.deepcopy(video_processor.frame_batch), copy.deepcopy(video_processor.face_batch), copy.deepcopy(video_processor.coords_batch)
	lip_sync(args, model,mel_batch,face_batch,frame_batch,coord_batch)
	video_player(video_processor.frame_batch, video_processor.fps)


if __name__ == '__main__':
	args = get_args()
	main(args)
