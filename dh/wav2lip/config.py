from dataclasses import dataclass

@dataclass
class VideoConfig:
    # Path configurations
    checkpoint_path = "/data/Code/AIGC/LLM-DH/model/wav2lip/wav2lip.pth"
    face = "/data/Code/AIGC/LLM-DH/dh/material/girl.mp4"
    audio = "audio1.wav"
    outfile = "results/result_voice.mp4"

    # Video/image processing configurations
    static = False
    fps = 25.0
    pads = [0, 10, 0, 0]
    resize_factor = 1
    crop = [0, -1, 0, -1]
    box = [-1, -1, -1, -1]
    rotate = False
    img_size = 96

    # Batch size configurations
    face_det_batch_size = 16
    wav2lip_batch_size = 128

    # Other flags
    nosmooth = False