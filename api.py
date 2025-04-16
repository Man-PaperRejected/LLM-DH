#!/usr/bin/env python3

import os
import wave
import json
import tempfile
from flask import Flask, request, jsonify
from vosk import Model, KaldiRecognizer, SetLogLevel
from pydub import AudioSegment

# --- 配置 ---
# 模型的路径 (指向解压后包含 am/, conf/, graph/, ivector/ 等文件夹的目录)
MODEL_PATH = "./vosk-model-small-en-us-0.15" # <--- 修改为你解压后的模型文件夹路径
# Vosk 期望的采样率 (大多数模型是 16000 Hz)
TARGET_SAMPLE_RATE = 16000
# 允许上传的文件扩展名 (可选，增加安全性)
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'm4a', 'flac'}

# --- 初始化 ---
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024 # 可选：限制上传文件大小为 32MB

# 检查模型路径是否存在
if not os.path.exists(MODEL_PATH):
    print(f"错误：找不到模型路径 '{MODEL_PATH}'。请下载模型并将其放在正确的位置，或更新 MODEL_PATH 变量。")
    print("模型可以从 https://alphacephei.com/vosk/models 下载")
    exit(1)

print(f"正在从 '{MODEL_PATH}' 加载 Vosk 模型...")
# SetLogLevel(0) # 显示 Vosk 详细日志，-1 关闭日志
try:
    model = Model(MODEL_PATH)
    print("Vosk 模型加载成功。")
except Exception as e:
    print(f"加载 Vosk 模型失败: {e}")
    exit(1)

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- API 端点 ---
@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """接收音频文件并返回转录文本"""
    # 1. 检查是否有文件部分
    if 'audio_file' not in request.files:
        return jsonify({"error": "请求中未找到 'audio_file' 部分"}), 400

    file = request.files['audio_file']

    # 2. 检查文件名
    if file.filename == '':
        return jsonify({"error": "未选择文件"}), 400

    # 3. 检查文件类型 (可选但推荐)
    if not allowed_file(file.filename):
         return jsonify({"error": f"不允许的文件类型。支持的格式: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    try:
        # 4. 保存上传的文件到临时位置
        # 使用 tempfile 更安全，避免覆盖和清理问题
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            file.save(tmp_file.name)
            tmp_file_path = tmp_file.name
        print(f"临时文件保存在: {tmp_file_path}")

        # 5. 使用 Pydub 处理音频
        print("使用 pydub 处理音频...")
        audio = AudioSegment.from_file(tmp_file_path)

        # 转换为单声道 & 目标采样率
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(TARGET_SAMPLE_RATE)

        # 6. 初始化 Vosk 识别器
        # KaldiRecognizer 需要知道音频的采样率
        rec = KaldiRecognizer(model, audio.frame_rate)
        rec.SetWords(True) # 可选：如果你需要每个词的时间戳信息
        # rec.SetPartialWords(True) # 可选：获取部分识别结果（更实时，但此处处理整个文件）

        # 7. 将音频数据块喂给识别器
        # Pydub 的 raw_data 是 bytes 类型，正是 Vosk 需要的
        print(f"向 Vosk 提供 {len(audio.raw_data)} 字节的音频数据...")
        rec.AcceptWaveform(audio.raw_data)

        # 8. 获取最终识别结果
        result_json = rec.FinalResult()
        print(f"Vosk 原始结果: {result_json}")
        result_dict = json.loads(result_json)

        # 提取文本
        transcribed_text = result_dict.get('text', '') # 使用 .get() 以防 'text' 键不存在

        # 9. 清理临时文件
        os.remove(tmp_file_path)
        print(f"临时文件已删除: {tmp_file_path}")

        # 10. 返回 JSON 响应
        return jsonify({"text": transcribed_text})

    except FileNotFoundError:
         # Pydub 依赖的 ffmpeg 可能没找到
        print(f"错误：处理音频时出错。请确保 ffmpeg 已安装并位于系统 PATH 中。")
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
             os.remove(tmp_file_path) # 确保即使出错也尝试清理
        return jsonify({"error": "音频处理失败，请检查服务器日志 (可能是 ffmpeg 问题)"}), 500
    except Exception as e:
        print(f"处理请求时发生错误: {e}")
        import traceback
        traceback.print_exc() # 打印详细错误堆栈到服务器日志
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
             os.remove(tmp_file_path) # 确保即使出错也尝试清理
        return jsonify({"error": f"服务器内部错误: {str(e)}"}), 500

# --- 运行服务器 ---
if __name__ == '__main__':
    # host='0.0.0.0' 使服务器可以从局域网内其他机器访问
    # debug=True 在开发时很有用，但在生产环境中应设为 False
    app.run(host='0.0.0.0', port=5000, debug=True)
