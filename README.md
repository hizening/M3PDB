该储存库提供了该工作除了数据库之外的所有代码。
1、数据预处理，
音视频数据分离：
````markdown
# 使用示例

## 1. 安装依赖并确认 `ffmpeg` 可用
```bash
pip install moviepy
# 确保 ffmpeg 在 PATH 中
```

## 2. 保存脚本并赋可执行权限（Linux/macOS）
```bash
chmod +x batch_split_media.py
```

## 3. 运行脚本（仅处理顶层 `.mp4`/`.mov` 等视频）
```bash
python batch_split_media.py /path/to/videos
```

## 4. 如果想输出到单独目录、递归子文件夹，并指定格式
```bash
python batch_split_media.py /path/to/videos \
  -o /path/to/output \
  -r \
  --audio_ext wav \
  --video_ext mkv
```

这样，脚本会遍历指定文件夹中的视频，将它们拆分成音频文件（如 `xxx_audio.mp3`）和无声视频文件（如 `xxx_video.mp4`）。  
````





The Emilia-Pipe includes the following major steps:

Standardization：Audio normalization
Source Separation: Long audio -> Long audio without BGM
Speaker Diarization: Get medium-length single-speaker speech data
Fine-grained Segmentation by VAD: Get 3-30s single-speaker speech segments
ASR: Get transcriptions of the speech segments
Filtering: Obtain the final processed dataset
Setup Steps 👨‍💻
0. Prepare Environment
Install Python and CUDA.

Run the following commands to install the required packages:

conda create -y -n AudioPipeline python=3.9 
conda activate AudioPipeline

bash env.sh
Download the model files from the third-party repositories.

Manually download the checkpoints of UVR-MDX-NET-Inst_HQ_3 (UVR-MDX-NET-Inst_3.onnx) and DNSMOS P.835 (sig_bak_ovr.onnx), then save their path for the next step configuration (i.e. #2 and #3 TODO).
Creat the access token to pyannote/speaker-diarization-3.1 following the guide, then save it for the next step configuration (i.e. #4 TODO).
Make sure you have stable connection to GitHub and HuggingFace. The checkpoints of Silero and Whisperx-medium will be downloaded automatically on the pipeline's first run.
1. Modify Config File
Change the config.json file according to the following TODOs.

{
    "language": {
        "multilingual": true,
        "supported": [
            "zh",
            "en",
            "fr",
            "ja",
            "ko",
            "de"
        ]
    },
    "entrypoint": {
        // TODO: Fill in the input_folder_path. 
        "input_folder_path": "examples", // #1: Data input folder for processing
        "SAMPLE_RATE": 24000
    },
    "separate": {
        "step1": {
            // TODO: Fill in the source separation model's path. 
            "model_path": "/path/to/model/separate_model/UVR-MDX-NET-Inst_HQ_3.onnx", // #2: Model path
            "denoise": true,
            "margin": 44100,
            "chunks": 15,
            "n_fft": 6144,
            "dim_t": 8,
            "dim_f": 3072
        }
    },
    "mos_model": {
        // TODO: Fill in the DNSMOS prediction model's path. 
        "primary_model_path": "/path/to/model/mos_model/DNSMOS/sig_bak_ovr.onnx" // #3: Model path
    },
     // TODO: Fill in your huggingface access token for pynannote. 
    "huggingface_token": "<HUGGINGFACE_ACCESS_TOKEN>" // #4: Huggingface access token for pyannote
}
2. Run Script
Change the input_folder_path in config.json to the folder path where the downloaded audio files are stored (i.e. #1 TODO).
Run the following command to process the audio files:
conda activate AudioPipeline
export CUDA_VISIBLE_DEVICES=0  # Setting the GPU to run the pipeline, separate by comma

python main.py
Processed audio will be saved into input_folder_path_processed folder.
3. Check the Results
The processed audio (default 24k sample rate) files will be saved into input_folder_path_processed folder. The results for a single audio will be saved in a same folder with its original name and include the following information:

MP3 file: <original_name>_<idx>.mp3 where idx is corresponding to the index in the JSON-encoded array.
JSON file: <original_name>.json
[
    {
        "text": "So, don't worry about that. But, like for instance, like yesterday was very hard for me to say, you know what, I should go to bed.", // Transcription
        "start": 67.18, // Start timestamp, in second unit
        "end": 74.41, // End timestamp, in second unit
        "language": "en", // Language
        "dnsmos": 3.44 // DNSMOS P.835 score
    }
]
