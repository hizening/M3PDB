# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import pytorch_lightning as pl

# import os
# import copy

# from config import ex
# from model.face_tts import FaceTTS
# from data import _datamodules

# import numpy as np
# from scipy.io.wavfile import write

# from text import text_to_sequence, cmudict
# from text.symbols import symbols
# from utils.tts_util import intersperse
# import cv2

# from tqdm import tqdm


# @ex.automain
# def main(_config):

#     _config = copy.deepcopy(_config)
#     pl.seed_everything(_config["seed"])

#     print("######## Initializing TTS model")
#     model = FaceTTS(_config).cuda()

#     if _config['use_custom']:      
#         print(f"######## Load {_config['test_faceimg']}")
#         # use custom face image to synthesize the speech
#         spk = cv2.imread(os.path.join(f"{_config['test_faceimg']}"))
#         spk = cv2.resize(spk, (224, 224))
#         spk = np.transpose(spk, (2, 0, 1))
#         spk = torch.FloatTensor(spk).unsqueeze(0).to(model.device)
#     else:
#         # use LRS3 image 
#         print(f"######## Load {_config['dataset']}")
#         dm = _datamodules[f"dataset_{_config['dataset']}"](_config)
#         dm.set_test_dataset()
#         sample = dm.test_dataset[0] # you can adjust the index of test sample
#         spk = sample['spk'].to(model.device)

#     print(f"######## Load checkpoint from {_config['resume_from']}")
#     _config['enc_dropout'] = 0.0
#     model.load_state_dict(torch.load(_config['resume_from'])['state_dict'])
        
#     model.eval()
#     model.zero_grad()

#     print("######## Initializing HiFi-GAN")
#     vocoder = torch.hub.load('bshall/hifigan:main', 'hifigan').eval().cuda()

#     print(f"######## Load text description from {_config['test_txt']}")
#     with open(_config['test_txt'], 'r', encoding='utf-8') as f:
#         texts = [line.strip() for line in f.readlines()]

#     cmu = cmudict.CMUDict(_config['cmudict_path'])

#     with torch.no_grad():
        
#         for i, text in tqdm(enumerate(texts)):
#             x = torch.LongTensor(
#                 intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))
#             ).to(model.device)[None]
            
#             x_len = torch.LongTensor([x.size(-1)]).to(model.device)
#             y_enc, y_dec, attn = model.forward(
#                 x,
#                 x_len,
#                 n_timesteps=_config["timesteps"],
#                 temperature=1.5,
#                 stoc=False,
#                 spk=spk,
#                 length_scale=0.91,
#             )

#             audio = (
#                 vocoder.forward(y_dec[-1]).cpu().squeeze().clamp(-1, 1).numpy()
#                 * 32768
#             ).astype(np.int16)
            
#             if not os.path.exists(_config["output_dir"]):
#                 os.makedirs(_config["output_dir"])

#             write(
#                 f"{_config['output_dir']}/sample_{i}.wav",
#                 _config["sample_rate"],
#                 audio,
#             )

#     print(f"######## Done inference. Check '{_config['output_dir']}' folder")




import os 
import copy
import glob

import torch
import numpy as np
from scipy.io.wavfile import write
import cv2
from tqdm import tqdm

import pytorch_lightning as pl
from sacred import Experiment

from config import ex           # 上面那份 config.py
from data import _datamodules
from model.face_tts import FaceTTS
from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils.tts_util import intersperse

@ex.automain
def main(_config):
    # 深拷贝 config 并设定随机种子
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    # 1. 初始化模型
    print("######## Initializing TTS model")
    model = FaceTTS(_config).cuda()
    _config['enc_dropout'] = 0.0
    print(f"######## Load checkpoint from {_config['resume_from']}")
    ckpt = torch.load(_config['resume_from'], map_location="cpu")
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    model.zero_grad()

    # 2. 初始化 HiFi-GAN vocoder
    print("######## Initializing HiFi-GAN")
    vocoder = torch.hub.load('bshall/hifigan:main', 'hifigan').eval().cuda()

    # 3. 读取所有待合成文本
    print(f"######## Load text description from {_config['test_txt']}")
    with open(_config['test_txt'], 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]

    # 4. 准备 CMU 字典
    cmu = cmudict.CMUDict(_config['cmudict_path'])

    # 5. 准备人脸图片列表
    if _config['use_custom']: 
        img_folder = _config['test_faceimg_dir_2']
        img_paths = sorted(glob.glob(os.path.join(img_folder, "**", "*.png"), recursive=True))  # 递归查找所有子文件夹中的 .png 文件
        if not img_paths:
            raise RuntimeError(f"No .png files found in {img_folder}")

    else:
        # LRS3 单样本分支
        dm = _datamodules[f"dataset_{_config['dataset']}"](_config)
        dm.set_test_dataset()
        sample = dm.test_dataset[0]
        img_paths = [None]  # 用 None 做占位

    # 6. 批量推理
    with torch.no_grad():
        for img_path in img_paths:
            if _config['use_custom']:
                print(f"######## Load face image {img_path}")
                img = cv2.imread(img_path)
                img = cv2.resize(img, (_config['image_size'], _config['image_size']))
                img = np.transpose(img, (2, 0, 1))
                spk = torch.FloatTensor(img).unsqueeze(0).to(model.device)
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                # 获取图像所在的子文件夹路径
                subfolder = os.path.dirname(img_path)
            else:
                spk = sample['spk'].to(model.device)
                img_name = "lrs3"
                subfolder = img_folder  # 默认保存到 img_folder 中

            for i, text in enumerate(tqdm(texts, desc=f"Texts for {img_name}")):
                # 文本转序列
                seq = text_to_sequence(text, dictionary=cmu)
                x = torch.LongTensor(intersperse(seq, len(symbols))).to(model.device)[None]
                x_len = torch.LongTensor([x.size(-1)]).to(model.device)

                # 模型前向
                y_enc, y_dec, attn = model.forward(
                    x, x_len,
                    n_timesteps=_config['timesteps'],
                    temperature=1.5,
                    stoc=False,
                    spk=spk,
                    length_scale=0.91,
                )

                # Vocoder 合成
                audio = (
                    vocoder.forward(y_dec[-1])
                           .cpu()
                           .squeeze()
                           .clamp(-1, 1)
                           .numpy()
                           * 32768
                ).astype(np.int16)

                # 在对应子文件夹下保存 .wav 文件
                os.makedirs(subfolder, exist_ok=True)  # 确保子文件夹存在
                out_fn = f"{img_name}_sample_{i}.wav"
                out_path = os.path.join(subfolder, out_fn)
                write(out_path, _config['sample_rate'], audio)

    print(f"######## Done inference. Check '{img_folder}' folder")
