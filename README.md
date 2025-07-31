<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>M<sup>3</sup>PDB</title>
  <style>
    body {
      background-color: #FFFFFF; /* Sets the background color to white */
    }
  </style>
</head>
<body>
  <!-- Your content goes here -->
  <h3 align="center">M<sup>3</sup>PDB</h3>
  <!-- Rest of your content -->
</body>
</html>


<!-- PROJECT SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />

<p align="center">
  <a href="https://github.com/shaojintian/Best_README_template/">
    <img src="images/logo.png" alt="Logo" width="180" height="180">
  </a>

  <h3 align="center">M<sup>3</sup>PDB</h3>
  <p align="center">
    A Multi-Modal, Multi-Label, Multilingual Prompt Database
    <br />
    <a href="https://github.com/hizening/M3PDB"><strong>Explore the documentation of this project
 »</strong></a>
    <br />
    <br />
    <a href="https://jiangyu1205.github.io/subjective">View Demo (Demo and Subjective Test)</a>
    ·
    <a href="https://github.com/shaojintian/Best_README_template/issues">Report Bug</a>
    ·
    <a href="https://github.com/shaojintian/Best_README_template/issues">Make a Suggestion</a>
  </p>

</p>

 This README.md is intended for developers.

### What‘s new :fire:
- [2025.06] Update [code](https://github.com/hizening/M3PDB) , [demo](https://jiangyu1205.github.io/token2emo/) and [dataset](https://huggingface.co/datasets/M3PDB/M3PDB) for M<sup>3</sup>PDB.

### Table of Contents

- [Getting Started Guide](#getting-started-guide) 
  - [Development Configuration Requirements](#development-configuration-requirements)
  - [Installation Steps](#installation-steps)
- [File Directory Description](#file-directory-description)
- [Dataset Construction](#dataset-construction)
  - [Multimodal Data Preprocessing](#multimodal-data-preprocessing)
  - [Annotation System](#annotation-system)
  - [Unseen Language Annotation](#unseen-language-annotation)
- [Dataset Usage](#dataset-usage)
  - [Multi-model Prompt Registration](#multi-model-prompt-registration)
  - [Latency Aware Online Selection](#latency-aware-online-selection)
- [How to Contribute to the Open Source Project](#how-to-contribute-to-the-open-source-project)
- [Version Control](#version-control)
- [Contact](#contact)
- [License](#license)
- [Acknowledgements](#acknowledgements)


### Getting Started Guide

###### **Development Configuration Requirements**

Due to the significant differences in the configuration environments of the various models in this study, we chose to use separate environments for each model in practice. These models interact through API calls to achieve collaboration. The configuration method for each model's environment is documented separately in its respective folder.

###### **Installation Steps**

1. Get a free API Key at [https://chatgpt.com/](https://chatgpt.com/)
2. Clone the repo
```sh
git clone https://github.com/hizening/M3PDB.git
```
3. Different systems require different environments. Please refer to the `readme.md` of each subsystem for configuration.

### File Directory Description
```
filetree 
├── /annotation_system/
│  ├── /Qwen2-Audio/
│  ├── /SenseVoice/
│  ├── /emotion2vec/
│  ├── /llmware/
│  ├── /readme.md/
├── /latency_aware_online_system/
│  ├── /latency_aware_online_selection.py/
│  ├── /readme.md/
├── /multi-model_prompt_registration/
│  ├── /facetts/
│  ├── /f2s.py/
│  ├── /s2s.py/
│  ├── /t2s.py/
│  ├── /readme.md/
├── /multimodal_data_preprocessing/
│  ├── /3D-Speaker/
│  ├── /speech/
│  ├── /video/
│  ├── /readme.md/
├── /unseen_language_annotation/
│  ├── /lang_prob_confirm/
│  ├── /selection/
│  ├── /readme.md/
```

### Dataset Construction 

###### **Multimodal Data Preprocessing**
<p align="center">
  <a href="https://github.com/shaojintian/Best_README_template/"> 
    <img src="images\appendixA.2.png" alt="Logo" style="width: auto; height: auto;">
  </a>

1.Run the code below to achieve audio-video separation.
```sh
python multimodal_data_preprocessing/video/split_media.py
```
2.Run the code below to achieve speech format standardization.
```sh
python multimodal_data_preprocessing/speech/format_standardization.py
```
3.Run the code below to achieve video format standardization.
```sh
python multimodal_data_preprocessing/video/format_standardization.py
```
4.Run the code below to achieve speech enhancement.
```sh
python multimodal_data_preprocessing/speech/speech_enhancement.py
```
5.Run the code below to achieve video quality enhancement.
```sh
python multimodal_data_preprocessing/video/VideoSuperResolution/Train/eval.py
```
6.Run the code below to achieve multimodal speaker diarization and VAD.
```sh
cd multimodal_data_preprocessing/3D-Speaker/egs/3dspeaker/speaker-diarization/
bash run_audio.sh
bash run_video.sh
```
......
For more detailed information, please read the `/multimodal_data_preprocessing/readme.md`.

###### **Annotation System**
<p align="center">
  <a href="https://github.com/shaojintian/Best_README_template/"> 
    <img src="images\fig_RAG.png" alt="Logo" style="width: auto; height: auto;">
  </a>

For more detailed information, please read the `/annotation_system/readme.md`.

###### **Unseen Language Annotation**
<p align="center">
  <a href="https://github.com/shaojintian/Best_README_template/"> 
    <img src="images\unseenlanguage.png" alt="Logo" style="width: auto; height: auto;">
  </a>

1.Run the code below to generate speech.
```sh
python unseen_language_annotation/lang_prob_confirm/tts/tts.py
```
2.Run the code below to evaluate the quality of the synthesized speech.
```sh
python dnsmos_local.py -t C:\temp\SampleClips -o sample.csv
```
......
For more detailed information, please read the `/unseen_language_annotation/readme.md`.
### Dataset Usage 

###### **Multi-model Prompt Registration**
<p align="center">
  <a href="https://github.com/shaojintian/Best_README_template/"> 
    <img src="images\translate_prompt_selection—3.png" alt="Logo" style="width: auto; height: auto;">
  </a>

1.Run the code below to match and register speech similar to the registered speech.
```sh
python /multi-model_prompt_registration/s2s.py
```
2.Run the code below to generate phase-based reference speech based on the registered face. 
```sh
python /multi-model_prompt_registration/facetts/inference.py
```
3.Run the code below to match and register speech similar to the registered face.
```sh
python /multi-model_prompt_registration/f2s.py
```
4.Run the code below to match and register speech similar to the registered text.
```sh
python /multi-model_prompt_registration//t2s.py
```
......
For more detailed information, please read the `/multi-model_prompt_registration/readme.md`.
###### **Latency Aware Online Selection**
<p align="center">
  <a href="https://github.com/shaojintian/Best_README_template/"> 
    <img src="images\appendixG.png" alt="Logo" style="width: auto; height: auto;">
  </a>

1.Run the code below to dynamically find the most suitable speech.
```sh
python /latency_aware_online_selection/latency_aware_online_selection.py
```
......
For more detailed information, please read the `/latency_aware_online_selection/readme.md`.
### How to Contribute to the Open Source Project

Contributions make the open-source community an excellent place for learning, inspiration, and creation. Any contribution you make is **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Version Control

This project uses Git for version control. You can check the current available version in the repository.

### Contact
If you have any comment or question about M<sup>3</sup>PDB, please contact us by
- email: zhuboyu@mail.nwpu.edu.cn

### License
M<sup>3</sup>PDB is released under the [MIT](https://github.com/hizening/M3PDB/blob/main/LICENSE.txt).

### Acknowledgements
M<sup>3</sup>PDB contains third-party components and code modified from some open-source repos, including: <br>
1. datasets
[Emilia Dataset](https://github.com/open-mmlab/Amphion/tree/main/preprocessors/Emilia), 
2. code
[3D-Speaker](https://github.com/modelscope/3D-Speaker), 


<!-- ## Citations
If you find this repository useful, please consider giving a star :star: and citation :t-rex::
```BibTeX
@article{chen20243d,
  title={3D-Speaker-Toolkit: An Open Source Toolkit for Multi-modal Speaker Verification and Diarization},
  author={Chen, Yafeng and Zheng, Siqi and Wang, Hui and Cheng, Luyao and others},
  booktitle={ICASSP},
  year={2025}
}
``` -->

<!-- links -->
[your-project-path]:hizening/M3PDB
[contributors-shield]: https://img.shields.io/github/contributors/shaojintian/Best_README_template.svg?style=flat-square
[contributors-url]: https://github.com/hizening/M3PDB/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/shaojintian/Best_README_template.svg?style=flat-square
[forks-url]: https://github.com/hizening/M3PDB/network/members
[stars-shield]: https://img.shields.io/github/stars/shaojintian/Best_README_template.svg?style=flat-square
[stars-url]: https://github.com/hizening/M3PDB/stargazers
[issues-shield]: https://img.shields.io/github/issues/shaojintian/Best_README_template.svg?style=flat-square
[issues-url]: https://img.shields.io/github/issues/hizening/M3PDB.svg
[license-shield]: https://img.shields.io/github/license/shaojintian/Best_README_template.svg?style=flat-square
[license-url]: https://github.com/hizening/M3PDB/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/shaojintian



