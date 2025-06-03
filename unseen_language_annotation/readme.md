### 1. lang_prob_confirm Folder

**Overview:**  
This folder contains core code verifying the significant positive correlation between speech language probability and audio translation/synthesis quality. It includes:

- s2st code based on the seamless_expressivity translation model,
- tts code based on the xTTS speech synthesis model,
- DNSMOS calculation and filtering files based on the DNS-challenge,
- language probability detection, CER, and ASR code files based on the large-v3-turbo model from the Whisper project,
- speaker similarity and emotion similarity code files based on kkkk and llll datasets respectively.

Below is a detailed introduction to each code file:

#### 1. s2st Folder

In our experiments, we utilize the seamless_expressivity translation model from the open-source project **[seamless_communication](https://github.com/facebookresearch/seamless_communication)** (https://huggingface.co/facebook/seamless-expressive) for streaming speech translation. We select 2,000 audio samples per language with DNSMOS scores close to μ=3.1981 and σ²=0.0313 to measure and calculate multiple metrics. This folder is designed for s2st from various languages into Italian and subsequent data analysis.

##### (1) predict.py

This file modifies the original `predict.py` found in `seamless_communication-main/src/seamless_communication/cli/expressivity/predict/` to enable batch inference functionality.

##### (2) s2st.sh

A batch inference shell script that takes a CSV file containing a column named "Original Audio Path" as input and outputs the translated audio and text.

##### (3) mk_csv_fr.py

Randomly reads a specified number of MP3 files from the Emilia_FR dataset, storing their paths in a summary CSV file to support batch s2st inference.

##### (4) mk_csv_zh.py

Constructs a new summary CSV file from a DNSMOS-filtered audio dataset for batch s2st inference. The DNSMOS filtering criteria are based on the mean and variance of DNSMOS scores from randomly selected French audio samples in `mk_csv_fr.py`.

##### (5) harmonic.py

Calculates weighted values of four metrics—CER, speaker similarity, emotion similarity, and speech rate difference—for each translated audio listed in the summary CSV. These metrics are computed in other code files.

##### (6) analyze.py

Computes the mean and variance of multiple metrics for audio data in different languages, writes the results to a text file, and plots cumulative distribution function (CDF) graphs.

#### 2. tts Folder

For the TTS validation experiments, the same reference audios used in the translation model are chosen, and speech synthesis is performed using the xTTS model from the project at https://github.com/coqui-ai/TTS.

##### (1) mk_csvs.py

Generates a new tts summary CSV file containing the original audio paths for all languages used in the translation tasks, their corresponding DNSMOS scores, and randomly selected Italian transcription texts to prepare for subsequent tts tasks.

##### (2) tts.py

Uses the xtts_v2 model to generate speech files from reference audios and texts, and appends the generated audio paths to the CSV file.

##### (3) analyze.py

Calculates the mean and variance of various metrics in the tts summary CSV files across languages and plots cumulative distribution function (CDF) graphs.

##### (4) analyze2.py

Performs stratified sampling on the tts summary CSV files for each language, then computes the mean and variance of the metrics.

#### 3. DNSMOS Folder

This folder contains files used to filter audio datasets by selecting audio samples whose DNSMOS scores match the mean and variance of the benchmark DNSMOS audio dataset. These filtered datasets are used in subsequent s2st and tts comparison experiments.

We employ the open-source project https://github.com/microsoft/DNS-Challenge to calculate DNSMOS scores. Using randomly selected French audio files as the benchmark (from `mk_csv_fr.py`), an equal number of audio samples from other languages are filtered and recorded in CSV files. To run this in the terminal, use the command:

```shell
python dnsmos_local.py -t C:\temp\SampleClips -o sample.csv
```

and the scores can be computed.

##### (1) fr_mp3_wav.py

This project only supports WAV audio files as input, so this script converts randomly selected MP3-format French audio files to WAV format.

##### (2) zh_mp3_wav.py

Reads MP3 audio files of the corresponding language from the Emilia dataset and converts them to WAV format to facilitate DNSMOS score filtering.

##### (3) simulated_annealing.py

Uses a simulated annealing algorithm to filter generated DNSMOS score files across languages, selecting audio datasets whose means and variances match those of the randomly selected French audio dataset, and writes the results to new CSV files.

#### 4. whisper Folder

This folder contains files for calculating relevant metrics of translated or synthesized audio, including language probability, CER, and speech rate difference. The code is developed based on the large-v3-turbo model from the [Whisper project](https://github.com/openai/whisper).

##### (1) prob&asr.py

Calculates the probability that translated or synthesized audio is in Italian based on s2st or tts summary files for each language, obtains ASR transcription texts, and writes these results back into the summary files.

##### (2) cer.py

Calculates CER from transcription texts compared to original texts and writes results into the summary files.

##### (3) speed_diff.py

Calculates speech rate differences between generated audio and original reference audio, recording the results in the summary files.

#### 5. cal_spk_sim.py

Based on the open-source project [Resemblyzer](https://github.com/resemble-ai/Resemblyzer), this script computes speaker similarity between reference and synthesized audio. It reads s2st or tts summary files across languages, calculates speaker similarity, and writes the results back.

#### 6. cal_emo_sim.py

Based on the open-source project [emotion2vec](https://github.com/ddlBoJack/emotion2vec), this script calculates emotional similarity between reference and synthesized audio, updating the corresponding s2st or tts summary files with results.

---

### 2. selection Folder

This folder contains code to verify that a two-stage filtering strategy allows cross-lingual prompts to achieve or even surpass the synthesis quality of same-language prompts. The metrics for evaluating synthesis quality remain CER, speaker similarity, emotional similarity, and speech rate difference. The general workflow is:

- Use French as a prompt to synthesize Italian audio, measure multiple metrics, and perform two-stage filtering.
- Calculate DNSMOS scores of the filtered French prompts as a baseline, then filter Italian audio with similar DNSMOS mean and variance.
- Use the filtered Italian audio as prompts for TTS, measure the same metrics, and perform comparative analysis.

#### 1. tts Folder

Contains three subfolders for tasks: French-to-Italian prompt synthesis (fr2it), Italian-to-Italian prompt synthesis (it2it), and analysis of cross-lingual vs. same-language prompt synthesis. The experiments also use the xTTS model from https://github.com/coqui-ai/TTS.

##### (1) fr2it Folder

- **singal_lang_tts.py**  
  Multi-GPU and multi-process TTS code based on the Emilia dataset for French-to-Italian speech synthesis, writing synthesized audio info to text files.

- **singal_log_to_csv.py**  
  Converts generated TXT log files into a fr2it summary CSV file.

##### (2) it2it Folder

- **it_to_it_tts.py**  
  Randomly selects Italian prompts from the Italian dataset to generate Italian speech and computes the average language probability using Whisper (explained in the whisper folder). This serves as the first-stage filtering threshold for French-to-Italian generation.

- **selected_it2it_tts.py**  
  Applies two-stage filtering on French-generated Italian speech to obtain a set of French audios with optimal expressiveness for cross-lingual synthesis under the current TTS model. It also filters an Italian dataset matching the DNSMOS mean and variance of this French set (see DNSMOS folder). This file synthesizes speech from the filtered Italian dataset and writes results to text files.

- **it_log_to_csv.py**  
  Similar to singal_log_to_csv.py, it compiles the results from selected_it2it_tts.py into an it2it summary CSV file.

##### (3) analyze Folder

- **text_process.py**  
  Prepares Italian text sets before speech synthesis by extracting qualifying Italian texts.

- **stage2_analyze.py**  
  Performs second-stage filtering on the French-generated Italian audio set (after first-stage filtering and evaluation of four metrics) and conducts partial statistical analysis.

- **it2it_fr2it_com.py**  
  Compares statistical characteristics of four metrics between the two-stage filtered French-generated Italian audio and native Italian-generated Italian audio, with visualization.

- **prepare_for_dnsmos.py**  
  Since the DNS-challenge project for DNSMOS scoring only processes WAV audio, this script converts filtered French and all Italian audio files to WAV format in preparation for DNSMOS calculation.

#### 2. whisper Folder

Contains three subfolders responsible for ASR transcription, CER calculation, and language probability computation for French-to-Italian and Italian-to-Italian synthesis, all based on the large-v3-turbo model from https://github.com/openai/whisper.

##### (1) asr Folder

- **asr_selected_it.py**  
  Computes transcription text for filtered Italian-to-Italian synthesized speech (used for comparison), generates intermediate logs.

- **asr_test.py**  
  Computes transcription text for original French-to-Italian synthesized speech, generates intermediate logs.

- **log_asr_process.py**  
  Integrates original French-to-Italian ASR transcription texts into corresponding summary CSV files.

- **log_selected_asr_process.py**  
  Integrates filtered Italian-to-Italian ASR transcription texts into corresponding summary CSV files.

##### (2) cer Folder

- **CER_cal.py**  
  Calculates CER for French-to-Italian synthesized speech and generates intermediate logs.

- **CER_selected_cal.py**  
  Calculates CER for filtered Italian-to-Italian synthesized speech and generates intermediate logs.

- **log_cer_process.py**  
  Integrates French-to-Italian CER results into summary CSV files.

- **log_selected_cer_process.py**  
  Integrates filtered Italian-to-Italian CER results into summary CSV files.

##### (3) lang_accuracy Folder

- **cal_accuracy_it_to_it.py**  
  Calculates language probability of randomly selected Italian-to-Italian generated speech (from it_to_it_tts.py) as the first-stage filtering baseline and generates logs.

- **cal_accuracy_selected_it2it.py**  
  Calculates language probability of filtered Italian-to-Italian generated speech (from selected_it2it_tts.py) and generates logs.

- **cal_accuracy_singal.py**  
  Calculates language probability of French-to-Italian synthesized speech (from singal_lang_tts.py) and generates logs.

- **log_it_to_it_process.py**  
  Post-processes logs from cal_accuracy_it_to_it.py to derive filtering baselines and plots score distributions.

- **log_selected_it2it_process.py**  
  Post-processes logs from cal_accuracy_selected_it2it.py, integrates language probability scores into it2it summary CSV files, and plots distributions.

- **log_singal_process.py**  
  Post-processes logs from cal_accuracy_singal.py, integrates language probability scores into fr2it summary CSV files, performs first-stage filtering based on cal_accuracy_it_to_it.py baseline, and plots score distributions.

#### 3. DNSMOS Folder

Contains a single file **it_all_selection.py** which applies simulated annealing to select an Italian audio set with approximately matching mean and variance. Before using this code, DNSMOS score CSV files for filtered French audio and all Italian audio need to be obtained using the commands described previously.

```shell
python dnsmos_local.py -t C:\temp\SampleClips -o sample.csv
```

#### 4. emotion2vec Folder

This folder contains two subfolders used to calculate emotional similarity and speech rate difference between synthesized speech and reference speech. Emotional similarity is computed using the open-source project [emotion2vec](https://github.com/ddlBoJack/emotion2vec).

##### (1) cal_emo_sim Folder

- **cal_emo_similarity.py**  
  Calculates the emotional similarity between reference speech (French prompt) and synthesized speech (Italian), writing results to intermediate files.

- **cal_selected_emo_similarity.py**  
  Calculates emotional similarity between reference speech (filtered Italian) and synthesized speech (Italian), writing results to intermediate files.

- **log_process.py**  
  Processes the log files generated by `cal_emo_similarity.py` and writes the data into the fr2it summary CSV file.

- **log_selected_emo_process.py**  
  Processes the log files generated by `cal_selected_emo_similarity.py` and writes the data into the it2it summary CSV file.

##### (2) cal_spd_diff Folder

- **cal_speed.py**  
  Calculates speech rate difference between reference speech (French prompt) and synthesized speech (Italian), writing results to intermediate files.

- **cal_selected_speed.py**  
  Calculates speech rate difference between reference speech (filtered Italian) and synthesized speech (Italian).

- **log_selected_speed_process.py**  
  Processes logs generated by `cal_selected_speed.py` and writes results into the it2it summary CSV file.

- **log_speed_process.py**  
  Processes logs generated by `cal_speed.py` and writes results into the fr2it summary CSV file.

#### 5. resemblyzer Folder

This folder contains four files used to calculate speaker similarity between synthesized and reference speech, developed using the open-source project [Resemblyzer](https://github.com/resemble-ai/Resemblyzer).

- **cal_selected_similarity.py**  
  Calculates speaker similarity between filtered Italian prompts and their generated Italian speech, writing results to intermediate files.

- **cal_similarity_singal_lang.py**  
  Calculates speaker similarity for French-to-Italian synthesized speech, writing results to intermediate files.

- **log_process.py**  
  Integrates the output logs from `cal_similarity_singal_lang.py` into the fr2it summary CSV file.

- **log_selected_process.py**  
  Integrates the output logs from `cal_selected_similarity.py` into the it2it summary CSV file.
