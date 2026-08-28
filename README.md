# Nepali Speech-to-Text

A Nepali Automatic Speech Recognition (ASR) project that evaluates and compares **Faster-Whisper** and **IndicWav2Vec** for Nepali speech transcription.

## Models

* **Faster-Whisper** — `large-v3`
* **IndicWav2Vec** — `sumanpaudel1997/nepali-asr-indicwav2vec`

## Evaluation

The models are evaluated using:

* **WER** — Word Error Rate
* **CER** — Character Error Rate
* **Processing Time**
* **RTF** — Real-Time Factor
* **Generated Transcription**

## Workflow

```text
Nepali Audio
     ↓
Preprocessing
     ↓
Faster-Whisper / IndicWav2Vec
     ↓
Generated Transcription
     ↓
WER / CER / Speed Evaluation
     ↓
Model Comparison
```

## Installation

```bash
pip install faster-whisper transformers torch librosa soundfile numpy jiwer
```

## Goal

The goal of this project is to evaluate the **accuracy and inference performance** of different speech recognition approaches for Nepali audio.

## Status

🚧 **Work in Progress**
