# Nepali Speech-to-Text with Faster-Whisper

A Nepali Speech-to-Text project using **Faster-Whisper** to transcribe Nepali audio recordings into text.

## Overview

This project evaluates Faster-Whisper for Nepali speech recognition and measures the quality of the generated transcriptions using standard speech-to-text evaluation metrics.

The project focuses on:

* Nepali audio transcription
* Faster-Whisper model evaluation
* Audio preprocessing and resampling
* Word Error Rate (WER)
* Character Error Rate (CER)
* Transcription speed
* Generated transcription output

## Model

The project currently uses:

**Model:** `large-v3`
**Framework:** Faster-Whisper
**Language:** Nepali (`ne`)

Faster-Whisper is an optimized implementation of Whisper using CTranslate2, allowing faster and more memory-efficient inference.

## Project Workflow

```text
Nepali Audio
     ↓
Audio Preprocessing
     ↓
Resampling to 16 kHz
     ↓
Faster-Whisper
     ↓
Nepali Transcription
     ↓
Compare with Reference Text
     ↓
WER / CER / Speed Evaluation
```

## Installation

Install the required Python packages:

```bash
pip install faster-whisper librosa soundfile numpy jiwer
```

## Usage

Set the audio file:

```python
AUDIO_FILE = "your_audio.wav"
```

Set the model:

```python
MODEL_SIZE = "large-v3"
```

Load the model:

```python
from faster_whisper import WhisperModel

model = WhisperModel(
    MODEL_SIZE,
    device="cpu",
    compute_type="int8"
)
```

Then transcribe the audio:

```python
segments, info = model.transcribe(
    AUDIO_FILE,
    language="ne"
)

transcription = " ".join(
    segment.text for segment in segments
)
```

## Evaluation Metrics

The project uses the following metrics:

### Word Error Rate (WER)

Measures errors at the word level.

```text
WER = (Substitutions + Deletions + Insertions) / Number of Reference Words
```

Lower WER indicates better transcription accuracy.

### Character Error Rate (CER)

Measures errors at the character level.

Lower CER indicates better character-level transcription accuracy.

### Transcription Speed

The processing time is measured to evaluate how quickly the model can transcribe the audio.

## Output

The evaluation produces:

* Generated Nepali transcription
* WER
* CER
* Processing time
* Real-Time Factor (RTF), when calculated

Example:

```text
Audio File: example.wav

Generated Text:
नमस्ते तपाईंलाई कस्तो छ

WER: 0.XX
CER: 0.XX
Processing Time: XX.XX seconds
```

## Hardware

The current experiments can be run on CPU, although GPU inference is recommended for faster processing.

Example CPU configuration:

```python
device="cpu"
compute_type="int8"
```

## Project Structure

```text
nepali-speech-to-text/
│
├── audio/
│   └── *.wav
│
├── transcriptions/
│   └── *.txt
│
├── scripts/
│   └── transcription.py
│
├── README.md
└── requirements.txt
```

## Future Improvements

* Evaluate different Faster-Whisper model sizes
* Compare CPU and GPU performance
* Improve Nepali audio preprocessing
* Evaluate VAD settings
* Test on a larger Nepali speech dataset
* Fine-tune Whisper for Nepali speech recognition
* Compare Faster-Whisper results with other speech-to-text models

## Status

**Work in Progress**

This repository is being used for experimentation and evaluation of Nepali speech-to-text using Faster-Whisper.

