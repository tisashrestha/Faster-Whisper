import os
import glob
import numpy as np
import soundfile as sf
import librosa
import noisereduce as nr
import torch
import torchaudio

from silero_vad import load_silero_vad, get_speech_timestamps

INPUT_DIR = "recordings"
OUTPUT_DIR = "processed"

TARGET_SR = 16000

MAX_CLIP_SECONDS = 30

MIN_CLIP_SECONDS = 1.0

PADDING_MS = 150

SPLIT_SEARCH_SECONDS = 3

TARGET_PEAK = 0.95

NOISE_REDUCTION_STRENGTH = 0.7


os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading Silero VAD...")

vad_model = load_silero_vad()

print("Silero VAD loaded.\n")


def load_audio(path):

    print(f"Loading: {os.path.basename(path)}")

    audio, sr = librosa.load(
        path,
        sr=TARGET_SR,
        mono=True
    )

    return audio.astype(np.float32), TARGET_SR


def normalize_volume(audio):

    peak = np.max(np.abs(audio))

    if peak == 0:
        return audio

    audio = audio / peak

    audio = audio * TARGET_PEAK

    return audio



def reduce_noise(audio, sr):

    print("Reducing background noise...")

    noise_clip_length = min(
        int(sr * 1.0),
        len(audio)
    )

    noise_clip = audio[:noise_clip_length]

    try:

        cleaned = nr.reduce_noise(
            y=audio,
            sr=sr,
            y_noise=noise_clip,
            prop_decrease=NOISE_REDUCTION_STRENGTH,
            stationary=False
        )

    except Exception as e:

        print("Noise reduction warning:", e)

        cleaned = audio

    return cleaned.astype(np.float32)



def detect_speech(audio, sr):

    print("Detecting speech...")

    audio_tensor = torch.from_numpy(audio)

    timestamps = get_speech_timestamps(
        audio_tensor,
        vad_model,
        sampling_rate=sr,
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=350,
        speech_pad_ms=PADDING_MS
    )

    return timestamps


def merge_speech_regions(timestamps, sr):

    if not timestamps:
        return []

    regions = []

    current_start = timestamps[0]["start"]
    current_end = timestamps[0]["end"]

    for ts in timestamps[1:]:

        start = ts["start"]
        end = ts["end"]

        gap = (start - current_end) / sr

        if gap <= 1.2:

            current_end = end

        else:

            regions.append(
                (
                    current_start,
                    current_end
                )
            )

            current_start = start
            current_end = end

    regions.append(
        (
            current_start,
            current_end
        )
    )

    return regions


def find_split_point(audio, start, max_end, sr):
    search_start = max(
        start + int(5 * sr),
        max_end - int(SPLIT_SEARCH_SECONDS * sr)
    )

    search_end = min(
        len(audio),
        max_end + int(1 * sr)
    )

    if search_end <= search_start:
        return max_end

    section = audio[search_start:search_end]

    rms = librosa.feature.rms(
        y=section,
        frame_length=512,
        hop_length=128
    )[0]

    if len(rms) == 0:
        return max_end

    quiet_index = np.argmin(rms)

    split = search_start + quiet_index * 128

    max_allowed = start + int(MAX_CLIP_SECONDS * sr)

    split = min(split, max_allowed)

    return split


def create_clips(audio, regions, sr, base_name):

    clips = []

    for start, end in regions:

        current = start

        while current < end:

            remaining = end - current

            max_samples = int(
                MAX_CLIP_SECONDS * sr
            )

            if remaining <= max_samples:

                clips.append(
                    audio[current:end]
                )

                break

            max_end = current + max_samples

            split = find_split_point(
                audio,
                current,
                max_end,
                sr
            )

            if split <= current:

                split = max_end

            clips.append(
                audio[current:split]
            )

            current = split

    saved = 0

    for i, clip in enumerate(clips, start=1):

        duration = len(clip) / sr

        if duration < MIN_CLIP_SECONDS:

            continue

        clip = normalize_volume(clip)

        filename = (
            f"{base_name}_{i:03d}.wav"
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            filename
        )

        sf.write(
            output_path,
            clip,
            sr,
            subtype="PCM_16"
        )

        saved += 1

        print(
            f"  Saved: {filename} "
            f"({duration:.2f} sec)"
        )

    return saved


def process_file(path):

    base_name = os.path.splitext(
        os.path.basename(path)
    )[0]

    print("\n" + "=" * 60)

    print(
        f"PROCESSING: {base_name}"
    )

    print("=" * 60)


    audio, sr = load_audio(path)

    original_duration = len(audio) / sr

    print(
        f"Original duration: "
        f"{original_duration:.2f} sec"
    )


    timestamps = detect_speech(
        audio,
        sr
    )

    if not timestamps:

        print(
            "No speech detected. Skipping."
        )

        return 0



    regions = merge_speech_regions(
        timestamps,
        sr
    )

    print(
        f"Speech regions detected: "
        f"{len(regions)}"
    )


    speech_parts = []

    for start, end in regions:

        speech_parts.append(
            audio[start:end]
        )

    speech_audio = np.concatenate(
        speech_parts
    )

    speech_duration = (
        len(speech_audio) / sr
    )

    print(
        f"Speech duration kept: "
        f"{speech_duration:.2f} sec"
    )

    cleaned_audio = reduce_noise(
        speech_audio,
        sr
    )


    cleaned_audio = normalize_volume(
        cleaned_audio
    )


    cleaned_tensor = torch.from_numpy(
        cleaned_audio
    )

    final_timestamps = get_speech_timestamps(
        cleaned_tensor,
        vad_model,
        sampling_rate=sr,
        threshold=0.5,
        min_speech_duration_ms=300,
        min_silence_duration_ms=500,
        speech_pad_ms=PADDING_MS
    )

    final_regions = merge_speech_regions(
        final_timestamps,
        sr
    )


    saved = create_clips(
        cleaned_audio,
        final_regions,
        sr,
        base_name
    )

    print(
        f"Finished: {saved} clips"
    )

    return saved



def main():

    files = []

    extensions = [
        "*.wav",
        "*.mp3",
        "*.m4a",
        "*.flac",
        "*.ogg"
    ]

    for ext in extensions:

        files.extend(
            glob.glob(
                os.path.join(
                    INPUT_DIR,
                    ext
                )
            )
        )

    files.sort()

    if not files:

        print(
            f"No audio files found in "
            f"'{INPUT_DIR}'"
        )

        return

    print(
        f"Found {len(files)} recordings."
    )

    total_clips = 0

    for path in files:

        try:

            total_clips += process_file(
                path
            )

        except Exception as e:

            print(
                f"\nERROR processing "
                f"{os.path.basename(path)}:"
            )

            print(e)

    print("\n" + "=" * 60)

    print("ALL RECORDINGS PROCESSED")

    print(
        f"Total clips created: "
        f"{total_clips}"
    )

    print(
        f"Output folder: "
        f"{OUTPUT_DIR}"
    )

    print("=" * 60)


if __name__ == "__main__":

    main()