import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import torch.nn.functional as F

from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_RATE = 16000
DURATION = 5

RAW_AUDIO_FILE = (
    "speaker_verification/models/live_test.wav"
)

PROCESSED_AUDIO_FILE = (
    "speaker_verification/models/live_processed.wav"
)

REGISTERED_EMBEDDING_FILE = (
    "speaker_verification/models/registered_embedding.pt"
)


# ============================================================
# SPEAKER THRESHOLDS
# ============================================================

SPEAKER_THRESHOLD = 0.50


# ============================================================
# START
# ============================================================

print("=" * 60)
print("VOXGUARD - LIVE SPEAKER VERIFICATION")
print("=" * 60)


# ============================================================
# LOAD ECAPA
# ============================================================

print("\nLoading Speaker Recognition Model...")

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="models/spkrec-ecapa-voxceleb",
    local_strategy=LocalStrategy.COPY
)

print("Speaker model loaded successfully.")


# ============================================================
# LOAD REGISTERED EMBEDDING
# ============================================================

print("\nLoading Registered Speaker Profile...")

registered_embedding = torch.load(
    REGISTERED_EMBEDDING_FILE,
    map_location="cpu"
)

registered_embedding = registered_embedding.squeeze()

registered_embedding = F.normalize(
    registered_embedding,
    p=2,
    dim=0
)

print(
    "Registered embedding shape:",
    registered_embedding.shape
)

print("Registered speaker profile loaded.")


# ============================================================
# RECORD AUDIO
# ============================================================

print("\n" + "=" * 60)
print("LIVE MICROPHONE RECORDING")
print("=" * 60)

print(f"\nSpeak normally for {DURATION} seconds.")
print("Starting in 2 seconds...")

import time
time.sleep(2)

print("\nRecording...")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32"
)

sd.wait()

print("Recording completed!")


# ============================================================
# CONVERT TO MONO
# ============================================================

audio = np.asarray(audio)

if audio.ndim > 1:
    audio = audio.mean(axis=1)

audio = audio.flatten()


# ============================================================
# AUDIO DIAGNOSTICS
# ============================================================

print("\n" + "=" * 60)
print("RAW AUDIO INFORMATION")
print("=" * 60)

print("Samples:", len(audio))
print("Sample rate:", SAMPLE_RATE)
print("Duration:", len(audio) / SAMPLE_RATE)

rms = np.sqrt(np.mean(audio ** 2))
peak = np.max(np.abs(audio))

print(f"RMS: {rms:.6f}")
print(f"Peak: {peak:.6f}")


# ============================================================
# REMOVE DC OFFSET
# ============================================================

audio = audio - np.mean(audio)


# ============================================================
# NORMALIZE AUDIO
# ============================================================

max_value = np.max(np.abs(audio))

if max_value > 0:

    audio = audio / max_value

print("\nAudio normalization completed.")


# ============================================================
# SAVE RAW/NORMALIZED AUDIO
# ============================================================

sf.write(
    RAW_AUDIO_FILE,
    audio,
    SAMPLE_RATE
)

print(
    "Live recording saved:",
    RAW_AUDIO_FILE
)


# ============================================================
# SIMPLE SPEECH ACTIVITY DETECTION
# ============================================================

print("\nDetecting speech region...")

absolute_audio = np.abs(audio)

threshold = max(
    0.02,
    np.percentile(absolute_audio, 20)
)

speech_indices = np.where(
    absolute_audio > threshold
)[0]


if len(speech_indices) > 0:

    start = speech_indices[0]
    end = speech_indices[-1]

    # Add small padding around speech
    padding = int(0.25 * SAMPLE_RATE)

    start = max(
        0,
        start - padding
    )

    end = min(
        len(audio),
        end + padding
    )

    speech_audio = audio[start:end]

else:

    speech_audio = audio


# ============================================================
# CHECK SPEECH LENGTH
# ============================================================

print(
    f"Detected speech duration: "
    f"{len(speech_audio) / SAMPLE_RATE:.2f} seconds"
)

if len(speech_audio) < int(1.0 * SAMPLE_RATE):

    print("\nWARNING: Very little speech detected.")

    speech_audio = audio


# ============================================================
# SAVE PROCESSED AUDIO
# ============================================================

sf.write(
    PROCESSED_AUDIO_FILE,
    speech_audio,
    SAMPLE_RATE
)

print(
    "Processed audio saved:",
    PROCESSED_AUDIO_FILE
)


# ============================================================
# LOAD PROCESSED AUDIO
# ============================================================

print("\n" + "=" * 60)
print("EXTRACTING LIVE SPEAKER EMBEDDING")
print("=" * 60)

signal = classifier.load_audio(
    PROCESSED_AUDIO_FILE
)

print(
    "Live audio tensor shape:",
    signal.shape
)

print(
    "Live audio dtype:",
    signal.dtype
)


# ============================================================
# ECAPA EMBEDDING
# ============================================================

with torch.no_grad():

    live_embedding = classifier.encode_batch(
        signal
    )

print(
    "Raw live embedding shape:",
    live_embedding.shape
)

live_embedding = live_embedding.squeeze()

print(
    "Squeezed embedding shape:",
    live_embedding.shape
)

live_embedding = F.normalize(
    live_embedding,
    p=2,
    dim=0
)

print(
    "Final live embedding shape:",
    live_embedding.shape
)


# ============================================================
# COSINE SIMILARITY
# ============================================================

similarity = F.cosine_similarity(
    registered_embedding.unsqueeze(0),
    live_embedding.unsqueeze(0)
).item()


# ============================================================
# SPEAKER DECISION
# ============================================================

print("\n" + "=" * 60)
print("SPEAKER VERIFICATION RESULT")
print("=" * 60)

print(
    f"\nSpeaker Similarity: {similarity:.4f}"
)

print(
    f"Speaker Threshold: {SPEAKER_THRESHOLD:.2f}"
)


if similarity >= SPEAKER_THRESHOLD:

    speaker_status = "VERIFIED"

else:

    speaker_status = "MISMATCH"


print(
    "Speaker Status:",
    speaker_status
)


# ============================================================
# SIMILARITY INTERPRETATION
# ============================================================

print("\n" + "-" * 60)
print("SIMILARITY INTERPRETATION")
print("-" * 60)

if similarity >= 0.70:

    print("VERY STRONG SPEAKER MATCH")

elif similarity >= 0.50:

    print("STRONG ENOUGH SPEAKER MATCH")

elif similarity >= 0.30:

    print("MODERATE / UNCERTAIN MATCH")

else:

    print("LOW SPEAKER SIMILARITY")


# ============================================================
# FINAL ACCESS DECISION
# ============================================================

print("\n" + "=" * 60)
print("VOXGUARD ACCESS DECISION")
print("=" * 60)

if speaker_status == "VERIFIED":

    print("ACCESS: ALLOWED")

else:

    print("ACCESS: BLOCKED")

print("=" * 60)