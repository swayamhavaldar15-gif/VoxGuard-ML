import sounddevice as sd
import scipy.io.wavfile as wav
import torch
import time
import os

from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy


# -----------------------------
# SETTINGS
# -----------------------------

SAMPLE_RATE = 16000
DURATION = 10

AUDIO_FILE = "speaker_verification/models/enrollment_audio.wav"

EMBEDDING_FILE = "speaker_verification/models/registered_embedding.pt"


# -----------------------------
# LOAD MODEL
# -----------------------------

print("Loading Speaker Recognition Model...")

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="models/spkrec-ecapa-voxceleb",
    local_strategy=LocalStrategy.COPY
)


# -----------------------------
# COUNTDOWN
# -----------------------------

print("\nGet ready to enroll your voice!")

for i in range(3, 0, -1):

    print(i)
    time.sleep(1)


print("\n🎤 SPEAK NOW!")


# -----------------------------
# RECORD AUDIO
# -----------------------------

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32"
)

sd.wait()


print("Recording completed!")


# -----------------------------
# SAVE AUDIO
# -----------------------------

os.makedirs(
    "speaker_verification/models",
    exist_ok=True
)


wav.write(
    AUDIO_FILE,
    SAMPLE_RATE,
    audio
)


# -----------------------------
# CREATE SPEAKER EMBEDDING
# -----------------------------

print("Generating Speaker Embedding...")


signal = classifier.load_audio(
    AUDIO_FILE
)


embedding = classifier.encode_batch(
    signal
)


embedding = embedding.squeeze()


# -----------------------------
# SAVE EMBEDDING
# -----------------------------

torch.save(
    embedding,
    EMBEDDING_FILE
)


print("\n✅ ENROLLMENT SUCCESSFUL!")

print(
    f"Speaker embedding saved at:\n{EMBEDDING_FILE}"
)