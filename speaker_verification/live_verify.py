import sounddevice as sd
import scipy.io.wavfile as wav
import torch
import torch.nn.functional as F
import time
import os

from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy


# --------------------------------
# SETTINGS
# --------------------------------

SAMPLE_RATE = 16000
DURATION = 5

LIVE_AUDIO_FILE = "speaker_verification/models/live_test.wav"

REGISTERED_EMBEDDING_FILE = (
    "speaker_verification/models/registered_embedding.pt"
)


# --------------------------------
# LOAD MODEL
# --------------------------------

print("Loading Speaker Recognition Model...")

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="models/spkrec-ecapa-voxceleb",
    local_strategy=LocalStrategy.COPY
)


# --------------------------------
# LOAD REGISTERED SPEAKER
# --------------------------------

print("Loading Registered Speaker Profile...")

registered_embedding = torch.load(
    REGISTERED_EMBEDDING_FILE,
    weights_only=True
)


# --------------------------------
# COUNTDOWN
# --------------------------------

print("\nLive verification will start in:")

for i in range(3, 0, -1):

    print(i)
    time.sleep(1)


print("\n🎤 SPEAK NOW!")


# --------------------------------
# RECORD LIVE AUDIO
# --------------------------------

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32"
)

sd.wait()


print("Recording completed!")


# --------------------------------
# SAVE LIVE AUDIO
# --------------------------------

wav.write(
    LIVE_AUDIO_FILE,
    SAMPLE_RATE,
    audio
)


# --------------------------------
# GENERATE LIVE EMBEDDING
# --------------------------------

print("Analyzing Voice...")


signal = classifier.load_audio(
    LIVE_AUDIO_FILE
)


live_embedding = classifier.encode_batch(
    signal
)


live_embedding = live_embedding.squeeze()


# --------------------------------
# CALCULATE SIMILARITY
# --------------------------------

similarity = F.cosine_similarity(
    registered_embedding.unsqueeze(0),
    live_embedding.unsqueeze(0)
)


similarity_score = similarity.item()


# --------------------------------
# DECISION
# --------------------------------

THRESHOLD = 0.50


print("\n-----------------------------")

print(
    f"Similarity Score: {similarity_score:.4f}"
)

print(
    f"Threshold: {THRESHOLD}"
)


if similarity_score >= THRESHOLD:

    print("\n✅ VERIFIED")
    print("Registered speaker detected.")

else:

    print("\n🚨 SPEAKER MISMATCH")
    print("Speaker does not match registered profile.")


print("-----------------------------")