import os
import torch
import torch.nn.functional as F
import soundfile as sf

from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy


# ============================================================
# FILE PATHS
# ============================================================

REGISTERED_EMBEDDING_FILE = (
    "speaker_verification/models/registered_embedding.pt"
)

ENROLLMENT_AUDIO = (
    "speaker_verification/models/enrollment_audio.wav"
)

LIVE_AUDIO = (
    "speaker_verification/models/live_test.wav"
)


# ============================================================
# CHECK FILES
# ============================================================

print("=" * 60)
print("VOXGUARD - LIVE ECAPA SPEAKER DIAGNOSTIC")
print("=" * 60)

for file_path in [
    REGISTERED_EMBEDDING_FILE,
    ENROLLMENT_AUDIO,
    LIVE_AUDIO
]:
    if not os.path.exists(file_path):
        print(f"\nERROR: File not found:")
        print(file_path)
        raise SystemExit


# ============================================================
# LOAD SPEAKER MODEL
# ============================================================

print("\nLoading Speaker Recognition Model...")

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="models/spkrec-ecapa-voxceleb",
    local_strategy=LocalStrategy.COPY
)

print("Speaker model loaded successfully.")


# ============================================================
# FUNCTIONS
# ============================================================

def get_embedding(audio_path):

    signal = classifier.load_audio(audio_path)

    print(f"\nAudio: {audio_path}")
    print(f"Loaded audio shape: {signal.shape}")
    print(f"Audio dtype: {signal.dtype}")

    with torch.no_grad():

        embedding = classifier.encode_batch(signal)

    print(f"Raw embedding shape: {embedding.shape}")

    embedding = embedding.squeeze()

    print(f"Squeezed embedding shape: {embedding.shape}")

    embedding = F.normalize(
        embedding,
        p=2,
        dim=0
    )

    print(f"Final embedding shape: {embedding.shape}")

    return embedding


def cosine_similarity(embedding1, embedding2):

    return F.cosine_similarity(
        embedding1.unsqueeze(0),
        embedding2.unsqueeze(0)
    ).item()


def inspect_audio(audio_path):

    data, sample_rate = sf.read(audio_path)

    print("\n" + "-" * 60)
    print("AUDIO INFORMATION")
    print("-" * 60)

    print(f"File: {audio_path}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Shape: {data.shape}")

    if len(data.shape) == 1:
        mono_data = data
        print("Channels: Mono")
    else:
        print(f"Channels: {data.shape[1]}")
        mono_data = data.mean(axis=1)

    print(f"Samples: {len(mono_data)}")

    duration = len(mono_data) / sample_rate

    print(f"Duration: {duration:.2f} seconds")

    rms = (mono_data ** 2).mean() ** 0.5
    peak = abs(mono_data).max()

    print(f"RMS volume: {rms:.6f}")
    print(f"Peak volume: {peak:.6f}")

    if rms < 0.005:
        print("WARNING: Audio volume is very low.")

    elif rms < 0.02:
        print("Audio volume: LOW")

    else:
        print("Audio volume: GOOD")

    if peak < 0.05:
        print("WARNING: Very weak microphone signal.")

    return sample_rate, mono_data


# ============================================================
# INSPECT LIVE AUDIO
# ============================================================

print("\nChecking live microphone recording...")

inspect_audio(LIVE_AUDIO)


# ============================================================
# LOAD REGISTERED EMBEDDING
# ============================================================

print("\n" + "=" * 60)
print("LOADING REGISTERED SPEAKER PROFILE")
print("=" * 60)

registered_embedding = torch.load(
    REGISTERED_EMBEDDING_FILE,
    map_location="cpu"
)

print(
    f"Registered embedding shape BEFORE normalization: "
    f"{registered_embedding.shape}"
)

registered_embedding = registered_embedding.squeeze()

registered_embedding = F.normalize(
    registered_embedding,
    p=2,
    dim=0
)

print(
    f"Registered embedding shape AFTER normalization: "
    f"{registered_embedding.shape}"
)


# ============================================================
# CREATE FRESH ENROLLMENT EMBEDDING
# ============================================================

print("\n" + "=" * 60)
print("TEST 1: REGISTERED PROFILE VS ENROLLMENT AUDIO")
print("=" * 60)

fresh_enrollment_embedding = get_embedding(
    ENROLLMENT_AUDIO
)

registered_vs_enrollment = cosine_similarity(
    registered_embedding,
    fresh_enrollment_embedding
)

print(
    f"\nRegistered vs Enrollment Similarity: "
    f"{registered_vs_enrollment:.4f}"
)


# ============================================================
# CREATE LIVE EMBEDDING
# ============================================================

print("\n" + "=" * 60)
print("TEST 2: REGISTERED PROFILE VS LIVE AUDIO")
print("=" * 60)

live_embedding = get_embedding(
    LIVE_AUDIO
)

registered_vs_live = cosine_similarity(
    registered_embedding,
    live_embedding
)

print(
    f"\nRegistered vs Live Similarity: "
    f"{registered_vs_live:.4f}"
)


# ============================================================
# LIVE VS FRESH ENROLLMENT
# ============================================================

print("\n" + "=" * 60)
print("TEST 3: ENROLLMENT AUDIO VS LIVE AUDIO")
print("=" * 60)

enrollment_vs_live = cosine_similarity(
    fresh_enrollment_embedding,
    live_embedding
)

print(
    f"\nEnrollment vs Live Similarity: "
    f"{enrollment_vs_live:.4f}"
)


# ============================================================
# FINAL DIAGNOSIS
# ============================================================

print("\n" + "=" * 60)
print("FINAL DIAGNOSIS")
print("=" * 60)

print(
    f"Registered vs Enrollment : "
    f"{registered_vs_enrollment:.4f}"
)

print(
    f"Registered vs Live       : "
    f"{registered_vs_live:.4f}"
)

print(
    f"Enrollment vs Live       : "
    f"{enrollment_vs_live:.4f}"
)


print("\n" + "-" * 60)
print("INTERPRETATION")
print("-" * 60)


# ------------------------------------------------------------
# CHECK REGISTERED PROFILE
# ------------------------------------------------------------

if registered_vs_enrollment >= 0.90:

    print("✓ Registered speaker profile looks consistent.")

elif registered_vs_enrollment >= 0.70:

    print("⚠ Registered profile is somewhat consistent.")

else:

    print(
        "✗ Registered profile may not match the current "
        "enrollment audio."
    )


# ------------------------------------------------------------
# CHECK LIVE SPEAKER
# ------------------------------------------------------------

if registered_vs_live >= 0.50:

    print(
        "✓ Live voice is reasonably similar to registered speaker."
    )

elif registered_vs_live >= 0.30:

    print(
        "⚠ Live voice has moderate similarity."
    )

else:

    print(
        "✗ Live voice has LOW similarity to registered speaker."
    )


# ------------------------------------------------------------
# CHECK LIVE VS ENROLLMENT
# ------------------------------------------------------------

if enrollment_vs_live >= 0.50:

    print(
        "✓ Live audio also resembles the enrollment recording."
    )

elif enrollment_vs_live >= 0.30:

    print(
        "⚠ Live audio has moderate similarity to enrollment."
    )

else:

    print(
        "✗ Live audio does not strongly resemble enrollment audio."
    )


# ============================================================
# IMPORTANT RESULT
# ============================================================

print("\n" + "=" * 60)
print("VOXGUARD RESULT")
print("=" * 60)

if (
    registered_vs_live >= 0.50
    and enrollment_vs_live >= 0.50
):

    print("SPEAKER VERIFICATION: LIKELY MATCH")

else:

    print("SPEAKER VERIFICATION: NOT CONFIRMED")

print("=" * 60)