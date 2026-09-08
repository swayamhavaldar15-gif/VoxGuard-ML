import os
import torch
import torch.nn.functional as F

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

REFERENCE_AUDIO = (
    "speaker_verification/reference_audio/swayam_reference.wav"
)

SAME_SPEAKER_AUDIO = (
    "speaker_verification/test_audio/swayam_test.wav"
)

DIFFERENT_SPEAKER_AUDIO = (
    "speaker_verification/test_audio/different_person.wav"
)

LIVE_AUDIO = (
    "speaker_verification/models/live_test.wav"
)


# ============================================================
# CHECK FILES
# ============================================================

print("=" * 60)
print("VOXGUARD - SPEAKER VERIFICATION")
print("=" * 60)

files_to_check = [
    REGISTERED_EMBEDDING_FILE,
    ENROLLMENT_AUDIO,
    REFERENCE_AUDIO,
    SAME_SPEAKER_AUDIO,
    DIFFERENT_SPEAKER_AUDIO,
    LIVE_AUDIO
]

for file_path in files_to_check:

    if os.path.exists(file_path):
        print(f"FOUND: {file_path}")

    else:
        print(f"NOT FOUND: {file_path}")


# ============================================================
# LOAD ECAPA SPEAKER MODEL
# ============================================================

print("\n" + "=" * 60)
print("LOADING SPEAKER RECOGNITION MODEL")
print("=" * 60)

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="models/spkrec-ecapa-voxceleb",
    local_strategy=LocalStrategy.COPY
)

print("Speaker model loaded successfully.")


# ============================================================
# EMBEDDING FUNCTION
# ============================================================

def get_embedding(audio_path):

    signal = classifier.load_audio(audio_path)

    print(f"\nProcessing: {audio_path}")
    print(f"Audio tensor shape: {signal.shape}")
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


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(embedding1, embedding2):

    return F.cosine_similarity(
        embedding1.unsqueeze(0),
        embedding2.unsqueeze(0)
    ).item()


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
    "Registered embedding shape:",
    registered_embedding.shape
)

registered_embedding = registered_embedding.squeeze()

registered_embedding = F.normalize(
    registered_embedding,
    p=2,
    dim=0
)

print(
    "Normalized registered embedding shape:",
    registered_embedding.shape
)


# ============================================================
# 1. ENROLLMENT VS REGISTERED
# ============================================================

print("\n" + "=" * 60)
print("TEST 1: REGISTERED PROFILE VS ENROLLMENT")
print("=" * 60)

enrollment_embedding = get_embedding(
    ENROLLMENT_AUDIO
)

registered_enrollment_similarity = cosine_similarity(
    registered_embedding,
    enrollment_embedding
)

print(
    f"\nRegistered vs Enrollment similarity: "
    f"{registered_enrollment_similarity:.4f}"
)


# ============================================================
# 2. REFERENCE VS REGISTERED
# ============================================================

print("\n" + "=" * 60)
print("TEST 2: REGISTERED PROFILE VS REFERENCE")
print("=" * 60)

reference_embedding = get_embedding(
    REFERENCE_AUDIO
)

registered_reference_similarity = cosine_similarity(
    registered_embedding,
    reference_embedding
)

print(
    f"\nRegistered vs Reference similarity: "
    f"{registered_reference_similarity:.4f}"
)


# ============================================================
# 3. SAME SPEAKER VS REGISTERED
# ============================================================

print("\n" + "=" * 60)
print("TEST 3: REGISTERED PROFILE VS SAME SPEAKER")
print("=" * 60)

same_speaker_embedding = get_embedding(
    SAME_SPEAKER_AUDIO
)

registered_same_similarity = cosine_similarity(
    registered_embedding,
    same_speaker_embedding
)

print(
    f"\nRegistered vs Same Speaker similarity: "
    f"{registered_same_similarity:.4f}"
)


# ============================================================
# 4. DIFFERENT SPEAKER VS REGISTERED
# ============================================================

print("\n" + "=" * 60)
print("TEST 4: REGISTERED PROFILE VS DIFFERENT SPEAKER")
print("=" * 60)

different_speaker_embedding = get_embedding(
    DIFFERENT_SPEAKER_AUDIO
)

registered_different_similarity = cosine_similarity(
    registered_embedding,
    different_speaker_embedding
)

print(
    f"\nRegistered vs Different Speaker similarity: "
    f"{registered_different_similarity:.4f}"
)


# ============================================================
# 5. LIVE AUDIO VS REGISTERED
# ============================================================

if os.path.exists(LIVE_AUDIO):

    print("\n" + "=" * 60)
    print("TEST 5: REGISTERED PROFILE VS LIVE AUDIO")
    print("=" * 60)

    live_embedding = get_embedding(
        LIVE_AUDIO
    )

    registered_live_similarity = cosine_similarity(
        registered_embedding,
        live_embedding
    )

    print(
        f"\nRegistered vs Live similarity: "
        f"{registered_live_similarity:.4f}"
    )

else:

    registered_live_similarity = None

    print("\nLIVE AUDIO NOT FOUND.")
    print("Run live_verify.py first.")


# ============================================================
# LIVE VS FRESH ENROLLMENT
# ============================================================

if os.path.exists(LIVE_AUDIO):

    print("\n" + "=" * 60)
    print("TEST 6: ENROLLMENT VS LIVE AUDIO")
    print("=" * 60)

    enrollment_live_similarity = cosine_similarity(
        enrollment_embedding,
        live_embedding
    )

    print(
        f"\nEnrollment vs Live similarity: "
        f"{enrollment_live_similarity:.4f}"
    )

else:

    enrollment_live_similarity = None


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL SPEAKER VERIFICATION RESULTS")
print("=" * 60)

print(
    f"Registered vs Enrollment : "
    f"{registered_enrollment_similarity:.4f}"
)

print(
    f"Registered vs Reference  : "
    f"{registered_reference_similarity:.4f}"
)

print(
    f"Registered vs Same       : "
    f"{registered_same_similarity:.4f}"
)

print(
    f"Registered vs Different  : "
    f"{registered_different_similarity:.4f}"
)

if registered_live_similarity is not None:

    print(
        f"Registered vs Live       : "
        f"{registered_live_similarity:.4f}"
    )

    print(
        f"Enrollment vs Live      : "
        f"{enrollment_live_similarity:.4f}"
    )


# ============================================================
# BASIC BEHAVIOR CHECK
# ============================================================

print("\n" + "=" * 60)
print("ECAPA BEHAVIOR CHECK")
print("=" * 60)

if registered_same_similarity > registered_different_similarity:

    print("✓ SAME speaker > DIFFERENT speaker")
    print("✓ ECAPA speaker separation looks correct.")

else:

    print("✗ SAME speaker is not greater than DIFFERENT speaker.")
    print("✗ Investigate the speaker model/audio setup.")


# ============================================================
# LIVE SPEAKER CHECK
# ============================================================

if registered_live_similarity is not None:

    print("\n" + "=" * 60)
    print("LIVE SPEAKER CHECK")
    print("=" * 60)

    if registered_live_similarity >= 0.50:

        print("✓ LIVE SPEAKER: LIKELY MATCH")

    elif registered_live_similarity >= 0.30:

        print("⚠ LIVE SPEAKER: SUSPICIOUS / UNCERTAIN")

    else:

        print("✗ LIVE SPEAKER: MISMATCH")


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)
print("IMPORTANT")
print("=" * 60)

print("These similarity thresholds are temporary.")
print("They must be calibrated using multiple genuine and")
print("impostor recordings before being used in the final system.")

print("=" * 60)