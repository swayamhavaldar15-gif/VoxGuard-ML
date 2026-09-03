from fastapi import FastAPI, UploadFile, File
import os
import shutil
import torch
import torch.nn.functional as F

from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy


app = FastAPI(
    title="VoxGuard Speaker Verification API"
)


# ---------------------------------------
# PATHS
# ---------------------------------------

REGISTERED_EMBEDDING_FILE = (
    "speaker_verification/models/registered_embedding.pt"
)

TEMP_AUDIO_FILE = (
    "speaker_verification/models/temp_audio.wav"
)


# ---------------------------------------
# LOAD MODEL ONCE
# ---------------------------------------

print("Loading Speaker Recognition Model...")


classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="models/spkrec-ecapa-voxceleb",
    local_strategy=LocalStrategy.COPY
)


# ---------------------------------------
# LOAD REGISTERED SPEAKER
# ---------------------------------------

registered_embedding = torch.load(
    REGISTERED_EMBEDDING_FILE,
    weights_only=True
)


# ---------------------------------------
# HEALTH CHECK
# ---------------------------------------

@app.get("/")
def home():

    return {
        "message": "VoxGuard Speaker Verification API Running"
    }


# ---------------------------------------
# SPEAKER VERIFICATION
# ---------------------------------------

@app.post("/verify-speaker")
async def verify_speaker(
    audio: UploadFile = File(...)
):

    # Save uploaded audio

    with open(
        TEMP_AUDIO_FILE,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            audio.file,
            buffer
        )


    # Load audio

    signal = classifier.load_audio(
        TEMP_AUDIO_FILE
    )


    # Generate embedding

    live_embedding = classifier.encode_batch(
        signal
    ).squeeze()


    # Cosine similarity

    similarity = F.cosine_similarity(
        registered_embedding.unsqueeze(0),
        live_embedding.unsqueeze(0)
    ).item()


    # -----------------------------------
    # BASIC RISK LOGIC
    # -----------------------------------

    if similarity >= 0.50:

        speaker_status = "VERIFIED"
        risk_score = 10
        risk_level = "LOW"

    elif similarity >= 0.30:

        speaker_status = "SUSPICIOUS"
        risk_score = 50
        risk_level = "MEDIUM"

    else:

        speaker_status = "MISMATCH"
        risk_score = 90
        risk_level = "HIGH"


    # -----------------------------------
    # RESPONSE
    # -----------------------------------

    return {

        "speaker_similarity": round(
            similarity,
            4
        ),

        "speaker_status": speaker_status,

        "impersonation_risk": risk_level,

        "risk_score": risk_score

    }