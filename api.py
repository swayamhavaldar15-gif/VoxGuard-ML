from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from supabase import create_client, Client

from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

import torch
import torch.nn.functional as F
import soundfile as sf

import subprocess
import tempfile
import os
import sys


# ============================================================
# AASIST PATH
# ============================================================

ROOT = os.path.dirname(os.path.abspath(__file__))

AASIST_PATH = os.path.join(
    ROOT,
    "anti_spoofing"
)

sys.path.insert(0, AASIST_PATH)


# Import your existing AASIST inference function
from aasist_inference import predict_spoof


# ============================================================
# SUPABASE
# ============================================================

SUPABASE_URL = "https://mdppxocipqgidtgvugis.supabase.co"

SUPABASE_KEY = "sb_publishable_Q4ECgaxcVqHb5ohI1BjBGw_n0i6kvY0"

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="VoxGuard API",
    description="Speaker Verification + Anti-Spoofing + Risk Analysis"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD ECAPA
# ============================================================

print()
print("==========================================")
print("Loading VoxGuard Speaker Recognition Model")
print("==========================================")

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="models/spkrec-ecapa-voxceleb",
    local_strategy=LocalStrategy.COPY
)

print("ECAPA model loaded successfully.")
print()


# ============================================================
# AASIST
# ============================================================

print()
print("==========================================")
print("Loading VoxGuard AASIST Anti-Spoofing Model")
print("==========================================")

print("AASIST model loaded successfully.")
print()


# ============================================================
# AUDIO CONVERSION
# ============================================================

def convert_to_wav(input_path, output_path):

    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-ar",
        "16000",
        "-ac",
        "1",
        output_path
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if result.returncode != 0:

        raise RuntimeError(
            "FFmpeg audio conversion failed:\n"
            + result.stderr.decode(
                errors="ignore"
            )
        )


# ============================================================
# ECAPA EMBEDDING
# ============================================================

def get_embedding(audio_path):

    audio, sample_rate = sf.read(
        audio_path,
        dtype="float32"
    )

    # Stereo -> mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    signal = torch.from_numpy(
        audio
    )

    # Add batch dimension
    signal = signal.unsqueeze(0)

    with torch.no_grad():

        embedding = classifier.encode_batch(
            signal
        )

    embedding = embedding.squeeze()

    embedding = F.normalize(
        embedding,
        p=2,
        dim=0
    )

    return embedding


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    embedding1,
    embedding2
):

    embedding1 = F.normalize(
        embedding1,
        p=2,
        dim=0
    )

    embedding2 = F.normalize(
        embedding2,
        p=2,
        dim=0
    )

    similarity = torch.dot(
        embedding1,
        embedding2
    )

    return float(
        similarity.item()
    )


# ============================================================
# SPEAKER RISK
# ============================================================

def calculate_speaker_risk(
    similarity
):

    if similarity >= 0.50:

        return {
            "speaker_status": "VERIFIED",
            "risk_level": "LOW",
            "risk_score": 10
        }

    elif similarity >= 0.30:

        return {
            "speaker_status": "SUSPICIOUS",
            "risk_level": "MEDIUM",
            "risk_score": 50
        }

    else:

        return {
            "speaker_status": "MISMATCH",
            "risk_level": "HIGH",
            "risk_score": 90
        }


# ============================================================
# COMBINED VOXGUARD RISK ENGINE
# ============================================================

def calculate_final_risk(
    similarity,
    spoof_score,
    speaker_status
):

    # --------------------------------------------------------
    # Base speaker risk
    # --------------------------------------------------------

    speaker_risk = calculate_speaker_risk(
        similarity
    )

    speaker_score = speaker_risk[
        "risk_score"
    ]


    # --------------------------------------------------------
    # AASIST spoof risk
    #
    # spoof_score:
    # 0.0 = more bona-fide
    # 1.0 = more spoof
    # --------------------------------------------------------

    spoof_risk_score = int(
        round(
            spoof_score * 100
        )
    )


    # --------------------------------------------------------
    # Combined score
    #
    # 60% speaker verification
    # 40% anti-spoofing
    # --------------------------------------------------------

    final_score = int(
        round(
            (speaker_score * 0.60)
            +
            (spoof_risk_score * 0.40)
        )
    )


    # --------------------------------------------------------
    # Important:
    # Speaker mismatch should always remain HIGH risk.
    # --------------------------------------------------------

    if speaker_status == "MISMATCH":

        final_score = max(
            final_score,
            80
        )


    # --------------------------------------------------------
    # AASIST currently uses a temporary threshold.
    #
    # We will calibrate this later using validation data.
    # --------------------------------------------------------

    if spoof_score >= 0.50:

        anti_spoof_status = "POTENTIAL_SPOOF"

    else:

        anti_spoof_status = "LIKELY_BONA_FIDE"


    # --------------------------------------------------------
    # Final risk level
    # --------------------------------------------------------

    if final_score >= 70:

        final_risk_level = "HIGH"

    elif final_score >= 40:

        final_risk_level = "MEDIUM"

    else:

        final_risk_level = "LOW"


    return {

        "final_risk_score": final_score,

        "final_risk_level":
            final_risk_level,

        "anti_spoof_status":
            anti_spoof_status

    }


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {

        "project": "VoxGuard",

        "status": "running",

        "services": [

            "ECAPA Speaker Verification",

            "AASIST Anti-Spoofing",

            "Risk Analysis"

        ]

    }


# ============================================================
# REGISTER VOICE
# ============================================================

@app.post("/register-voice")
async def register_voice(
    audio: UploadFile = File(...),
    speaker_name: str = Form(...)
):

    print()
    print("==========================================")
    print("NEW SPEAKER REGISTRATION")
    print("==========================================")

    temp_input = None
    temp_wav = None

    try:

        # ----------------------------------------------------
        # Temporary input
        # ----------------------------------------------------

        input_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".webm"
        )

        temp_input = input_file.name

        input_file.close()


        # ----------------------------------------------------
        # Temporary WAV
        # ----------------------------------------------------

        wav_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        )

        temp_wav = wav_file.name

        wav_file.close()


        # ----------------------------------------------------
        # Save uploaded audio
        # ----------------------------------------------------

        audio_data = await audio.read()

        with open(
            temp_input,
            "wb"
        ) as f:

            f.write(audio_data)


        # ----------------------------------------------------
        # Convert
        # ----------------------------------------------------

        convert_to_wav(
            temp_input,
            temp_wav
        )


        # ----------------------------------------------------
        # ECAPA
        # ----------------------------------------------------

        embedding = get_embedding(
            temp_wav
        )


        embedding_list = (
            embedding
            .cpu()
            .tolist()
        )


        # ----------------------------------------------------
        # Supabase
        # ----------------------------------------------------

        response = supabase.table(
            "speaker_registrations"
        ).insert({

            "speaker_name":
                speaker_name,

            "embedding":
                embedding_list

        }).execute()


        if not response.data:

            raise RuntimeError(
                "Speaker registration failed."
            )


        registration_id = (
            response.data[0]["id"]
        )


        print(
            "Speaker registered successfully."
        )

        print(
            "Speaker:",
            speaker_name
        )

        print(
            "Registration ID:",
            registration_id
        )


        return {

            "success": True,

            "message":
                "Speaker registered successfully",

            "registration_id":
                registration_id,

            "speaker_name":
                speaker_name

        }


    except Exception as e:

        print(
            "Registration error:",
            str(e)
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


    finally:

        if temp_input and os.path.exists(
            temp_input
        ):

            os.remove(temp_input)


        if temp_wav and os.path.exists(
            temp_wav
        ):

            os.remove(temp_wav)


# ============================================================
# VERIFY VOICE
# ============================================================

@app.post("/verify-voice")
async def verify_voice(
    audio: UploadFile = File(...),
    registration_id: int | None = Form(None)
):

    print()
    print("==========================================")
    print("NEW VOXGUARD VOICE VERIFICATION")
    print("==========================================")

    temp_input = None
    temp_wav = None

    try:

        # ----------------------------------------------------
        # Find registered speaker
        # ----------------------------------------------------

        if registration_id is not None:

            response = supabase.table(
                "speaker_registrations"
            ).select(
                "*"
            ).eq(
                "id",
                registration_id
            ).execute()

        else:

            response = supabase.table(
                "speaker_registrations"
            ).select(
                "*"
            ).order(
                "created_at",
                desc=True
            ).limit(
                1
            ).execute()


        if not response.data:

            raise HTTPException(
                status_code=404,
                detail="No registered speaker found."
            )


        registered_speaker = (
            response.data[0]
        )


        registered_embedding = torch.tensor(
            registered_speaker["embedding"],
            dtype=torch.float32
        )


        # ----------------------------------------------------
        # Temporary files
        # ----------------------------------------------------

        input_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".webm"
        )

        temp_input = input_file.name

        input_file.close()


        wav_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        )

        temp_wav = wav_file.name

        wav_file.close()


        # ----------------------------------------------------
        # Save uploaded audio
        # ----------------------------------------------------

        audio_data = await audio.read()

        with open(
            temp_input,
            "wb"
        ) as f:

            f.write(audio_data)


        # ----------------------------------------------------
        # Convert to 16kHz mono WAV
        # ----------------------------------------------------

        convert_to_wav(
            temp_input,
            temp_wav
        )


        # ====================================================
        # ECAPA SPEAKER VERIFICATION
        # ====================================================

        print()
        print("Running ECAPA speaker verification...")

        live_embedding = get_embedding(
            temp_wav
        )


        similarity = cosine_similarity(
            registered_embedding,
            live_embedding
        )


        speaker_risk = calculate_speaker_risk(
            similarity
        )


        print(
            "ECAPA Similarity:",
            round(similarity, 4)
        )

        print(
            "Speaker Status:",
            speaker_risk[
                "speaker_status"
            ]
        )


        # ====================================================
        # AASIST ANTI-SPOOFING
        # ====================================================

        print()
        print("Running AASIST anti-spoofing...")

        aasist_result = predict_spoof(
            temp_wav
        )


        spoof_score = float(
            aasist_result[
                "spoof_score"
            ]
        )

        bona_fide_score = float(
            aasist_result[
                "bona_fide_score"
            ]
        )

        spoof_label = (
            aasist_result[
                "spoof_label"
            ]
        )


        print(
            "AASIST Spoof Score:",
            spoof_score
        )

        print(
            "AASIST Bona-fide Score:",
            bona_fide_score
        )

        print(
            "AASIST Label:",
            spoof_label
        )


        # ====================================================
        # FINAL RISK
        # ====================================================

        final_risk = calculate_final_risk(
            similarity,
            spoof_score,
            speaker_risk[
                "speaker_status"
            ]
        )


        print()
        print("==========================================")
        print("VOXGUARD FINAL RESULT")
        print("==========================================")

        print(
            "Speaker:",
            registered_speaker[
                "speaker_name"
            ]
        )

        print(
            "Similarity:",
            round(similarity, 4)
        )

        print(
            "Speaker Status:",
            speaker_risk[
                "speaker_status"
            ]
        )

        print(
            "Spoof Score:",
            spoof_score
        )

        print(
            "Anti-Spoof:",
            final_risk[
                "anti_spoof_status"
            ]
        )

        print(
            "Final Risk Level:",
            final_risk[
                "final_risk_level"
            ]
        )

        print(
            "Final Risk Score:",
            final_risk[
                "final_risk_score"
            ]
        )

        print()


        # ====================================================
        # SAVE VERIFICATION LOG
        # ====================================================

        log_data = {

            "registration_id":
                registered_speaker[
                    "id"
                ],

            "similarity":
                similarity,

            "speaker_status":
                speaker_risk[
                    "speaker_status"
                ],

            "risk_level":
                final_risk[
                    "final_risk_level"
                ],

            "risk_score":
                final_risk[
                    "final_risk_score"
                ]

        }


        # ----------------------------------------------------
        # Try to save extra AASIST information if columns
        # exist. If they don't, save the original columns.
        # ----------------------------------------------------

        try:

            extended_log = {

                **log_data,

                "spoof_score":
                    spoof_score,

                "bona_fide_score":
                    bona_fide_score,

                "anti_spoof_status":
                    final_risk[
                        "anti_spoof_status"
                    ]

            }

            supabase.table(
                "verification_logs"
            ).insert(
                extended_log
            ).execute()


        except Exception:

            # Existing database schema only
            supabase.table(
                "verification_logs"
            ).insert(
                log_data
            ).execute()


        # ====================================================
        # RETURN RESULT TO FRONTEND
        # ====================================================

        return {

            "success": True,

            "speaker_name":
                registered_speaker[
                    "speaker_name"
                ],

            "registration_id":
                registered_speaker[
                    "id"
                ],

            # ECAPA
            "similarity":
                round(
                    similarity,
                    4
                ),

            "speaker_status":
                speaker_risk[
                    "speaker_status"
                ],


            # AASIST
            "spoof_score":
                round(
                    spoof_score,
                    4
                ),

            "bona_fide_score":
                round(
                    bona_fide_score,
                    4
                ),

            "spoof_label":
                spoof_label,

            "anti_spoof_status":
                final_risk[
                    "anti_spoof_status"
                ],


            # FINAL RISK
            "risk_level":
                final_risk[
                    "final_risk_level"
                ],

            "risk_score":
                final_risk[
                    "final_risk_score"
                ]

        }


    except HTTPException:

        raise


    except Exception as e:

        print()
        print(
            "Verification error:",
            str(e)
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


    finally:

        if temp_input and os.path.exists(
            temp_input
        ):

            os.remove(temp_input)


        if temp_wav and os.path.exists(
            temp_wav
        ):

            os.remove(temp_wav)


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )