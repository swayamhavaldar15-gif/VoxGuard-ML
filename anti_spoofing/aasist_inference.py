import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import soundfile as sf


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

AASIST_ROOT = ROOT / "aasist"

CHECKPOINT = (
    AASIST_ROOT
    / "models"
    / "weights"
    / "AASIST.pth"
)


# ============================================================
# AASIST MODEL CONFIGURATION
# ============================================================

MODEL_CONFIG = {
    "architecture": "AASIST",
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [
        70,
        [1, 32],
        [32, 32],
        [32, 64],
        [64, 64],
    ],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}


# ============================================================
# AUDIO SETTINGS
# ============================================================

TARGET_SAMPLE_RATE = 16000
TARGET_SAMPLES = 64600


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD AASIST MODEL
# ============================================================

print("Loading AASIST model...")
print(f"Device: {DEVICE}")
print(f"Checkpoint: {CHECKPOINT}")


if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"AASIST checkpoint not found:\n{CHECKPOINT}"
    )


# Add AASIST repository to Python path
sys.path.insert(0, str(AASIST_ROOT))


from models.AASIST import Model


# Create model
MODEL = Model(MODEL_CONFIG).to(DEVICE)


# Load pretrained weights
checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=True
)


MODEL.load_state_dict(checkpoint)

MODEL.eval()


print("AASIST model loaded successfully.")


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(audio_path):

    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found:\n{audio_path}"
        )

    print()
    print("Loading audio...")


    # Read WAV file
    waveform, sample_rate = sf.read(
        str(audio_path),
        dtype="float32"
    )


    print(f"Original sample rate: {sample_rate} Hz")


    # --------------------------------------------------------
    # Check sample rate
    # --------------------------------------------------------

    if sample_rate != TARGET_SAMPLE_RATE:

        raise ValueError(
            "\n"
            "AASIST currently expects 16 kHz audio.\n"
            f"Your audio is {sample_rate} Hz.\n\n"
            "Please convert the WAV file to 16 kHz "
            "before running AASIST."
        )

    print("Audio already at 16 kHz.")


    # --------------------------------------------------------
    # Convert stereo → mono
    # --------------------------------------------------------

    if waveform.ndim > 1:

        waveform = waveform.mean(axis=1)

        print("Converted stereo audio to mono.")

    else:

        print("Audio is already mono.")


    # --------------------------------------------------------
    # Convert NumPy → PyTorch tensor
    # --------------------------------------------------------

    waveform = torch.from_numpy(waveform)


    # Remove invalid values
    waveform = torch.nan_to_num(
        waveform,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


    # --------------------------------------------------------
    # Audio length
    # --------------------------------------------------------

    original_samples = waveform.numel()

    print(
        f"Audio length: {original_samples} samples"
    )


    # --------------------------------------------------------
    # Trim or pad to AASIST expected length
    # --------------------------------------------------------

    if original_samples >= TARGET_SAMPLES:

        waveform = waveform[:TARGET_SAMPLES]

        print(
            f"Using first {TARGET_SAMPLES} samples."
        )

    else:

        padding = (
            TARGET_SAMPLES - original_samples
        )

        waveform = F.pad(
            waveform,
            (0, padding)
        )

        print(
            f"Padded with {padding} samples."
        )


    return waveform


# ============================================================
# AASIST PREDICTION
# ============================================================

def predict_spoof(audio_path):

    # Load and prepare audio
    waveform = load_audio(audio_path)


    # Add batch dimension
    #
    # Shape:
    # [64600]
    #
    # becomes:
    # [1, 64600]

    waveform = waveform.unsqueeze(0)


    # Move to CPU/GPU
    waveform = waveform.to(DEVICE)


    # --------------------------------------------------------
    # Run AASIST
    # --------------------------------------------------------

    with torch.no_grad():

        _, output = MODEL(waveform)


        # Convert model output to probabilities
        class_probabilities = torch.softmax(
            output,
            dim=1
        )


        # AASIST:
        #
        # Class 0 = spoof
        # Class 1 = bona fide
        #
        # Therefore:
        #
        # bona_fide_score = probability of real speech

        bona_fide_score = (
            class_probabilities[:, 1].item()
        )


    # --------------------------------------------------------
    # Convert to VoxGuard spoof score
    # --------------------------------------------------------

    #
    # Higher spoof_score = higher spoof risk
    #

    spoof_score = 1.0 - bona_fide_score


    # --------------------------------------------------------
    # Initial label
    # --------------------------------------------------------

    #
    # NOTE:
    # 0.50 is currently a TEMPORARY threshold.
    #
    # We will calibrate this later using real test data.
    #

    if spoof_score >= 0.50:

        spoof_label = "spoof"

    else:

        spoof_label = "bonafide"


    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {

        "spoof_score": round(
            spoof_score,
            4
        ),

        "spoof_label": spoof_label,

        "bona_fide_score": round(
            bona_fide_score,
            4
        )

    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Check command-line arguments
    # --------------------------------------------------------

    if len(sys.argv) != 2:

        print()
        print("Usage:")
        print(
            "python anti_spoofing\\aasist_inference.py <audio.wav>"
        )
        print()

        sys.exit(1)


    # Get audio file
    audio_file = Path(sys.argv[1])


    # --------------------------------------------------------
    # Check file
    # --------------------------------------------------------

    if not audio_file.exists():

        print()
        print("ERROR: Audio file not found:")
        print(audio_file)
        print()

        sys.exit(1)


    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("VOXGUARD AASIST ANTI-SPOOFING TEST")
    print("=" * 60)
    print()


    # --------------------------------------------------------
    # Run prediction
    # --------------------------------------------------------

    try:

        result = predict_spoof(
            audio_file
        )

    except Exception as e:

        print()
        print(
            "ERROR during AASIST inference:"
        )
        print()
        print(str(e))
        print()

        sys.exit(1)


    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print()
    print(f"Audio:            {audio_file}")

    print(
        f"Spoof Score:      {result['spoof_score']}"
    )

    print(
        f"Spoof Label:      {result['spoof_label']}"
    )

    print(
        f"Bona-fide Score:  {result['bona_fide_score']}"
    )

    print()
    print("=" * 60)
    print("AASIST TEST COMPLETE")
    print("=" * 60)