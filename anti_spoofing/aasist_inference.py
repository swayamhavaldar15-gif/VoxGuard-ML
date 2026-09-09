"""
VoxGuard - AASIST Anti-Spoofing Inference

Uses the official AASIST model from:
https://github.com/clovaai/aasist

Designed to work both:
1. Locally on Windows
2. On Render Linux deployment

The model detects whether speech is likely:
- BONAFIDE / genuine
- SPOOF / potentially replayed, synthesized, or manipulated
"""

from pathlib import Path
import importlib.util

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F


# ============================================================
# PATHS
# ============================================================

# Project root:
# VoxGuard-ML/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Official AASIST repository:
# VoxGuard-ML/anti_spoofing/aasist/
AASIST_ROOT = PROJECT_ROOT / "anti_spoofing" / "aasist"

# Official AASIST model file:
AASIST_MODEL_FILE = AASIST_ROOT / "models" / "AASIST.py"

# Pretrained weights:
AASIST_WEIGHTS = (
    AASIST_ROOT
    / "models"
    / "weights"
    / "AASIST.pth"
)


# ============================================================
# AASIST CONFIGURATION
# ============================================================

AASIST_CONFIG = {
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


# AASIST works with 16 kHz audio.
SAMPLE_RATE = 16000

# Official AASIST input size.
NUM_SAMPLES = 64600


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("==============================================")
print("VoxGuard AASIST Anti-Spoofing")
print("==============================================")
print(f"Device: {DEVICE}")
print(f"AASIST root: {AASIST_ROOT}")
print(f"AASIST model: {AASIST_MODEL_FILE}")
print(f"AASIST weights: {AASIST_WEIGHTS}")


# ============================================================
# LOAD AASIST MODEL CLASS
# ============================================================

def load_aasist_model_class():
    """
    Load the official AASIST Model class directly from
    models/AASIST.py.

    This avoids the error:

        ModuleNotFoundError:
        No module named 'models.AASIST'

    because we do not depend on the current working directory
    or Python package search path.
    """

    if not AASIST_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"AASIST model source file was not found:\n"
            f"{AASIST_MODEL_FILE}"
        )

    print("Loading official AASIST model source...")

    spec = importlib.util.spec_from_file_location(
        "voxguard_aasist_model",
        str(AASIST_MODEL_FILE),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load AASIST module from:\n"
            f"{AASIST_MODEL_FILE}"
        )

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    if not hasattr(module, "Model"):
        raise AttributeError(
            "AASIST.py does not contain the expected "
            "'Model' class."
        )

    print("AASIST Model class loaded successfully.")

    return module.Model


# ============================================================
# INITIALIZE MODEL
# ============================================================

print("Loading AASIST model...")

if not AASIST_WEIGHTS.exists():
    raise FileNotFoundError(
        "\nAASIST pretrained weights were not found.\n"
        f"Expected file:\n{AASIST_WEIGHTS}\n\n"
        "Make sure the Render build command downloads "
        "AASIST.pth before starting the server."
    )


# Load the class directly from the official source.
AASISTModel = load_aasist_model_class()


# Create model.
model = AASISTModel(AASIST_CONFIG)


# Load pretrained weights.
print("Loading AASIST checkpoint...")

checkpoint = torch.load(
    AASIST_WEIGHTS,
    map_location=DEVICE,
)


# Some checkpoints may contain a state_dict wrapper.
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]


# Load weights.
model.load_state_dict(checkpoint)


# Move model to CPU/GPU.
model.to(DEVICE)


# Evaluation mode.
model.eval()


print("AASIST model loaded successfully.")
print("==============================================")


# ============================================================
# AUDIO PREPROCESSING
# ============================================================

def load_audio(audio_path: str):
    """
    Load an audio file and convert it into:
        - mono
        - float32
        - 16 kHz
        - exactly 64600 samples
    """

    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found:\n{audio_path}"
        )

    # Read audio using soundfile.
    audio, sample_rate = sf.read(
        str(audio_path),
        dtype="float32",
    )

    # --------------------------------------------------------
    # Convert stereo -> mono
    # --------------------------------------------------------

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # --------------------------------------------------------
    # Convert to float32
    # --------------------------------------------------------

    audio = audio.astype(np.float32)

    # --------------------------------------------------------
    # Resample if necessary
    # --------------------------------------------------------

    if sample_rate != SAMPLE_RATE:

        print(
            f"Resampling audio: "
            f"{sample_rate} Hz -> {SAMPLE_RATE} Hz"
        )

        # Use torch interpolation instead of scipy.
        waveform = torch.from_numpy(audio)

        new_length = int(
            len(audio)
            * SAMPLE_RATE
            / sample_rate
        )

        waveform = waveform.unsqueeze(0).unsqueeze(0)

        waveform = F.interpolate(
            waveform,
            size=new_length,
            mode="linear",
            align_corners=False,
        )

        audio = (
            waveform
            .squeeze()
            .numpy()
            .astype(np.float32)
        )

    # --------------------------------------------------------
    # Convert to torch tensor
    # --------------------------------------------------------

    audio_tensor = torch.from_numpy(audio)

    # --------------------------------------------------------
    # Make exactly 64600 samples
    # --------------------------------------------------------

    if audio_tensor.numel() < NUM_SAMPLES:

        # Pad short audio with zeros.
        padding = NUM_SAMPLES - audio_tensor.numel()

        audio_tensor = F.pad(
            audio_tensor,
            (0, padding),
        )

    elif audio_tensor.numel() > NUM_SAMPLES:

        # Take first 64600 samples.
        audio_tensor = audio_tensor[:NUM_SAMPLES]

    return audio_tensor


# ============================================================
# SPOOF PREDICTION
# ============================================================

def predict_spoof(audio_path: str):
    """
    Run AASIST on an audio file.

    Returns:

        spoof_score
            Probability-like score for spoof class.

        bona_fide_score
            Probability-like score for genuine speech.

        label
            "SPOOF" or "BONAFIDE"

        anti_spoof_status
            Human-readable status.
    """

    print("----------------------------------------------")
    print("Running AASIST anti-spoofing...")
    print(f"Audio: {audio_path}")

    # Load and preprocess audio.
    audio = load_audio(audio_path)

    # Add batch dimension.
    audio = audio.unsqueeze(0)

    # Move to CPU/GPU.
    audio = audio.to(DEVICE)

    # --------------------------------------------------------
    # Run inference
    # --------------------------------------------------------

    with torch.no_grad():

        _, logits = model(audio)

        # Convert logits to probabilities.
        probabilities = torch.softmax(
            logits,
            dim=1,
        )

    # --------------------------------------------------------
    # Official AASIST output:
    #
    # class 0 = spoof
    # class 1 = bona fide
    # --------------------------------------------------------

    spoof_score = float(
        probabilities[0, 0].item()
    )

    bona_fide_score = float(
        probabilities[0, 1].item()
    )

    # --------------------------------------------------------
    # Determine label
    # --------------------------------------------------------

    if spoof_score >= 0.50:

        label = "SPOOF"

        anti_spoof_status = (
            "POTENTIAL_SPOOF"
        )

    else:

        label = "BONAFIDE"

        anti_spoof_status = (
            "LIKELY_BONA_FIDE"
        )

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print(
        f"Spoof score: {spoof_score:.4f}"
    )

    print(
        f"Bona fide score: "
        f"{bona_fide_score:.4f}"
    )

    print(
        f"AASIST label: {label}"
    )

    print(
        f"Anti-spoof status: "
        f"{anti_spoof_status}"
    )

    print("----------------------------------------------")

    return {
        "spoof_score": spoof_score,
        "bona_fide_score": bona_fide_score,
        "label": label,
        "anti_spoof_status": anti_spoof_status,
    }


# ============================================================
# TEST MODE
# ============================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print(
            "\nUsage:"
        )

        print(
            "python anti_spoofing/aasist_inference.py "
            "<audio_file.wav>"
        )

        sys.exit(1)

    test_audio = sys.argv[1]

    result = predict_spoof(test_audio)

    print("\nFinal result:")
    print(result)