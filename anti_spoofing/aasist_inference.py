from pathlib import Path
import importlib.util
import sys

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

AASIST_MODEL_FILE = BASE_DIR / "AASIST.py"
WEIGHTS_FILE = BASE_DIR / "aasist" / "models" / "weights" / "AASIST.pth"

DEVICE = torch.device("cpu")

print("=" * 60)
print("VoxGuard AASIST Anti-Spoofing")
print("Device:", DEVICE)
print("AASIST model:", AASIST_MODEL_FILE)
print("AASIST weights:", WEIGHTS_FILE)
print("=" * 60)


# ============================================================
# AASIST configuration
# ============================================================

MODEL_CONFIG = {
    "architecture": "AASIST",
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}


# ============================================================
# Load AASIST Model class
# ============================================================

def load_aasist_model_class():
    if not AASIST_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"AASIST model source file not found:\n{AASIST_MODEL_FILE}"
        )

    spec = importlib.util.spec_from_file_location(
        "voxguard_aasist_model",
        str(AASIST_MODEL_FILE)
    )

    module = importlib.util.module_from_spec(spec)

    # AASIST.py imports other model files using "models..."
    models_dir = BASE_DIR / "aasist" / "models"

    if str(models_dir) not in sys.path:
        sys.path.insert(0, str(models_dir.parent))

    spec.loader.exec_module(module)

    return module.Model


# ============================================================
# Load model
# ============================================================

def load_aasist_model():

    if not WEIGHTS_FILE.exists():
        raise FileNotFoundError(
            f"AASIST weights not found:\n{WEIGHTS_FILE}"
        )

    print("Loading AASIST model...")

    Model = load_aasist_model_class()

    model = Model(MODEL_CONFIG)

    checkpoint = torch.load(
        WEIGHTS_FILE,
        map_location=DEVICE,
        weights_only=False
    )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    print("AASIST model loaded successfully.")

    return model


AASISTModel = load_aasist_model()


# ============================================================
# Audio preprocessing
# ============================================================

def load_audio(audio_path):

    audio, sample_rate = sf.read(audio_path)

    # Stereo -> mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)

    # Convert to torch tensor
    audio = torch.from_numpy(audio)

    # Resample to 16 kHz if required
    if sample_rate != 16000:

        audio = audio.unsqueeze(0).unsqueeze(0)

        new_length = int(
            audio.shape[-1] * 16000 / sample_rate
        )

        audio = F.interpolate(
            audio,
            size=new_length,
            mode="linear",
            align_corners=False
        )

        audio = audio.squeeze()

    return audio


# ============================================================
# Predict spoof
# ============================================================

def predict_spoof(audio_path):

    audio = load_audio(audio_path)

    required_length = MODEL_CONFIG["nb_samp"]

    # Make exactly 64600 samples
    if audio.numel() < required_length:

        audio = F.pad(
            audio,
            (0, required_length - audio.numel())
        )

    else:

        audio = audio[:required_length]

    # Shape: [batch, channel, samples]
    audio = audio.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        _, output = AASISTModel(audio)

        probabilities = torch.softmax(output, dim=1)

    # Official AASIST convention:
    # class 0 = spoof
    # class 1 = bona fide

    spoof_score = float(probabilities[0, 0].item())
    bona_fide_score = float(probabilities[0, 1].item())

    if spoof_score >= 0.50:
        label = "SPOOF"
        anti_spoof_status = "POTENTIAL_SPOOF"
    else:
        label = "BONAFIDE"
        anti_spoof_status = "LIKELY_BONA_FIDE"

    return {
        "spoof_score": spoof_score,
        "bona_fide_score": bona_fide_score,
        "label": label,
        "anti_spoof_status": anti_spoof_status,
    }


# ============================================================
# Command-line test
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print()
        print("Usage:")
        print(
            "python anti_spoofing/aasist_inference.py <audio.wav>"
        )
        sys.exit(1)

    audio_file = sys.argv[1]

    result = predict_spoof(audio_file)

    print()
    print("AASIST RESULT")
    print("-" * 40)
    print("Spoof score:      ", result["spoof_score"])
    print("Bona fide score:  ", result["bona_fide_score"])
    print("Label:             ", result["label"])
    print("Anti-spoof status: ", result["anti_spoof_status"])