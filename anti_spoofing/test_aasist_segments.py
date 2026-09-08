
import soundfile as sf
from pathlib import Path
import subprocess
import sys


INPUT = "speaker_verification/models/enrollment_audio.wav"
OUTPUT_DIR = Path("anti_spoofing/aasist_test_audio")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

audio, sr = sf.read(INPUT, dtype="float32")

print("=" * 60)
print("CREATING AASIST TEST SEGMENTS")
print("=" * 60)

segment_seconds = 4
segment_samples = sr * segment_seconds

total_segments = len(audio) // segment_samples

print(f"Sample rate: {sr}")
print(f"Total duration: {len(audio) / sr:.2f} seconds")
print(f"Creating {total_segments} segments...\n")

for i in range(total_segments):
    start = i * segment_samples
    end = start + segment_samples

    segment = audio[start:end]

    output = OUTPUT_DIR / f"enrollment_segment_{i+1}.wav"

    sf.write(
        str(output),
        segment,
        sr
    )

    print(f"Created: {output}")

print("\nSegments created successfully.")
print(f"Location: {OUTPUT_DIR}")