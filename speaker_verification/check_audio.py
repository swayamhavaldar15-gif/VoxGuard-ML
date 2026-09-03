import soundfile as sf
import os

audio_files = [
    "speaker_verification/reference_audio/swayam_reference.wav",
    "speaker_verification/test_audio/swayam_test.wav",
    "speaker_verification/test_audio/different_person.wav"
]

for audio_file in audio_files:

    print("\nChecking:", audio_file)

    if not os.path.exists(audio_file):
        print("❌ FILE NOT FOUND")
        continue

    try:
        data, sample_rate = sf.read(audio_file)

        print("✅ Audio is readable")
        print("Sample Rate:", sample_rate)
        print("Shape:", data.shape)

    except Exception as e:
        print("❌ Cannot read this audio")
        print("Error:", e)