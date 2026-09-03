import torch
import torch.nn.functional as F

from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy


print("Loading speaker embedding model...")


classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="models/spkrec-ecapa-voxceleb",
    local_strategy=LocalStrategy.COPY
)


reference_audio = "speaker_verification/reference_audio/swayam_reference.wav"

same_speaker_audio = "speaker_verification/test_audio/swayam_test.wav"

different_speaker_audio = "speaker_verification/test_audio/different_person.wav"


def get_embedding(audio_file):

    signal = classifier.load_audio(audio_file)

    embedding = classifier.encode_batch(signal)

    return embedding.squeeze()


def compare_speakers(reference, test):

    reference_embedding = get_embedding(reference)

    test_embedding = get_embedding(test)

    similarity = F.cosine_similarity(
        reference_embedding.unsqueeze(0),
        test_embedding.unsqueeze(0)
    )

    return similarity.item()


# --------------------------------
# TEST 1: SAME SPEAKER
# --------------------------------

print("\nTesting SAME speaker...\n")

same_score = compare_speakers(
    reference_audio,
    same_speaker_audio
)

print("Cosine Similarity:", same_score)


# --------------------------------
# TEST 2: DIFFERENT SPEAKER
# --------------------------------

print("\nTesting DIFFERENT speaker...\n")

different_score = compare_speakers(
    reference_audio,
    different_speaker_audio
)

print("Cosine Similarity:", different_score)


# --------------------------------
# COMPARISON
# --------------------------------

print("\n----------------------------")

if same_score > different_score:
    print("RESULT: GOOD")
    print("Same speaker has higher similarity.")
else:
    print("RESULT: NEEDS IMPROVEMENT")
    print("Different speaker has higher similarity.")

print("----------------------------")