from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy
import torchaudio
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

classifier = EncoderClassifier.from_hparams(
    source='speechbrain/spkrec-ecapa-voxceleb',
    savedir='pretrained_models/spkrec-ecapa-voxceleb',
    local_strategy=LocalStrategy.COPY
)

def get_embedding(wav_path):
    signal, fs = torchaudio.load(wav_path)
    if fs != 16000:
        signal = torchaudio.functional.resample(signal, fs, 16000)
    embedding = classifier.encode_batch(signal)
    return embedding.squeeze().detach().cpu().numpy()

def cosine_sim(emb1, emb2):
    return cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]

if __name__ == "__main__":
    # Replace these paths with actual audio files once you have some
    wav1 = "path/to/speaker1_utt1.wav"
    wav2 = "path/to/speaker1_utt2.wav"  # same speaker, different utterance
    wav3 = "path/to/speaker2_utt1.wav"  # different speaker

    emb1 = get_embedding(wav1)
    emb2 = get_embedding(wav2)
    emb3 = get_embedding(wav3)

    print(f"Same speaker similarity: {cosine_sim(emb1, emb2):.4f}")
    print(f"Different speaker similarity: {cosine_sim(emb1, emb3):.4f}")
