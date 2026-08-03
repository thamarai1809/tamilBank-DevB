from datasets import load_dataset, Audio
import soundfile as sf
import io
import numpy as np
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy
import torch
from sklearn.metrics.pairwise import cosine_similarity



# Load classifier
classifier = EncoderClassifier.from_hparams(
    source='speechbrain/spkrec-ecapa-voxceleb',
    savedir='pretrained_models/spkrec-ecapa-voxceleb',
    local_strategy=LocalStrategy.COPY
)

def get_embedding_from_array(audio_array, sample_rate):
    signal = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)
    if sample_rate != 16000:
        import torchaudio
        signal = torchaudio.functional.resample(signal, sample_rate, 16000)
    embedding = classifier.encode_batch(signal)
    return embedding.squeeze().detach().cpu().numpy()

def cosine_sim(emb1, emb2):
    return cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]

# Load dataset, keep audio undecoded so we can decode manually
dataset = load_dataset('SPRINGLab/IndicTTS_Tamil', split='train', streaming=True)
dataset = dataset.cast_column('audio', Audio(decode=False))

male_samples = []
female_samples = []

for i, sample in enumerate(dataset):
    if sample['gender'] == 0 and len(female_samples) < 2:
        female_samples.append(sample)
    elif sample['gender'] == 1 and len(male_samples) < 1:
        male_samples.append(sample)
    if len(female_samples) >= 2 and len(male_samples) >= 1:
        break
    if i >= 8000:  # safety cap so it doesn't scan forever
        break

print(f"Collected {len(female_samples)} female samples, {len(male_samples)} male samples")

def decode(sample):
    audio_bytes = sample['audio']['bytes']
    data, sr = sf.read(io.BytesIO(audio_bytes))
    return data, sr

data1, sr1 = decode(female_samples[0])
data2, sr2 = decode(female_samples[1])

emb1 = get_embedding_from_array(data1, sr1)
emb2 = get_embedding_from_array(data2, sr2)
print(f"Same speaker (female-female) similarity: {cosine_sim(emb1, emb2):.4f}")

if male_samples:
    data3, sr3 = decode(male_samples[0])
    emb3 = get_embedding_from_array(data3, sr3)
    print(f"Different speaker (female-male) similarity: {cosine_sim(emb1, emb3):.4f}")
else:
    print("No male samples found within scan limit — try increasing the cap")
