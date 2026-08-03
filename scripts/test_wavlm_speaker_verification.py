from datasets import load_dataset, Audio
import soundfile as sf
import io
import torch
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity

model_name = "microsoft/wavlm-base-plus-sv"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
wavlm_model = WavLMModel.from_pretrained(model_name)
wavlm_model.eval()

def get_wavlm_embedding(audio_array, sample_rate):
    if sample_rate != 16000:
        import torchaudio
        signal = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)
        signal = torchaudio.functional.resample(signal, sample_rate, 16000)
        audio_array = signal.squeeze().numpy()
    inputs = feature_extractor(audio_array, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = wavlm_model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

def cosine_sim(emb1, emb2):
    return cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]

dataset = load_dataset('SPRINGLab/IndicTTS_Tamil', split='train', streaming=True)
dataset = dataset.cast_column('audio', Audio(decode=False))

female_samples = []
male_samples = []

for i, sample in enumerate(dataset):
    if sample['gender'] == 0 and len(female_samples) < 2:
        female_samples.append(sample)
    elif sample['gender'] == 1 and len(male_samples) < 1:
        male_samples.append(sample)
    if len(female_samples) >= 2 and len(male_samples) >= 1:
        break
    if i >= 8000:
        break

def decode(sample):
    audio_bytes = sample['audio']['bytes']
    data, sr = sf.read(io.BytesIO(audio_bytes))
    return data, sr

data1, sr1 = decode(female_samples[0])
data2, sr2 = decode(female_samples[1])
data3, sr3 = decode(male_samples[0])

emb1 = get_wavlm_embedding(data1, sr1)
emb2 = get_wavlm_embedding(data2, sr2)
emb3 = get_wavlm_embedding(data3, sr3)

print(f"[WavLM] Same speaker (female-female) similarity: {cosine_sim(emb1, emb2):.4f}")
print(f"[WavLM] Different speaker (female-male) similarity: {cosine_sim(emb1, emb3):.4f}")
