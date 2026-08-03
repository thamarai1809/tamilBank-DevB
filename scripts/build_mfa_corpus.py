from datasets import load_dataset, Audio
import soundfile as sf
import io
import os

OUTPUT_DIR = "data/processed/mfa_corpus"
NUM_SAMPLES_PER_SPEAKER = 20  # start small for a pilot validation run

os.makedirs(OUTPUT_DIR, exist_ok=True)

dataset = load_dataset('SPRINGLab/IndicTTS_Tamil', split='train', streaming=True)
dataset = dataset.cast_column('audio', Audio(decode=False))

gender_map = {0: "female", 1: "male"}
counts = {"female": 0, "male": 0}

for i, sample in enumerate(dataset):
    gender = gender_map[sample['gender']]
    if counts[gender] >= NUM_SAMPLES_PER_SPEAKER:
        if all(c >= NUM_SAMPLES_PER_SPEAKER for c in counts.values()):
            break
        continue

    speaker_dir = os.path.join(OUTPUT_DIR, gender)
    os.makedirs(speaker_dir, exist_ok=True)

    utt_id = f"{gender}_{counts[gender]:04d}"
    wav_path = os.path.join(speaker_dir, f"{utt_id}.wav")
    lab_path = os.path.join(speaker_dir, f"{utt_id}.lab")

    # decode and save audio
    audio_bytes = sample['audio']['bytes']
    data, sr = sf.read(io.BytesIO(audio_bytes))
    sf.write(wav_path, data, sr)

    # save transcript
    with open(lab_path, "w", encoding="utf-8") as f:
        f.write(sample['text'].strip())

    counts[gender] += 1

    if i >= 20000:  # safety cap
        break

print(f"Done. Extracted: {counts}")
