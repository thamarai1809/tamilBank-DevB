import epitran
import os
import glob
import panphon

CORPUS_DIR = "data/processed/mfa_corpus"
DICT_OUTPUT_PATH = "aligner/tamil_dictionary.txt"

epi = epitran.Epitran('tam-Taml')
ft = panphon.FeatureTable()

unique_words = set()

lab_files = glob.glob(os.path.join(CORPUS_DIR, "**", "*.lab"), recursive=True)
print(f"Found {len(lab_files)} transcript files")

for lab_file in lab_files:
    with open(lab_file, "r", encoding="utf-8") as f:
        text = f.read().strip()
        words = text.split()
        unique_words.update(words)

print(f"Found {len(unique_words)} unique words")

os.makedirs(os.path.dirname(DICT_OUTPUT_PATH), exist_ok=True)

with open(DICT_OUTPUT_PATH, "w", encoding="utf-8") as f:
    for word in sorted(unique_words):
        try:
            phones_str  = epi.transliterate(word)
            segmented = ft.segs_safe(phones_str)
            phone_seq = " ".join(list(segmented))
            f.write(f"{word}\t{phone_seq}\n")
        except Exception as e:
            print(f"Failed on word: {word} -> {e}")

print(f"Dictionary written to {DICT_OUTPUT_PATH}")
