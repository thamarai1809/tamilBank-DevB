import epitran
import wandb

wandb.init(project="tamilbank", name="g2p-batch-test")

epi = epitran.Epitran('tam-Taml')

# Placeholder sentences - replace with Dev A's actual literary/colloquial pool once ready
test_sentences = [
    ("literary", "வணக்கம்"),
    ("literary", "நான் நலமாக இருக்கிறேன்"),
    ("literary", "இது ஒரு சோதனை வாக்கியம்"),
    ("literary", "தமிழ் மொழி மிகவும் பழமையானது"),
    ("literary", "அவர் நாளை வருவார்"),
    ("colloquial", "என்ன பண்ற?"),
    ("colloquial", "நல்லா இருக்கியா?"),
    ("colloquial", "இங்க வா"),
    ("colloquial", "எனக்கு பசிக்குது"),
    ("colloquial", "அவன் எங்க போனான்?"),
    # add up to 20-30 total, mixing literary + colloquial
]

results = []
for register, sentence in test_sentences:
    try:
        phones = epi.transliterate(sentence)
        print(f"[{register}] {sentence} -> {phones}")
        results.append({"register": register, "sentence": sentence, "phones": phones, "status": "ok"})
    except Exception as e:
        print(f"[{register}] {sentence} -> ERROR: {e}")
        results.append({"register": register, "sentence": sentence, "phones": None, "status": "error"})

success_count = sum(1 for r in results if r["status"] == "ok")
wandb.log({"total_sentences": len(results), "successful": success_count, "failed": len(results) - success_count})

wandb.finish()
