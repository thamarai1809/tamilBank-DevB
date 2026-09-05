import argparse
import glob
import random
import numpy as np
import parselmouth
from parselmouth.praat import call
import soundfile as sf
import librosa
import simulator.degradation as degradation


def extract_acoustic_measures(wav_path):
    """Extract HNR, jitter, shimmer from a wav file using Praat via parselmouth."""
    snd = parselmouth.Sound(wav_path)
    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)
    jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    return {
        "hnr": hnr if not np.isnan(hnr) else None,
        "jitter": jitter if not np.isnan(jitter) else None,
        "shimmer": shimmer if not np.isnan(shimmer) else None,
    }


def average_measures(file_list, label, n_sample=100):
    sample = random.sample(file_list, min(n_sample, len(file_list)))
    results = {"hnr": [], "jitter": [], "shimmer": []}
    for f in sample:
        try:
            m = extract_acoustic_measures(f)
            for k in results:
                if m[k] is not None:
                    results[k].append(m[k])
        except Exception as e:
            print(f"  Skipped {f}: {e}")
    print(f"\n--- {label} (n={len(sample)}) ---")
    for k, vals in results.items():
        if vals:
            print(f"{k}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/real_corpus")
    ap.add_argument("--alignments", default="data/processed/mfa_alignments_real")
    args = ap.parse_args()

    random.seed(42)

    torgo_dysarthric = glob.glob("data/raw/torgo/torgo_data/dysarthria_*/*.wav")
    print(f"Found {len(torgo_dysarthric)} real dysarthric TORGO files")

    torgo_healthy = glob.glob("data/raw/torgo/torgo_data/non_dysarthria_*/*.wav")
    print(f"Found {len(torgo_healthy)} real healthy TORGO files")

    print("\n=== ACOUSTIC MEASURE COMPARISON (TORGO baseline) ===")
    average_measures(torgo_dysarthric, "Real TORGO Dysarthric Speech")
    average_measures(torgo_healthy, "Real TORGO Healthy Speech")

    print("\n=== SIMULATOR SEVERITY PRESETS (5 seeds each) ===")
    test_audio, test_sr = librosa.load("data/processed/mfa_corpus/female/female_0000.wav", sr=None)

    for level in ["mild", "moderate", "severe", "profound"]:
        results = {"hnr": [], "jitter": [], "shimmer": []}
        for seed in range(5):
            degraded = degradation.apply_severity(test_audio, test_sr, level=level, seed=seed)
            sf.write(f"results/_tmp_{level}_{seed}.wav", degraded, test_sr)
            m = extract_acoustic_measures(f"results/_tmp_{level}_{seed}.wav")
            for k in results:
                if m[k] is not None:
                    results[k].append(m[k])
        print(f"\n--- Your Simulator ({level} preset, n={len(results['hnr'])} seeds) ---")
        for k, vals in results.items():
            if vals:
                print(f"{k}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}")

    real_tamil_files = sorted(glob.glob(f"{args.input}/*.wav"))
    print(f"\nFound {len(real_tamil_files)} real TamilBank corpus files in {args.input}")

    part_a_files = [f for f in real_tamil_files if f.lower().endswith("parta.wav")]
    part_b_files = [f for f in real_tamil_files if f.lower().endswith("partb.wav")]

    print("\n=== REAL TAMILBANK CORPUS vs TORGO-CALIBRATED SIMULATOR ===")
    average_measures(part_a_files, "Real TamilBank Speakers - Part A (formal)", n_sample=len(part_a_files))
    average_measures(part_b_files, "Real TamilBank Speakers - Part B (casual)", n_sample=len(part_b_files))
