import parselmouth
from parselmouth.praat import call
import numpy as np
import glob
import random

def extract_acoustic_measures(wav_path):
    """Extract CPP, HNR, jitter, shimmer from a wav file using Praat via parselmouth."""
    snd = parselmouth.Sound(wav_path)
    
    # HNR (Harmonics-to-Noise Ratio)
    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    
    # Jitter and Shimmer (need a PointProcess first)
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
    random.seed(42)
    
    # Real dysarthric speech from TORGO
    torgo_dysarthric = glob.glob("data/raw/torgo/torgo_data/dysarthria_*/*.wav")
    print(f"Found {len(torgo_dysarthric)} real dysarthric TORGO files")
    
    # Real healthy speech from TORGO (as a baseline control)
    torgo_healthy = glob.glob("data/raw/torgo/torgo_data/non_dysarthria_*/*.wav")
    print(f"Found {len(torgo_healthy)} real healthy TORGO files")
    
    # Your simulator's degraded output (severe level)
    simulated_degraded = ["results/test_severity_severe.wav"] * 1  # single file, will expand below
    
    print("\n=== ACOUSTIC MEASURE COMPARISON ===")
    real_dysarthric_stats = average_measures(torgo_dysarthric, "Real TORGO Dysarthric Speech")
    real_healthy_stats = average_measures(torgo_healthy, "Real TORGO Healthy Speech")
    
    # Single-file measures for your simulated severe-degraded sample
    for level in ["mild", "moderate", "severe", "profound"]:
    	sim_measures = extract_acoustic_measures(f"results/test_severity_{level}.wav")
    	print(f"\n--- Your Simulator ({level} preset, n=1) ---")
    	for k, v in sim_measures.items():
            print(f"{k}: {v:.4f}" if v is not None else f"{k}: N/A")    
    orig_measures = extract_acoustic_measures("data/processed/mfa_corpus/female/female_0000.wav")
    print(f"\n--- Your Original Clean Audio (n=1) ---")
    for k, v in orig_measures.items():
        print(f"{k}: {v:.4f}" if v is not None else f"{k}: N/A")
