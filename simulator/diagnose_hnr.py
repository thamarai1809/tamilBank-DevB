import parselmouth
from parselmouth.praat import call
import librosa
import soundfile as sf
import numpy as np
import sys
sys.path.insert(0, '.')
from simulator.degradation import (
    add_jitter_shimmer, add_breathiness, add_formant_smoothing,
    add_reduced_f0_and_slowing, add_room_and_bandlimit
)

def get_hnr(audio, sr):
    sf.write("results/temp_test.wav", audio, sr)
    snd = parselmouth.Sound("results/temp_test.wav")
    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    return call(harmonicity, "Get mean", 0, 0)

audio, sr = librosa.load("data/processed/mfa_corpus/female/female_0000.wav", sr=None)

print(f"Original HNR: {get_hnr(audio, sr):.4f}")

step1 = add_jitter_shimmer(audio, sr, jitter_factor=0.008, shimmer_factor=0.03)
print(f"After jitter/shimmer only: {get_hnr(step1, sr):.4f}")

step2 = add_breathiness(audio, sr, noise_level=0.005, tilt_strength=0.08)
print(f"After breathiness only: {get_hnr(step2, sr):.4f}")

step3 = add_formant_smoothing(audio, sr, smoothing_strength=0.1)
print(f"After formant smoothing only: {get_hnr(step3, sr):.4f}")

step4 = add_reduced_f0_and_slowing(audio, sr, f0_compression=0.15, rate_factor=0.97)
print(f"After F0/rate reduction only: {get_hnr(step4, sr):.4f}")

# Test chaining like apply_severity does (mild level)
chained = add_jitter_shimmer(audio, sr, jitter_factor=0.008, shimmer_factor=0.03)
chained = add_breathiness(chained, sr, noise_level=0.005, tilt_strength=0.08)
chained = add_formant_smoothing(chained, sr, smoothing_strength=0.1)
chained = add_reduced_f0_and_slowing(chained, sr, f0_compression=0.15, rate_factor=0.97)
print(f"After all 4 chained (mild): {get_hnr(chained, sr):.4f}")

# Test without formant smoothing in the chain
chained2 = add_jitter_shimmer(audio, sr, jitter_factor=0.008, shimmer_factor=0.03)
chained2 = add_breathiness(chained2, sr, noise_level=0.005, tilt_strength=0.08)
chained2 = add_reduced_f0_and_slowing(chained2, sr, f0_compression=0.15, rate_factor=0.97)
print(f"After 3 (skip formant): {get_hnr(chained2, sr):.4f}")

# Test without F0/rate in the chain
chained3 = add_jitter_shimmer(audio, sr, jitter_factor=0.008, shimmer_factor=0.03)
chained3 = add_breathiness(chained3, sr, noise_level=0.005, tilt_strength=0.08)
chained3 = add_formant_smoothing(chained3, sr, smoothing_strength=0.1)
print(f"After 3 (skip F0/rate): {get_hnr(chained3, sr):.4f}")
