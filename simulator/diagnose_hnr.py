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
def get_shimmer(audio, sr):
    sf.write("results/temp_test.wav", audio, sr)
    snd = parselmouth.Sound("results/temp_test.wav")
    point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)
    return call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
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
step5 = add_room_and_bandlimit(chained, sr)
print(f"After room/bandlimit added on top of mild chain: {get_hnr(step5, sr):.4f}")

room_only = add_room_and_bandlimit(audio, sr)
print(f"Room/bandlimit alone on clean audio: {get_hnr(room_only, sr):.4f}")
print("\n--- Shimmer breakdown (mild settings) ---")
print(f"Original shimmer: {get_shimmer(audio, sr):.4f}")

s1 = add_jitter_shimmer(audio, sr, jitter_factor=0.008, shimmer_factor=0.025)
print(f"After jitter/shimmer only: {get_shimmer(s1, sr):.4f}")

s2 = add_breathiness(s1, sr, noise_level=0.005, tilt_strength=0.08)
print(f"After + breathiness: {get_shimmer(s2, sr):.4f}")

s3 = add_formant_smoothing(s2, sr, smoothing_strength=0.1)
print(f"After + formant smoothing: {get_shimmer(s3, sr):.4f}")

s5 = add_room_and_bandlimit(s3, sr, absorption=0.75, band_low=50, band_high=7900)
print(f"After + room/bandlimit (mild settings): {get_shimmer(s5, sr):.4f}")

print("\n--- HNR breakdown (same chain, mild settings) ---")
print(f"Original HNR: {get_hnr(audio, sr):.4f}")
print(f"After jitter/shimmer only: {get_hnr(s1, sr):.4f}")
print(f"After + breathiness: {get_hnr(s2, sr):.4f}")
print(f"After + formant smoothing: {get_hnr(s3, sr):.4f}")
print(f"After + room/bandlimit (mild settings): {get_hnr(s5, sr):.4f}")
