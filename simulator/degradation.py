import numpy as np
import librosa
import soundfile as sf

def add_jitter_shimmer(audio, sr, jitter_factor=0.01, shimmer_factor=0.05):
    """
    Simulates cycle-to-cycle pitch (jitter) and amplitude (shimmer) perturbation,
    characteristic of dysarthric/impaired speech.
    
    jitter_factor: proportion of random pitch variation (0.01 = mild, 0.05 = severe)
    shimmer_factor: proportion of random amplitude variation (0.05 = mild, 0.15 = severe)
    """
    # Find pitch periods using zero-crossing approximation
    # Simple approach: apply frame-wise random time-stretching (jitter proxy)
    # and frame-wise random amplitude scaling (shimmer)
    
    frame_length = int(sr * 0.01)  # 10ms frames, roughly one pitch period at typical F0
    num_frames = len(audio) // frame_length
    
    output = np.copy(audio).astype(np.float64)
    
    for i in range(num_frames):
        start = i * frame_length
        end = start + frame_length
        if end > len(output):
            break
        
        # Shimmer: random amplitude scaling per frame
        amp_scale = 1.0 + np.random.uniform(-shimmer_factor, shimmer_factor)
        output[start:end] *= amp_scale
        
        # Jitter: random small time-shift per frame (approximated via micro-resampling)
        jitter_shift = int(frame_length * np.random.uniform(-jitter_factor, jitter_factor))
        if jitter_shift != 0 and start + jitter_shift >= 0 and end + jitter_shift <= len(audio):
            output[start:end] = audio[start+jitter_shift:end+jitter_shift]
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(output))
    if max_val > 1.0:
        output = output / max_val
    
    return output.astype(np.float32)


if __name__ == "__main__":
    # Quick test on a sample file
    test_file = "data/processed/mfa_corpus/female/female_0000.wav"
    audio, sr = librosa.load(test_file, sr=None)
    
    degraded = add_jitter_shimmer(audio, sr, jitter_factor=0.02, shimmer_factor=0.08)
    
    sf.write("results/test_jitter_shimmer.wav", degraded, sr)
    print(f"Original duration: {len(audio)/sr:.2f}s, Degraded saved to results/test_jitter_shimmer.wav")
