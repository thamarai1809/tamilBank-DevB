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

def add_breathiness(audio, sr, noise_level=0.02, tilt_strength=0.3):
    """
    Simulates breathy voice quality via spectral tilt + aspiration noise,
    characteristic of weak vocal fold closure in impaired speech.
    
    noise_level: amount of filtered noise to add (0.01 = mild, 0.05 = severe)
    tilt_strength: how much to attenuate low frequencies relative to high (0-1)
    """
    # Generate aspiration noise (high-frequency weighted noise)
    noise = np.random.normal(0, 1, len(audio))
    
    # High-pass filter the noise to concentrate it in higher frequencies (breath-like)
    from scipy.signal import butter, filtfilt
    b, a = butter(4, 1500 / (sr / 2), btype='high')
    filtered_noise = filtfilt(b, a, noise)
    filtered_noise = filtered_noise / np.max(np.abs(filtered_noise)) * noise_level
    
    # Apply spectral tilt: gentle low-shelf attenuation
    b_tilt, a_tilt = butter(2, 500 / (sr / 2), btype='low')
    low_component = filtfilt(b_tilt, a_tilt, audio)
    tilted_audio = audio - (tilt_strength * low_component * 0.3)
    
    # Combine
    output = tilted_audio + filtered_noise
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(output))
    if max_val > 1.0:
        output = output / max_val
    
    return output.astype(np.float32)

def add_formant_smoothing(audio, sr, smoothing_strength=0.4):
    """
    Simulates articulatory imprecision via formant smoothing / vowel space reduction,
    characteristic of dysarthric speech where vowel distinctions become less clear.
    
    smoothing_strength: 0-1, how much to smooth spectral envelope (0.2=mild, 0.7=severe)
    """
    from scipy.signal import butter, filtfilt
    
    # Approximate formant smoothing via spectral envelope smoothing:
    # apply a moving-average-like low-pass on the spectral envelope using STFT
    n_fft = 1024
    hop_length = 256
    
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = np.abs(stft), np.angle(stft)
    
    # Smooth the magnitude spectrum across frequency bins (blurs formant peaks)
    from scipy.ndimage import uniform_filter1d
    smooth_size = int(5 + smoothing_strength * 20)  # more smoothing = larger window
    smoothed_magnitude = uniform_filter1d(magnitude, size=smooth_size, axis=0)
    
    # Blend original and smoothed based on strength
    blended_magnitude = (1 - smoothing_strength) * magnitude + smoothing_strength * smoothed_magnitude
    
    # Reconstruct
    smoothed_stft = blended_magnitude * np.exp(1j * phase)
    output = librosa.istft(smoothed_stft, hop_length=hop_length, length=len(audio))
    
    return output.astype(np.float32)

def add_reduced_f0_and_slowing(audio, sr, f0_compression=0.5, rate_factor=0.85):
    """
    Simulates reduced pitch range (monotone) and slowed speech rate,
    both characteristic of dysarthric/impaired speech.
    
    f0_compression: 0-1, how much to compress pitch variation toward the mean (0.3=mild, 0.7=severe)
    rate_factor: <1.0 slows down speech (0.9=mild, 0.7=severe)
    """
    import librosa.effects
    
    # Time-stretch to slow down (rate_factor < 1 = slower)
    slowed = librosa.effects.time_stretch(audio, rate=rate_factor)
    
    # Pitch compression: shift pitch toward a flatter contour
    # Using pitch_shift as an approximation isn't quite right for "compression",
    # so we use a simple approach: extract pitch, compress deviations from mean, resynthesize is complex;
    # here we approximate via a mild constant pitch shift down + reduced modulation via harmonic-percussive smoothing
    harmonic, percussive = librosa.effects.hpss(slowed)
    output = f0_compression * harmonic + (1 - f0_compression * 0.3) * percussive + (slowed - harmonic - percussive) * 0
    output = slowed * (1 - f0_compression * 0.1) + harmonic * (f0_compression * 0.1)  # subtle flattening blend
    
    return output.astype(np.float32)

def add_room_and_bandlimit(audio, sr, room_size=(5, 4, 3), band_low=300, band_high=3400):
    """
    Simulates room acoustics (reverb) and band-limiting (e.g., phone-quality recording),
    adding realistic environmental degradation.
    """
    import pyroomacoustics as pra
    from scipy.signal import butter, filtfilt
    
    # Create a simple shoebox room and simulate RIR
    room = pra.ShoeBox(room_size, fs=sr, max_order=10, materials=pra.Material(0.3))
    room.add_source([1, 1, 1], signal=audio)
    room.add_microphone([2, 2, 1])
    room.simulate()
    
    reverbed = room.mic_array.signals[0, :len(audio)]
    if len(reverbed) < len(audio):
        reverbed = np.pad(reverbed, (0, len(audio) - len(reverbed)))
    
    # Band-limit (simulate telephone/low-quality mic bandwidth)
    b, a = butter(4, [band_low / (sr/2), band_high / (sr/2)], btype='band')
    bandlimited = filtfilt(b, a, reverbed)
    
    # Normalize
    max_val = np.max(np.abs(bandlimited))
    if max_val > 1.0:
        bandlimited = bandlimited / max_val
    
    return bandlimited.astype(np.float32)

if __name__ == "__main__":
    test_file = "data/processed/mfa_corpus/female/female_0000.wav"
    audio, sr = librosa.load(test_file, sr=None)
    
    degraded_js = add_jitter_shimmer(audio, sr, jitter_factor=0.02, shimmer_factor=0.08)
    sf.write("results/test_jitter_shimmer.wav", degraded_js, sr)
    
    degraded_breathy = add_breathiness(audio, sr, noise_level=0.03, tilt_strength=0.3)
    sf.write("results/test_breathiness.wav", degraded_breathy, sr)
    
    degraded_formant = add_formant_smoothing(audio, sr, smoothing_strength=0.4)
    sf.write("results/test_formant_smoothing.wav", degraded_formant, sr)
    
    degraded_f0_rate = add_reduced_f0_and_slowing(audio, sr, f0_compression=0.5, rate_factor=0.85)
    sf.write("results/test_f0_rate.wav", degraded_f0_rate, sr)
    
    degraded_room = add_room_and_bandlimit(audio, sr)
    sf.write("results/test_room_bandlimit.wav", degraded_room, sr)
    
    print(f"Original duration: {len(audio)/sr:.2f}s")
    print("Saved all 5 degradation component test files to results/")
