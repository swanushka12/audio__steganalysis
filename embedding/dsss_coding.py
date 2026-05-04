# embedding/dsss_coding.py

import numpy as np
import random
import librosa

class DSSSCoding:
    def __init__(self, frame_size = 4096, alpha = 1.0):  
        self.frame_size = frame_size
        self.alpha = alpha

    def pn_sequence(self, L, seed=42):
        rng = np.random.RandomState(seed)
        return rng.choice([-1, 1], size=L)

    def embed(self, signal, bits, seed=42):
        signal = np.asarray(signal, dtype=np.float32)

        if signal.ndim > 1:
            signal = signal[:, 0]

        max_val = np.max(np.abs(signal))
        if max_val > 0:
            signal_norm = signal / max_val
        else:
            signal_norm = signal

        n_frames = len(signal_norm) // self.frame_size
        bits = np.asarray(bits[:n_frames])

        if len(bits) < n_frames:
            bits = np.pad(bits, (0, n_frames - len(bits)))

        L = self.frame_size
        pn = self.pn_sequence(L, seed)

        out = signal_norm.copy()

        for i in range(n_frames):
            start = i * L
            end = start + L

            frame = signal_norm[start:end]
            
            if bits[i] == 0:
                out[start:end] = frame + pn * self.alpha
            else:
                out[start:end] = frame - pn * self.alpha

        out = out * max_val
        return out

    def embed_test_message(self, audio_data, fraction=0.1, seed=42):
        if isinstance(audio_data, str):
            audio_data, _ = librosa.load(audio_data, sr=None, mono=True)

        n_frames = len(audio_data) // self.frame_size
        max_capacity = n_frames

        target = max(8, int(max_capacity * fraction))

        rng = random.Random(seed)
        bits = [rng.randint(0, 1) for _ in range(target)]

        stego = self.embed(audio_data, bits, seed=seed)

        return {
            "audio_data": stego,
            "was_embedded": True,
            "embedded_bits": len(bits),
            "max_capacity": max_capacity,
            "frame_size_used": self.frame_size
        }