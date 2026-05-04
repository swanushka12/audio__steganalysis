
import numpy as np
import random
import librosa

class EchoCoding:
    def __init__(self, d0 = 150, d1 = 200, alpha = 0.5, frame_size = 2048):
        self.d0 = d0
        self.d1 = d1
        self.alpha = alpha
        self.frame_size = frame_size

    def _kernel(self, delay):
        k = np.zeros(delay + 1)
        k[-1] = self.alpha
        return k
    
    def embed(self, signal, bits):
        signal = signal.copy()
        k0 = self._kernel(self.d0)
        k1 = self._kernel(self.d1)

        # эхо
        echo0 = np.convolve(signal, k0, mode = "same")
        echo1 = np.convolve(signal, k1, mode = "same")

        # количество фреймов
        n_frames = len(signal) // self.frame_size
        total_frames = n_frames
        
        bits = bits[:total_frames]
        n_bits = len(bits)
        mix = np.zeros(len(signal))
        
        for i in range(n_bits): 
            start = i * self.frame_size
            end = min(start + self.frame_size, len(signal))
            mix[start:end] = bits[i]

        embed_len = n_bits * self.frame_size
        out = signal[:embed_len] + \
            echo0[:embed_len] * (1 - mix[:embed_len]) + \
            echo1[:embed_len] * mix[:embed_len]

        if len(signal) > embed_len:
            out = np.concatenate([out, signal[embed_len:]])

        return out

    def embed_test_message(self, audio_data, fraction = 0.1, seed = None):
        if isinstance(audio_data, str):
            audio_data, _ = librosa.load(audio_data, sr = None, mono = True)

        n_frames = len(audio_data) // self.frame_size
        max_capacity = n_frames

        target = max(8, int(max_capacity * min(fraction, 1.0)))

        rng = random.Random(seed)
        bits = [rng.randint(0, 1) for _ in range(target)]

        stego = self.embed(audio_data, bits)

        return {
            "audio_data": stego,
            "was_embedded": True,
            "embedded_bits": len(bits),
            "max_capacity": max_capacity,
            "frame_size_used": self.frame_size
    }