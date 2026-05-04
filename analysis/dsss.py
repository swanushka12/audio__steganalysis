# analysis/dsss.py
import numpy as np
import librosa

class DSSSDetector:
    def __init__(self, frame_sizes=[1024, 2048, 4096, 8192, 16384], delays = (150, 200)):
        self.frame_sizes = frame_sizes if isinstance(frame_sizes, list) else [frame_sizes]
        self.delays = delays

    def normalize(self, x):
        x = x - np.mean(x)
        std = np.std(x)
        return x / (std + 1e-10) if std > 1e-10 else x

    def corr_score(self, x, delay):
        x = self.normalize(x)
        if len(x) <= delay + 10:
            return 0
        shifted = np.roll(x, delay)
        score = np.mean(x * shifted)
        return abs(score)

    def analyze_frame_size(self, signal, frame_size):
        n_frames = len(signal) // frame_size
        if n_frames < 5:
            return None, None

        signal = signal[:n_frames * frame_size]
        frames = signal.reshape(n_frames, frame_size)

        scores = []
        for f in frames:
            frame_scores = [self.corr_score(f, d) for d in self.delays]
            scores.append(max(frame_scores))

        scores = np.array(scores)
        return np.mean(scores), np.std(scores)

    def analyze(self, signal = None, filepath = None):
        if filepath is not None:
            signal, _ = librosa.load(filepath, sr = None, mono = True)
        elif signal is None:
            return 0

        signal = np.asarray(signal)
        if signal.ndim > 1:
            signal = signal[:, 0]

        best_score = 0
        best_mean = 0
        best_std = 0
        best_frame_size = None
        
        # все размеры фреймов
        for frame_size in self.frame_sizes:
            mean_score, std_score = self.analyze_frame_size(signal, frame_size)
            
            if mean_score is None:
                continue
            
            # высокая корреляция + низкая вариативность
            quality = mean_score / (std_score + 1e-10)
            
            if quality > best_score:
                best_score = quality
                best_mean = mean_score
                best_std = std_score
                best_frame_size = frame_size

        if best_frame_size is None:
            return 0

        MEAN_THRESHOLD = 0.005
        STD_THRESHOLD = 0.015
        
        is_stego = (best_mean > MEAN_THRESHOLD) and (best_std < STD_THRESHOLD)

        return 1 if is_stego else 0