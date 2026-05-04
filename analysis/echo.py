

import numpy as np

class EchoDetector:
    def __init__(self, frame_sizes = [1024, 2048, 4096, 8192, 16384, 32768], d_min = 40, d_max = 400):
        self.frame_sizes = frame_sizes if isinstance(frame_sizes, list) else [frame_sizes]
        self.d_min = d_min
        self.d_max = d_max

    def cepstrum(self, x):
        x = x - np.mean(x)
        spectrum = np.fft.fft(x)
        log_spectrum = np.log(np.abs(spectrum) + 1e-12)
        return np.real(np.fft.ifft(log_spectrum))

    def analyze_frame_size(self, signal, frame_size):
        n_frames = len(signal) // frame_size

        if n_frames < 8:
            return None

        signal_cut = signal[:n_frames * frame_size]
        frames = signal_cut.reshape(n_frames, frame_size)

        peak_positions = []
        peak_values = []

        for frame in frames:
            cep = self.cepstrum(frame)

            if self.d_max >= len(cep):
                continue

            region = np.abs(cep[self.d_min:self.d_max])

            if len(region) < 10:
                continue

            region = (region - np.mean(region)) / (np.std(region) + 1e-8)

            peak_idx = np.argmax(region)
            peak_val = region[peak_idx]

            peak_positions.append(peak_idx + self.d_min)
            peak_values.append(peak_val)

        # if len(peak_values) < 6:
        #     return None

        peak_positions = np.array(peak_positions)
        peak_values = np.array(peak_values)
        # пики
        pos_std = np.std(peak_positions)
        pos_mean = np.mean(peak_positions)
        # стабильность амплитуды
        val_mean = np.mean(peak_values)
        val_std = np.std(peak_values)
        stability = val_mean / (val_std + 1e-6)

        strong_ratio = np.mean(peak_values > 1.5)
        quality = 0
        
        if pos_std < 30:
            quality += 0.4
        elif pos_std < 50:
            quality += 0.2
            
        if stability > 2.0:
            quality += 0.3
        elif stability > 1.2:
            quality += 0.2
            
        if strong_ratio > 0.3:
            quality += 0.2
        elif strong_ratio > 0.15:
            quality += 0.2
            
        # ср амплитуда пика
        if val_mean > 0.5:
            quality += 0.2

        return {
            'frame_size': frame_size,
            'quality': quality,
            'pos_std': pos_std,
            'pos_mean': pos_mean,
            'stability': stability,
            'strong_ratio': strong_ratio,
            'val_mean': val_mean,
            'n_frames': n_frames
        }

    def analyze(self, signal):
        signal = np.asarray(signal)
        if signal.ndim > 1:
            signal = signal[:, 0]

        results = []
        
        # для каждого фрейма
        for frame_size in self.frame_sizes:
            # if len(signal) < frame_size * 8:
            #     print(f"frame_size={frame_size}: сигнал слишком короткий")
            #     continue
                
            result = self.analyze_frame_size(signal, frame_size)
            if result is not None:
                results.append(result)

        if not results:
            return 0

        best = max(results, key = lambda x: x['quality'])
        is_stego = best['quality'] >= 0.8
        
        if is_stego:
            if best['pos_std'] > 60:
                is_stego = False
            if best['stability'] < 1.0:
                is_stego = False
            if best['n_frames'] < 6:
                is_stego = False

        return 1 if is_stego else 0
