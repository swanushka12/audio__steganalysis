# analysis/phase.py

import numpy as np
import soundfile as sf
from scipy.fft import fft
from scipy.signal import get_window
import warnings
warnings.filterwarnings('ignore')

class PhaseDetector:
    def __init__(self, conc_threshold = 0.3, tolerance = 0.25):
        self.conc_threshold = conc_threshold
        self.tolerance = tolerance
        self.target_phase = np.pi / 2
        self.segment_sizes = [256, 512, 1024, 2048, 4096]  

    def analyze(self, audio_input):
        # загрузка аудио 
        if isinstance(audio_input, str):
            try:
                signal, sr = sf.read(audio_input)
                if signal.ndim > 1:
                    signal = np.mean(signal, axis = 1)
                filename = audio_input
            except:
                return 0
        else:
            signal = audio_input
            if signal.ndim > 1:
                signal = np.mean(signal, axis = 1)
            filename = "audio_array"
        
        results = []
        
        # каждый размер сегмента
        for segment_size in self.segment_sizes:
            if len(signal) < segment_size:
                print(f"{segment_size:<10} {'слишком короткий':<20}")
                continue
            
            # первый сегмент
            window = get_window("hann", segment_size)
            segment = signal[:segment_size] * window
            
            # FFT
            spectrum = fft(segment, n = segment_size)
            phases = np.angle(spectrum)
            
            # анализ средних частот 
            low_idx = segment_size // 4
            high_idx = 3 * segment_size // 4
            middle = phases[low_idx:high_idx]
            
            # концентрация фаз около ±π/2
            dist_to_pos = np.abs(middle - self.target_phase)
            dist_to_neg = np.abs(middle + self.target_phase)
            near_critical = np.sum((dist_to_pos < self.tolerance) | (dist_to_neg < self.tolerance))            
            pos_count = np.sum(dist_to_pos < self.tolerance)
            neg_count = np.sum(dist_to_neg < self.tolerance)
            
            ratio = near_critical / len(middle)
            results.append((segment_size, ratio, pos_count, neg_count))
        
        
        if not results:
            print(f"Файл слишком короткий для всех размеров сегментов")
            return 0
        
        max_result = max(results, key = lambda x: x[1])
        max_size, max_ratio, max_pos, max_neg = max_result

        return 1 if max_ratio > self.conc_threshold else 0
