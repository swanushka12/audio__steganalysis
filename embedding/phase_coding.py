# embedding/phase_coding.py

import numpy as np
import random
import math
from scipy.fft import rfft, irfft

PHASE_STEP = np.pi / 2
MIN_SEGMENT_SIZE = 256
MAX_SEGMENT_SIZE = 4096


class PhaseCoding:    
    def __init__(self):
        pass
    
    def segment_size(self, message_bits):
        v = int(math.ceil(math.log2(message_bits) + 1))
        segment_size = 2 ** (v + 1)
        segment_size = max(MIN_SEGMENT_SIZE, min(MAX_SEGMENT_SIZE, segment_size))
        return segment_size
    
    def first_seg_capacity(self, segment_size):
        return (segment_size // 2) - 1
    
    def generate_random_bits(self, num_bits, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        return [random.randint(0, 1) for _ in range(num_bits)]
    
    def embed(self, audio_samples, secret_bits, verbose = False):
        samples = audio_samples.copy()
        n_samples = len(samples)
        message_bits = len(secret_bits)
        
        segment_size = self.segment_size(message_bits)
        segment_half = segment_size // 2
        
        first_capacity = self.first_seg_capacity(segment_size)
        
        if message_bits > first_capacity:
            secret_bits = secret_bits[:first_capacity]
            message_bits = first_capacity
        
        # разбиение на сегменты
        n_segments = int(math.ceil(n_samples / segment_size))
        padded_length = n_segments * segment_size
        
        if padded_length > n_samples:
            samples_padded = np.pad(samples, (0, padded_length - n_samples), mode='constant')
        else:
            samples_padded = samples[:padded_length]
        
        segments = samples_padded.reshape(n_segments, segment_size)
        
        # FFT
        fft_segments = rfft(segments, axis=1)
        magnitudes = np.abs(fft_segments)
        phases = np.angle(fft_segments)
        original_first_phases = phases[0].copy()
        
        # 1й сегмент
        phases_modified = phases.copy()
        
        for k in range(1, message_bits + 1):
            pos = segment_half + 1 - k
            if 0 < pos < len(phases_modified[0]) - 1:
                # смещение!
                if secret_bits[k-1] == 1:
                    # сдвигаем фазу к -π/2
                    target = -PHASE_STEP
                    current = phases_modified[0][pos]
                    diff = target - current
                    diff = np.mod(diff + np.pi, 2*np.pi) - np.pi
                    phases_modified[0][pos] = current + diff
                else:
                    # сдвигаем фазу к +π/2
                    target = PHASE_STEP
                    current = phases_modified[0][pos]
                    diff = target - current
                    diff = np.mod(diff + np.pi, 2*np.pi) - np.pi
                    phases_modified[0][pos] = current + diff
        
        phase_correction = phases_modified[0] - original_first_phases
        
        for n in range(1, n_segments):
            phases_modified[n] = phases[n] + phase_correction
        
        # плавный переход между сегментами
        window = np.hanning(segment_size)
        
        # обратное FFT
        stego_fft = magnitudes * np.exp(1j * phases_modified)
        stego_segments = irfft(stego_fft, n=segment_size, axis=1)
        
        # окно для сглаживания
        stego_segments_windowed = stego_segments * window
        
        # объединение сегментов с перекрытием 
        overlap = segment_size // 2
        stego_padded = np.zeros(padded_length + overlap)
        
        for n in range(n_segments):
            start = n * segment_size
            end = start + segment_size
            stego_padded[start:end] += stego_segments_windowed[n]
        
        stego_audio = stego_padded[:n_samples]
        
        # нормализация
        max_val = np.max(np.abs(stego_audio))
        if max_val > 0.95:
            stego_audio = stego_audio / max_val * 0.95
        
        total_capacity = n_segments * (segment_half - 1)
        
        if verbose:
            print(f"Встроено {message_bits} бит, сегмент={segment_size}, "
                  f"ёмкость={total_capacity} бит")
        
        return stego_audio, segment_size, total_capacity, first_capacity
    
    
    def embed_test_message(self, audio_data, embed_fraction = 0, seed = None):
        result = {
            'audio_data': audio_data.copy(),
            'was_embedded': False,
            'embedded_bits': 0,
            'segment_size_used': 0,
            'total_capacity_actual': 0,
            'first_capacity_actual': 0
        }
        
        if embed_fraction <= 0:
            return result
        
        try:
            # размера сегмента
            approx_segment_size = 256
            while approx_segment_size * 2 < len(audio_data) and approx_segment_size < 4096:
                approx_segment_size *= 2
            
            first_capacity = self.first_seg_capacity(approx_segment_size)
            target_bits = max(8, int(first_capacity * embed_fraction))
            target_bits = min(target_bits, first_capacity)
            
            rand_bits = self.generate_random_bits(target_bits, seed)
            
            stego_audio, segment_size, total_capacity, first_capacity_actual = self.embed(audio_data, rand_bits)
            
            result['audio_data'] = stego_audio
            result['was_embedded'] = True
            result['embedded_bits'] = target_bits
            result['segment_size_used'] = segment_size
            result['total_capacity_actual'] = total_capacity
            result['first_capacity_actual'] = first_capacity_actual
            
        except Exception as e:
            print(f"Ошибка встраивания: {e}")
            import traceback
            traceback.print_exc()
        
        return result
