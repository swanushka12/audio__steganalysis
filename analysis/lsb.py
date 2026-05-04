# analysis/lsb.py

import numpy as np
from scipy.stats import chisquare
import warnings
warnings.filterwarnings('ignore')

class LSBDetector:
    def __init__(self):
        # 1
        self.chi_strict = 0.01
        self.correlation_threshold = 0.49
        self.transition_threshold = 0.46
        # 2
        self.p_diff_threshold = 1.3
        self.entropy_diff_threshold = 0.04
        # 3
        self.block_size = 4096
        self.min_suspicious_ratio = 0.35
        self.block_diff_threshold = 0.08
        # 4
        self.chi_loose = 0.05
        self.entropy_threshold = 0.95

    def to_int16(self, audio):
        if audio.dtype in [np.float32, np.float64]:
            return np.asarray(audio * 32767, dtype = np.int16)
        return np.asarray(audio, dtype = np.int16)

    def extract_lsb(self, samples, bit=0):
        return np.asarray((samples >> bit) & 1, dtype = np.int32)

    def chi_square_test(self, bits):
        count_0 = np.sum(bits == 0)
        count_1 = np.sum(bits == 1)
        observed = np.array([count_0, count_1])
        expected = np.array([len(bits)/2, len(bits)/2])
        try:
            chi, p_value = chisquare(observed, expected)
        except:
            return 1.0
        return p_value

    def entropy(self, bits):
        p0 = np.mean(bits == 0)
        p1 = np.mean(bits == 1)
        eps = 1e-10
        entropy = 0
        if p0 > 0:
            entropy -= p0 * np.log2(p0 + eps)
        if p1 > 0:
            entropy -= p1 * np.log2(p1 + eps)
        return entropy

    def correlation(self, bits):
        if len(bits) < 2:
            return 0
        b1 = np.asarray(bits[:-1], dtype = np.float64)
        b2 = np.asarray(bits[1:], dtype = np.float64)
        std1, std2 = np.std(b1), np.std(b2)

        if std1 == 0 or std2 == 0:
            return 0
        corr = np.corrcoef(b1, b2)[0, 1]
        return corr if not np.isnan(corr) else 0

    def transition_ratio(self, bits):
        if len(bits) < 2:
            return 0
        transitions = np.sum(bits[:-1] != bits[1:])
        return transitions / (len(bits) - 1)

    def analyze(self, audio_data):
        samples = self.to_int16(audio_data)

        if len(samples) < 10000:
            print(f"LSB: слишком мало сэмплов: {len(samples)}")
            return 0

        lsb0 = self.extract_lsb(samples, 0)
        lsb1 = self.extract_lsb(samples, 1)
        lsb2 = self.extract_lsb(samples, 2)
        
        n_blocks = len(samples) // self.block_size
        
        p_value_0 = self.chi_square_test(lsb0)
        p_value_1 = self.chi_square_test(lsb1)
        p_value_2 = self.chi_square_test(lsb2)
        
        entropy_0 = self.entropy(lsb0)
        entropy_1 = self.entropy(lsb1)
        entropy_2 = self.entropy(lsb2)
        
        correlation_0 = self.correlation(lsb0)
        transitions_0 = self.transition_ratio(lsb0)
        
        # разница
        p_ratio_01 = p_value_0 / max(p_value_1, 1e-10)
        p_ratio_12 = p_value_1 / max(p_value_2, 1e-10)
        entropy_diff_01 = entropy_0 - entropy_1
        entropy_diff_12 = entropy_1 - entropy_2
        
        if p_value_0 < self.chi_strict:
            if correlation_0 > self.correlation_threshold:
                return 0
            
            if transitions_0 < self.transition_threshold:
                return 0
        
        lsb0_very_random = (p_value_0 > 0.1) and (entropy_0 > 0.98)
        lsb1_very_structured = (p_value_1 < 0.001)
        
        if lsb0_very_random and lsb1_very_structured:
            return 1
        
        suspicious_blocks_0 = 0
        suspicious_blocks_1 = 0
        
        for i in range(n_blocks):
            start = i * self.block_size
            end = start + self.block_size
            
            p0 = self.chi_square_test(lsb0[start:end])
            p1 = self.chi_square_test(lsb1[start:end])
            
            if p0 > 0.1:
                suspicious_blocks_0 += 1
            if p1 > 0.1:
                suspicious_blocks_1 += 1
        
        ratio_0 = suspicious_blocks_0 / n_blocks if n_blocks > 0 else 0
        ratio_1 = suspicious_blocks_1 / n_blocks if n_blocks > 0 else 0
        block_diff = ratio_0 - ratio_1
        
        p_diff_ok = (p_ratio_01 > self.p_diff_threshold)
        entropy_diff_ok = (entropy_diff_01 > self.entropy_diff_threshold)
        blocks_ok = (block_diff > self.block_diff_threshold) and (ratio_0 > self.min_suspicious_ratio)
        absolute_random = (p_value_0 > self.chi_loose) and (entropy_0 > self.entropy_threshold)
        
        lsb1_lsb2_similar = (abs(p_value_1 - p_value_2) < 0.1) and (abs(entropy_1 - entropy_2) < 0.04)
        
        score_relative = sum([p_diff_ok, entropy_diff_ok, blocks_ok])
        
        if score_relative >= 2:
            return 1
        
        if absolute_random and (p_diff_ok or entropy_diff_ok or blocks_ok):
            return 1
        
        if (p_diff_ok or entropy_diff_ok) and lsb1_lsb2_similar and score_relative >= 1:
            return 1
        
        return 0