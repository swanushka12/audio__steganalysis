# embedding/lsb_coding.py

import numpy as np
import random
import sys
import os

class LSBCoding:
    def generate_random_bits(self, num_bits, seed = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        return [random.randint(0, 1) for _ in range(num_bits)]
    
    def bits_to_bytes(self, bits):
        padded_bits = bits + [0] * ((8 - len(bits) % 8) % 8)
        
        bytes_data = bytearray()
        for i in range(0, len(padded_bits), 8):
            byte = 0
            for j in range(8):
                byte |= padded_bits[i + j] << (7 - j)
            bytes_data.append(byte)
        
        return bytes(bytes_data)
    
    def bytes_to_bits(self, data):
        bits = []
        for byte in data:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        return bits
    
    def embed(self, audio_samples, secret_bits):
        if audio_samples.dtype == np.float32 or audio_samples.dtype == np.float64:
            samples_int16 = (audio_samples * 32767).astype(np.int16)
        else:
            samples_int16 = audio_samples.astype(np.int16)
        
        samples = samples_int16.copy()
        n_samples = len(samples)
        max_capacity = n_samples 
        
        if len(secret_bits) > max_capacity:
            secret_bits = secret_bits[:max_capacity]
        
        embedded_count = 0
        bit_idx = 0
        
        mask = 0xFE  # 11111110
        for i in range(min(len(secret_bits), n_samples)):
            if bit_idx < len(secret_bits):
                samples[i] = (samples[i] & mask) | secret_bits[bit_idx]
                bit_idx += 1
                embedded_count += 1
        
        stego_audio = samples.astype(np.float32) / 32767.0
        return stego_audio, embedded_count, max_capacity

    
    def embed_test_message(self, audio_data, embed_fraction = 0, seed = None):
        result = {
            'audio_data': audio_data.copy(),
            'was_embedded': False,
            'embedded_bits': 0,
            'max_capacity': 0,
        }
        
        if embed_fraction <= 0:
            return result
        
        try:
            if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                samples_int16 = (audio_data * 32767).astype(np.int16)
            else:
                samples_int16 = audio_data.astype(np.int16)
            
            max_capacity = len(samples_int16)
            target_bits = max(8, int(max_capacity * embed_fraction))
            target_bits = min(target_bits, max_capacity)
            
            # генерация случайных бит
            rand_bits = self.generate_random_bits(target_bits, seed)
            
            # встраивание
            stego_audio, embedded_bits, _ = self.embed(audio_data, rand_bits)
            
            result['audio_data'] = stego_audio
            result['was_embedded'] = True
            result['embedded_bits'] = embedded_bits
            result['max_capacity'] = max_capacity
                        
        except Exception as e:
            print(f"Ошибка LSB встраивания: {e}")
        
        return result
    
    
    def get_capacity(self, audio_data):
        if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
            samples_int16 = (audio_data * 32767).astype(np.int16)
        else:
            samples_int16 = audio_data.astype(np.int16)
        
        return len(samples_int16)
