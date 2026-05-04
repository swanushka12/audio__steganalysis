# analysis/metadata.py

import os
import re
import zlib
import string
import numpy as np
from collections import Counter
from typing import Dict, List, Optional, Tuple

class MetadataDetector:    
    SIGNATURES = {
        'append': {
            'separators': [' || ', ' | ', ' // ', ' ;; ', ' ; ', ' , ', ': ', ' - '],
            'min_extra_len': 5 
        },
        'entropy': {
            'entropy_threshold': 5.0,     
            'charset_diversity': 0.4      
        },
        'padding': {
            'pad_chars': ['\x00', ' ', '\t', '\n', '\xa0', '\r', '\x0b', '\x0c'],
            'min_pad_ratio': 0.2          
        },
        'strange_fields': {
            'name_patterns': [
                r'^_meta_\d+$',           
                r'^__\w+__$',              
                r'^\w+_[0-9a-f]{4}$',     
                r'^X_PAD_\d+$',            
                r'^GAP_\d+$',              
                r'^[A-Z]{2,}_\d+$',        
                r'^[a-z]+_[a-z]+_\d+$',  
                r'^\w{8,}$'                
            ],
            'value_patterns': [
                r'^[0-9a-f]+$',
                r'^[0-9a-f]{8,}_',
                r'^\[GAP_SYNC\].*$',
                r'^\{FRAME_START\}.*$',
                r'^#AUDIO_BLOCK#.*$',
                r'^[A-Za-z0-9+/=]{20,}$' # base64
            ]
        },
        'checksum': {
            'field_names': ['X_CRC', 'CHECKSUM', 'VERIFY', 'SIG', 'HASH', 'MD5', 'SHA1', 'CRC32'],
            'crc_pattern': r'^[0-9a-f]{8,}_'
        },
        'frame_gaps': {
            'field_prefix': 'GAP_',
            'sync_markers': ['[GAP_SYNC]', '{FRAME_START}', '#AUDIO_BLOCK#', 'SYNC', 'FRAME']
        },
        'id3_anomaly': {
            'suspicious_frame_ids': ['TXXX', 'PRIV', 'GEOB', 'UFID', 'MCDI', 'ETCO', 'POSS', 'SYLT'],
            'min_extra_frames': 1          
        },
        'vorbis_anomaly': {
            'allowed_fields': {'TITLE', 'ARTIST', 'ALBUM', 'COMMENT', 'GENRE'},
            'max_normal_len': 512
        },
        'multiple_encodings': {
            'suspicious_combos': [
                ('TIT2', 'TXXX'), ('TPE1', 'TXXX'), ('COMM', 'TXXX'),
                ('TALB', 'TXXX'), ('TCON', 'TXXX'), ('TIT2', 'COMM'),
                ('TPE1', 'COMM'), ('USLT', 'COMM')
            ],
            'min_txxx_count': 1
        },
        'binary_data': {
            'null_byte_ratio': 0.03,       
            'non_printable_ratio': 0.2,    
            'max_text_ratio': 0.7         
        },
        'length_anomaly': {
            'max_normal_comment': 150,     
            'max_normal_txxx': 80,         
            'max_normal_field': 100        
        },
        'suspicious_chars': {
            'chars': ['\x00', '\x01', '\x02', '\x03', '\x04', '\x05', '\x06', '\x07',
                     '\x08', '\x0b', '\x0c', '\x0e', '\x0f', '\x10', '\x11', '\x12',
                     '\x13', '\x14', '\x15', '\x16', '\x17', '\x18', '\x19', '\x1a',
                     '\x1b', '\x1c', '\x1d', '\x1e', '\x1f', '\x7f'],
            'min_suspicious_ratio': 0.05   
        }
    }
    
    LEGIT_FIELDS = {
        'wav': {'TITLE', 'ARTIST', 'ALBUM', 'COMMENT', 'LYRICS', 'GENRE', 'YEAR', 'TRACK', 'IART', 'INAM', 'IPRD', 'ICMT', 
            'ICOP', 'ISFT', 'ITCH', 'IENG', 'ICRD','DATE', 'ISBJ', 'IGNR', 'ITRK', 'ISRC', 'IKEY', 'IMED'},
        'mp3': {'TIT2', 'TPE1', 'TALB', 'COMM', 'TCON'},
        'flac': {'TITLE', 'ARTIST', 'ALBUM', 'COMMENT', 'LYRICS', 'GENRE', 'DATE', 'TRACKNUMBER'},
        'ogg': {'TITLE', 'ARTIST', 'ALBUM', 'COMMENT', 'GENRE'}
    }
    
    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self.detection_log = []
        self.detection_results = []
    
    def calculate_entropy(self, text: str) -> float:
        if not text:
            return 0.0
        for encoding in ['utf-8', 'latin-1', 'ascii']:
            try:
                byte_data = text.encode(encoding, errors='strict')
                break
            except:
                continue
        else:
            byte_data = text.encode('utf-8', errors='replace')
        
        if len(byte_data) == 0:
            return 0.0
        counts = Counter(byte_data)
        length = len(byte_data)
        return -sum((c/length) * np.log2(c/length) for c in counts.values())
    
    def calculate_charset_diversity(self, text: str) -> float:
        if not text:
            return 0.0
        return len(set(text)) / max(len(text), 1)
    
    def is_printable_ratio(self, text: str) -> float:
        if not text:
            return 0.0
        printable = sum(1 for c in text if c in string.printable)
        return printable / len(text)
    
    def is_legit_field(self, field_name: str, format_hint: str) -> bool:
        legit = self.LEGIT_FIELDS.get(format_hint.lower(), set())
        field_upper = field_name.upper()
        
        if format_hint == 'mp3':
            if field_upper in ['TXXX', 'USLT', 'APIC', 'PRIV', 'GEOB', 'UFID']:
                return False
            return field_upper in legit or field_name in legit
        
        if format_hint == 'ogg':
            return field_upper in legit or field_name in legit
        
        return field_upper in legit or field_name in legit
    
    def looks_random(self, text: str) -> bool:
        if not text or len(text) < 6:         
            return False
        
        entropy = self.calculate_entropy(text)
        diversity = self.calculate_charset_diversity(text)
        
        return entropy > 3.0 or (entropy > 2.5 and diversity > 0.3)  
    
    
    def detect_append(self, key: str, value: str, format_hint: str) -> Tuple[bool, float, str]:
        for sep in self.SIGNATURES['append']['separators']:
            if sep in value:
                parts = value.split(sep)
                if len(parts) >= 2:
                    last_part = parts[-1]
                    if len(last_part) >= self.SIGNATURES['append']['min_extra_len']:
                        entropy = self.calculate_entropy(last_part)
                        if entropy > 2.8:   
                            return True, 0.9, f"Стего-append: энтропия {entropy:.2f}"
                        elif len(last_part) > 20:  
                            return True, 0.7, f"Стего-append: длинный суффикс ({len(last_part)} байт)"
        return False, 0.0, ""
    
    def detect_entropy(self, key: str, value: str, format_hint: str) -> Tuple[bool, float, str]:
        if len(value) < 10:                   
            return False, 0.0, ""
        
        entropy = self.calculate_entropy(value)
        diversity = self.calculate_charset_diversity(value)
        
        if format_hint in ['mp3', 'ogg']:
            if entropy > 3.5:
                return True, 0.85, f"Стего-entropy: {entropy:.2f}"
            elif entropy > 3.0 and diversity > 0.3:
                return True, 0.75, f"Стего-entropy: энтропия {entropy:.2f}, разнообразие {diversity:.2f}"
            elif entropy > 2.8:
                printable_ratio = self.is_printable_ratio(value)
                if printable_ratio > 0.9:
                    return True, 0.65, f"Стего-entropy: печатные с энтропией {entropy:.2f}"
        
        return False, 0.0, ""
    
    def detect_padding(self, key: str, value: str, format_hint: str) -> Tuple[bool, float, str]:
        if len(value) < 15:          
            return False, 0.0, ""
        
        pad_chars = self.SIGNATURES['padding']['pad_chars']
        pad_count = sum(1 for c in value if c in pad_chars)
        pad_ratio = pad_count / len(value)
        
        if pad_ratio >= self.SIGNATURES['padding']['min_pad_ratio']:
            stripped = value.lstrip(''.join(pad_chars))
            
            if key.startswith('X_PAD_') or key.startswith('x_pad_'):
                return True, 0.95, f"Стего-padding: поле X_PAD_"
            
            if len(stripped) > 5:
                return True, 0.8, f"Стего-padding: {pad_ratio*100:.0f}% padding с данными"
            
            if pad_ratio > 0.5:
                return True, 0.7, f"Стего-padding: {pad_ratio*100:.0f}% padding"
        
        if key.startswith('X_PAD_') or key.startswith('x_pad_'):
            return True, 0.75, "Поле X_PAD_ обнаружено"
        
        return False, 0.0, ""
    
    def detect_strange_fields(self, key: str, value: str, format_hint: str) -> Tuple[bool, float, str]:
        is_legit = self.is_legit_field(key, format_hint)
        
        if format_hint in ['mp3', 'ogg'] and not is_legit:
            for pattern in self.SIGNATURES['strange_fields']['name_patterns']:
                if re.match(pattern, key):
                    return True, 0.95, f"Стего-strange_fields: '{key}'"
            
            if len(value) > 10:
                return True, 0.7, f"Нелегитимное поле '{key}' в {format_hint.upper()}"
        
        for vp in self.SIGNATURES['strange_fields']['value_patterns']:
            if re.match(vp, value):
                if len(value) > 8:
                    return True, 0.85, f"Стего-данные в '{key}'"
        
        if re.match(r'^[0-9a-f]{10,}$', value, re.IGNORECASE):
            return True, 0.8, f"Hex-строка в '{key}'"
        
        return False, 0.0, ""
    
    def detect_checksum(self, key: str, value: str, format_hint: str) -> Tuple[bool, float, str]:
        checksum_names = [name.upper() for name in self.SIGNATURES['checksum']['field_names']]
        
        if key.upper() in checksum_names:
            return True, 0.95, f"Стего-checksum: поле {key}"
        
        if re.match(self.SIGNATURES['checksum']['crc_pattern'], value):
            return True, 0.9, f"Стего-checksum: CRC32_данные"
        
        key_upper = key.upper()
        if any(name in key_upper for name in ['CHECK', 'CRC', 'HASH', 'VERIFY', 'SIG']):
            if len(value) > 8:
                return True, 0.75, f"Подозрительное checksum-поле: {key}"
        
        return False, 0.0, ""
    
    def detect_frame_gaps(self, key: str, value: str, format_hint: str) -> Tuple[bool, float, str]:
        if key.startswith(self.SIGNATURES['frame_gaps']['field_prefix']):
            return True, 0.95, f"Стего-frame_gaps: поле {key}"
        
        for marker in self.SIGNATURES['frame_gaps']['sync_markers']:
            if marker in value:
                return True, 0.95, f"Стего-frame_gaps: {marker}"
        
        if 'GAP' in key.upper() and len(value) > 5:
            return True, 0.7, f"Подозрительное GAP поле: {key}"
        
        return False, 0.0, ""
    
    def detect_binary_data(self, key: str, value: str, format_hint: str) -> Tuple[bool, float, str]:
        if len(value) < 15:               
            return False, 0.0, ""
        
        null_count = value.count('\x00')
        null_ratio = null_count / len(value)
        
        non_printable = sum(1 for c in value if c not in string.printable)
        non_printable_ratio = non_printable / len(value)
        
        if null_ratio > self.SIGNATURES['binary_data']['null_byte_ratio']:
            return True, 0.95, f"Нулевые байты ({null_ratio*100:.1f}%)"
        
        if non_printable_ratio > self.SIGNATURES['binary_data']['non_printable_ratio']:
            return True, 0.85, f"Непечатные символы ({non_printable_ratio*100:.1f}%)"
        
        printable_ratio = self.is_printable_ratio(value)
        if printable_ratio < self.SIGNATURES['binary_data']['max_text_ratio']:
            return True, 0.75, f"Мало печатных ({printable_ratio*100:.1f}%)"
        
        return False, 0.0, ""
    
    def detect_length_anomaly(self, key: str, value: str, format_hint: str) -> Tuple[bool, float, str]:
        if len(value) < 30:                
            return False, 0.0, ""
        
        key_upper = key.upper()
        
        if key_upper in ['COMMENT', 'COMM', 'DESCRIPTION']:
            if len(value) > self.SIGNATURES['length_anomaly']['max_normal_comment']:
                return True, 0.7, f"Длинный COMMENT ({len(value)} байт)"
        
        elif key_upper == 'TXXX' or key.startswith('TXXX'):
            if len(value) > self.SIGNATURES['length_anomaly']['max_normal_txxx']:
                return True, 0.75, f"Длинный TXXX ({len(value)} байт)"
        
        else:
            if len(value) > self.SIGNATURES['length_anomaly']['max_normal_field']:
                if not self.is_legit_field(key, format_hint):
                    return True, 0.7, f"Длинное поле '{key}' ({len(value)} байт)"
        
        return False, 0.0, ""
    
    def detect_suspicious_chars(self, key: str, value: str, format_hint: str) -> Tuple[bool, float, str]:
        suspicious = self.SIGNATURES['suspicious_chars']['chars']
        suspicious_count = sum(1 for c in value if c in suspicious)
        
        if len(value) > 0:
            suspicious_ratio = suspicious_count / len(value)
            if suspicious_ratio > self.SIGNATURES['suspicious_chars']['min_suspicious_ratio']:
                return True, 0.85, f"Управляющие символы ({suspicious_ratio*100:.1f}%)"
        
        if '\\x' in value or '\\u' in value:
            return True, 0.7, "Escape-последовательности"
        
        return False, 0.0, ""
    
    def detect_base64(self, key: str, value: str, format_hint: str) -> Tuple[bool, float, str]:
        if len(value) < 16:               
            return False, 0.0, ""
        
        b64_pattern = r'^[A-Za-z0-9+/]+=*$'
        if re.match(b64_pattern, value):
            if len(value) % 4 == 0 and len(value) > 16:
                return True, 0.8, "Base64-кодированные данные"
        
        return False, 0.0, ""
    
    def detect_id3_anomaly(self, key: str, value: str, format_hint: str, all_metadata: Dict = None) -> Tuple[bool, float, str]:
        if format_hint != 'mp3':
            return False, 0.0, ""
        
        key_upper = key.upper()
        
        suspicious_frames = self.SIGNATURES['id3_anomaly']['suspicious_frame_ids']
        if key_upper in suspicious_frames:
            confidence = 0.95 if len(value) > 30 else 0.8
            return True, confidence, f"Подозрительный фрейм: {key}"
        
        if key.startswith('TXXX') or key_upper == 'TXXX':
            if len(value) > 20:
                return True, 0.85, f"TXXX фрейм с данными ({len(value)} байт)"
        
        return False, 0.0, ""
    
    def detect_vorbis_anomaly(self, key: str, value: str, format_hint: str, all_metadata: Dict = None) -> Tuple[bool, float, str]:
        if format_hint != 'ogg':
            return False, 0.0, ""
        
        allowed = self.SIGNATURES['vorbis_anomaly']['allowed_fields']
        key_upper = key.upper()
        
        if key_upper not in allowed:
            return True, 0.85, f"Нестандартное поле OGG: '{key}'"
        
        if len(value) > self.SIGNATURES['vorbis_anomaly']['max_normal_len']:
            return True, 0.75, f"Длинное поле {key} ({len(value)} байт)"
        
        return False, 0.0, ""
    
    def detect_multiple_encodings(self, key: str, value: str, format_hint: str, all_metadata: Dict = None) -> Tuple[bool, float, str]:
        if format_hint != 'mp3' or not all_metadata:
            return False, 0.0, ""
        
        for combo in self.SIGNATURES['multiple_encodings']['suspicious_combos']:
            if key.upper() in combo:
                other_fields = [f for f in combo if f != key.upper()]
                for other in other_fields:
                    if other in all_metadata:
                        other_val = all_metadata[other]
                        if isinstance(other_val, str) and len(value) > 10 and len(other_val) > 10:
                            return True, 0.7, f"Дублирование: {key} и {other}"
        
        return False, 0.0, ""
    
    def detect_timestamp_anomaly(self, key: str, value: str, format_hint: str) -> Tuple[bool, float, str]:
        if format_hint != 'mp3':
            return False, 0.0, ""
        
        if key.upper() in ['TDRC', 'TDAT', 'TYER', 'TIME', 'TRDA', 'TORY', 'TOFN', 'TKEY']:
            if len(value) > 10:
                return True, 0.6, f"Подозрительный timestamp: {key}"
        
        return False, 0.0, ""
    
    def analyze(self, metadata: Optional[Dict] = None, filepath: Optional[str] = None, format_hint: Optional[str] = None) -> int:
        if format_hint is None and filepath:
            ext = os.path.splitext(filepath)[1].lower().lstrip('.')
            format_hint = ext if ext in ['wav', 'mp3', 'flac', 'ogg'] else 'wav'
        elif format_hint is None:
            format_hint = 'wav'
        
        if metadata is None or len(metadata) == 0:
            return 0
        
        is_compressed = format_hint in ['mp3', 'ogg']
        
        SYSTEM_FIELDS = {'channels', 'samplerate', 'duration', 'format', 'subtype', 'bitrate', 'length', 'bitrate_mode', 'encoder', 
            'tracknumber', 'discnumber', 'total_tracks', 'total_discs', 'bitrate', 'framelength'}
        
        all_detectors = [
            ('append', self.detect_append),
            ('entropy', self.detect_entropy),
            ('padding', self.detect_padding),
            ('strange_fields', self.detect_strange_fields),
            ('checksum', self.detect_checksum),
            ('frame_gaps', self.detect_frame_gaps),
            ('binary_data', self.detect_binary_data),
            ('length_anomaly', self.detect_length_anomaly),
            ('suspicious_chars', self.detect_suspicious_chars),
            ('base64', self.detect_base64),
        ]
        
        if format_hint == 'mp3':
            all_detectors.extend([
                ('id3_anomaly', lambda k, v, f: self.detect_id3_anomaly(k, v, f, metadata)),
                ('multiple_encodings', lambda k, v, f: self.detect_multiple_encodings(k, v, f, metadata)),
                ('timestamp_anomaly', self.detect_timestamp_anomaly),
            ])
        elif format_hint == 'ogg':
            all_detectors.extend([
                ('vorbis_anomaly', lambda k, v, f: self.detect_vorbis_anomaly(k, v, f, metadata)),
            ])
        
        if is_compressed:
            DETECTION_RULES = {
                'single_detection_min_conf': 0.40,     
                'any_non_legit_field': True,             
                'min_suspicious_ratio': 0.1,             
                'any_detection_counts': True,           
                'lenient_confidence': 0.6              
            }
        else:
            DETECTION_RULES = {
                'single_detection_min_conf': 0.70,
                'any_non_legit_field': False,
                'min_suspicious_ratio': 0.30,
                'any_detection_counts': False,
                'lenient_confidence': 0.8
            }
        
        results = []
        max_confidence = 0.0
        total_detections = 0
        total_fields = 0
        suspicious_fields = 0
        non_legit_fields = 0
        
        for key, value in metadata.items():
            if not isinstance(value, str):
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], str):
                    value = value[0]
                else:
                    continue
            
            if not value or len(value) < 4:
                continue
            
            if key.lower() in SYSTEM_FIELDS:
                continue
            
            total_fields += 1
            
            is_legit = self.is_legit_field(key, format_hint)
            if not is_legit and is_compressed:
                non_legit_fields += 1
            
            field_detected = False
            for method_name, detector_func in all_detectors:
                try:
                    detected, confidence, reason = detector_func(key, value, format_hint)
                    if detected:
                        results.append({
                            'method': method_name,
                            'field': key,
                            'confidence': confidence,
                            'reason': reason
                        })
                        max_confidence = max(max_confidence, confidence)
                        total_detections += 1
                        field_detected = True
                        self.detection_log.append(f"[{method_name}] {key}: {reason} (conf={confidence:.2f})")
                except Exception as e:
                    continue
            
            if field_detected:
                suspicious_fields += 1
        
        if is_compressed:
            if DETECTION_RULES['any_detection_counts'] and total_detections >= 1:
                return 1
            
            if max_confidence >= DETECTION_RULES['single_detection_min_conf']:
                return 1
            
            if DETECTION_RULES['any_non_legit_field'] and non_legit_fields >= 1:
                return 1
            
            if total_fields > 0 and suspicious_fields / total_fields >= DETECTION_RULES['min_suspicious_ratio']:
                return 1
        
        else:
            if max_confidence >= DETECTION_RULES['single_detection_min_conf']:
                return 1
            if total_detections >= 2:
                return 1
            if total_fields > 0 and suspicious_fields / total_fields >= DETECTION_RULES['min_suspicious_ratio']:
                return 1
        
        self.detection_results = results
        
        return 0