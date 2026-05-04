# embedding/metadata_coding.py
import random
import string
import zlib
import os
import shutil
import struct

class MetadataCoding:
    METHODS = ['append', 'entropy', 'padding', 'strange_fields', 'checksum', 'frame_gaps']

    def __init__(self, seed = None):
        self.rng = random.Random(seed)
        self.last_method = None

    def embed_test_message(self, audio_path, fraction = 0.5, seed = None, output_path = None):
        if seed is not None:
            self.rng = random.Random(seed)

        # формат
        ext = os.path.splitext(audio_path)[1].lower()
        format_hint = ext.lstrip('.')
        
        # выбор метода
        method = self.rng.choice(self.METHODS)
        self.last_method = method
        payload_size = max(16, int(128 * fraction))
        payload = ''.join(self.rng.choice(string.printable) for _ in range(payload_size))

        methods_map = {
            'append': self.method_append,
            'entropy': self.method_entropy,
            'padding': self.method_padding,
            'strange_fields': self.method_strange_fields,
            'checksum': self.method_checksum,
            'frame_gaps': self.method_frame_gaps,
        }

        stego_metadata = methods_map[method](payload, format_hint)        
        if output_path is None:
            base_dir = os.path.dirname(audio_path) or '.'
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            output_path = os.path.join(base_dir, f"{base_name}_stego{ext}")
        
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)        
        success = self.write_metadata_to_file(audio_path, output_path, stego_metadata, format_hint)
        
        return {
            'was_embedded': success,
            'method_used': method,
            'embedded_bytes': len(payload),
            'stego_metadata': stego_metadata,
            'output_path': output_path if success else None,
            'format': format_hint
        }

    def write_metadata_to_file(self, input_path, output_path, metadata, format_hint):
        try:
            shutil.copy2(input_path, output_path)
            
            if format_hint == 'wav':
                self._write_wav_metadata(output_path, metadata)
            elif format_hint == 'flac':
                self._write_flac_metadata(output_path, metadata)
            elif format_hint == 'mp3':
                self._write_mp3_metadata(output_path, metadata)
            elif format_hint == 'ogg':
                self._write_ogg_metadata(output_path, metadata)
            else:
                pass
            
            return True
        except Exception as e:
            print(f"Ошибка записи метаданных: {e}")
            if not os.path.exists(output_path):
                try:
                    shutil.copy2(input_path, output_path)
                except:
                    pass
            return False

    def write_ogg_metadata(self, filepath, metadata):
        try:
            from mutagen.oggvorbis import OggVorbis
            audio = OggVorbis(filepath)
            
            for key, value in metadata.items():
                audio[key] = value
            
            audio.save()
        except ImportError:
            pass
        except Exception as e:
            print(f"Ошибка записи OGG метаданных: {e}")

    def write_metadata_to_file(self, input_path, output_path, metadata, format_hint):
        try:
            shutil.copy2(input_path, output_path)
            
            if format_hint == 'wav':
                self.write_wav_metadata(output_path, metadata)
            elif format_hint == 'flac':
                self.write_flac_metadata(output_path, metadata)
            elif format_hint == 'mp3':
                self.write_mp3_metadata(output_path, metadata)
            else:
                pass
            
            return True
        except Exception as e:
            print(f"Ошибка записи метаданных: {e}")
            if not os.path.exists(output_path):
                try:
                    shutil.copy2(input_path, output_path)
                except:
                    pass
            return False

    def write_wav_metadata(self, filepath, metadata):
        try:
            with open(filepath, 'rb') as f:
                # RIFF
                riff = f.read(4)
                if riff != b'RIFF':
                    return
                
                file_size = struct.unpack('<I', f.read(4))[0]
                wave = f.read(4)
                if wave != b'WAVE':
                    return
                
                chunks_before_data = bytearray()
                pos = 12
                
                while pos < file_size + 8:
                    f.seek(pos)
                    chunk_id = f.read(4)
                    if len(chunk_id) < 4:
                        break
                    
                    chunk_size = struct.unpack('<I', f.read(4))[0]
                    
                    if chunk_id == b'data':
                        data_chunk_start = pos
                        data_chunk_size = chunk_size
                        # все аудиоданные
                        f.seek(pos + 8)
                        audio_data = f.read(chunk_size)
                        break
                    
                    if chunk_id == b'LIST':
                        f.seek(pos + 8)
                        list_type = f.read(4)
                        if list_type == b'INFO':
                            pos += 8 + chunk_size
                            if chunk_size % 2:
                                pos += 1
                            continue
                    
                    f.seek(pos)
                    chunk_data = f.read(8 + chunk_size)
                    chunks_before_data.extend(chunk_data)
                    
                    pos += 8 + chunk_size
                    if chunk_size % 2:
                        pos += 1
                
                # новый INFO чанк
                info_data = b''
                for key, value in metadata.items():
                    key_bytes = key.encode('ascii', errors = 'replace')[:4].ljust(4, b'\x00')
                    
                    # значение
                    value_bytes = value.encode('utf-8', errors = 'replace')
                    
                    # RIFF - чётная длина
                    if len(value_bytes) % 2:
                        value_bytes += b'\x00'
                    
                    info_data += key_bytes + struct.pack('<I', len(value_bytes)) + value_bytes
                
                if info_data:
                    list_chunk = b'LIST' + struct.pack('<I', 4 + len(info_data)) + b'INFO' + info_data
                    
                    # padding
                    if len(list_chunk) % 2:
                        list_chunk += b'\x00'
                    
                    chunks_before_data.extend(list_chunk)
                
                new_content = b'RIFF'
                total_size = len(chunks_before_data) + 8 + len(audio_data)
                new_content += struct.pack('<I', total_size)
                new_content += b'WAVE'
                new_content += chunks_before_data
                new_content += b'data' + struct.pack('<I', len(audio_data)) + audio_data
            
            # новый файл
            with open(filepath, 'wb') as f:
                f.write(new_content)
                
        except Exception as e:
            print(f"Ошибка записи WAV метаданных: {e}")

    def write_flac_metadata(self, filepath, metadata):
        try:
            from mutagen.flac import FLAC
            audio = FLAC(filepath)
            
            for key, value in metadata.items():
                audio[key] = value
            
            audio.save()

        except ImportError:
            self._write_wav_metadata(filepath, metadata)
        except Exception as e:
            print(f"Ошибка записи FLAC метаданных: {e}")

    def write_mp3_metadata(self, filepath, metadata):
        try:
            from mutagen.id3 import ID3, TXXX, COMM, TIT2, TPE1, TALB
            from mutagen.mp3 import MP3
            
            audio = MP3(filepath, ID3 = ID3)
            
            # если нет ID3 тегов
            if audio.tags is None:
                audio.add_tags()
            
            for key, value in metadata.items():
                if key.upper() in ['COMM', 'COMMENT']:
                    audio.tags.add(COMM(encoding = 3, lang = 'eng', text = value))
                elif key.upper() in ['TITLE', 'TIT2']:
                    audio.tags.add(TIT2(encoding = 3, text = value))
                elif key.upper() in ['ARTIST', 'TPE1']:
                    audio.tags.add(TPE1(encoding = 3, text = value))
                elif key.upper() in ['ALBUM', 'TALB']:
                    audio.tags.add(TALB(encoding = 3, text = value))
                else:
                    audio.tags.add(TXXX(encoding = 3, desc = key, text = value))
            
            audio.save()
        except Exception as e:
            print(f"Ошибка записи MP3 метаданных: {e}")

    
    def method_append(self, payload, fh):
        field = 'COMM' if fh == 'mp3' else 'COMMENT'
        sep = self.rng.choice([' || ', ' | ', ' // ', ' ;; '])
        return {field: f"Standard metadata text{sep}{payload}"}

    def method_entropy(self, payload, fh):
        field = 'TXXX' if fh == 'mp3' else 'COMMENT'
        noise = ''.join(self.rng.choices(string.ascii_letters + string.digits, k=len(payload)))
        mixed = ''.join(a if self.rng.random() > 0.5 else b for a, b in zip(payload, noise))
        return {field: mixed}

    def method_padding(self, payload, fh):
        pad_char = self.rng.choice(['\x00', ' ', '\t', '\xa0'])
        field = f"X_PAD_{self.rng.randint(1000, 9999)}"
        return {field: pad_char * self.rng.randint(50, 150) + payload}

    def method_strange_fields(self, payload, fh):
        names = [f"_meta_{self.rng.randint(100, 999)}", f"__ext_{zlib.crc32(payload.encode()) & 0xFFFF:04x}__"]
        field = self.rng.choice(names)
        val = payload.encode().hex() if self.rng.random() > 0.5 else payload
        return {field: val}

    def method_checksum(self, payload, fh):
        crc = zlib.crc32(payload.encode()) & 0xFFFFFFFF
        field = self.rng.choice(['X_CRC', 'CHECKSUM', 'VERIFY'])
        return {field: f"{crc:08x}_{payload[:16]}"}

    def method_frame_gaps(self, payload, fh):
        gap_marker = self.rng.choice(['[GAP_SYNC]', '{FRAME_START}', '#AUDIO_BLOCK#'])
        field = f"GAP_{self.rng.randint(1, 99):02d}"
        return {field: gap_marker + payload[:48]}