# analysis/utils.py
import numpy as np
import scipy.stats as stats
from scipy.signal import spectrogram
import os
import struct

def is_compressed(file_path):
    compressed_extensions = ['.mp3', '.aac', '.ogg', '.m4a', '.wma', '.opus']
    _, ext = os.path.splitext(file_path)
    return ext.lower() in compressed_extensions

def get_metadata(audio_path):
    if not os.path.exists(audio_path):
        return {}
    
    ext = os.path.splitext(audio_path)[1].lower()
    metadata = {}
    
    try:
        if ext == '.wav':
            metadata = read_wav_metadata(audio_path)
        elif ext == '.flac':
            metadata = read_flac_metadata(audio_path)
        elif ext == '.mp3':
            metadata = read_mp3_metadata(audio_path)
        else:
            metadata = read_soundfile_metadata(audio_path)
    except Exception as e:
        try:
            import soundfile as sf
            with sf.SoundFile(audio_path) as f:
                metadata = {
                    'channels': f.channels,
                    'samplerate': f.samplerate,
                    'duration': f.frames / f.samplerate,
                    'format': f.format,
                    'subtype': f.subtype
                }
        except:
            pass
    
    return metadata

def read_wav_metadata(filepath):
    metadata = {}
    
    try:
        with open(filepath, 'rb') as f:
            # RIFF
            riff = f.read(4)
            if riff != b'RIFF':
                return metadata
            
            # размер файла
            file_size = struct.unpack('<I', f.read(4))[0]
            wave = f.read(4)
            if wave != b'WAVE':
                return metadata
            
            # все чанки
            pos = 12
            while pos < file_size + 8:
                f.seek(pos)
                chunk_id = f.read(4)
                if len(chunk_id) < 4:
                    break
                
                chunk_size = struct.unpack('<I', f.read(4))[0]
                
                if chunk_id == b'LIST':
                    list_type = f.read(4)
                    if list_type == b'INFO':
                        end_pos = pos + 8 + chunk_size
                        pos = f.tell()
                        while pos < end_pos:
                            f.seek(pos)
                            info_id = f.read(4)
                            if len(info_id) < 4:
                                break
                            
                            info_size = struct.unpack('<I', f.read(4))[0]
                            info_data = f.read(info_size)
                            
                            if info_size % 2:
                                f.read(1)
                                pos += 1
                            
                            try:
                                key = info_id.decode('ascii', errors='replace').strip('\x00').strip()
                                value = info_data.decode('utf-8', errors='replace').strip('\x00').strip()
                                if key and value:
                                    metadata[key] = value
                            except:
                                pass
                            
                            pos += 8 + info_size
                    else:
                        pass
                
                elif chunk_id == b'data':
                    break
                
                pos += 8 + chunk_size
                if chunk_size % 2:
                    pos += 1
    
    except Exception as e:
        pass
    
    return metadata

def read_flac_metadata(filepath):
    metadata = {}
    try:
        from mutagen.flac import FLAC
        audio = FLAC(filepath)
        
        if hasattr(audio, 'tags') and audio.tags:
            for key, values in audio.tags.items():
                if isinstance(values, list) and len(values) > 0:
                    metadata[key] = values[0]
                else:
                    metadata[key] = str(values)
        
        if hasattr(audio, 'info'):
            metadata['samplerate'] = audio.info.sample_rate
            metadata['channels'] = audio.info.channels
            metadata['duration'] = audio.info.length
    
    except ImportError:
        try:
            import soundfile as sf
            with sf.SoundFile(filepath) as f:
                metadata = {
                    'channels': f.channels,
                    'samplerate': f.samplerate,
                    'duration': f.frames / f.samplerate,
                    'format': f.format,
                    'subtype': f.subtype
                }
        except:
            pass
    except Exception as e:
        pass
    
    return metadata

def read_mp3_metadata(filepath):
    metadata = {}
    try:
        from mutagen.id3 import ID3
        audio = ID3(filepath)
        
        for key, value in audio.items():
            # ID3 фреймы в строки
            try:
                if hasattr(value, 'text'):
                    text = value.text
                    if isinstance(text, list):
                        metadata[key] = str(text[0]) if text else ''
                    else:
                        metadata[key] = str(text)
                elif hasattr(value, 'url'):
                    metadata[key] = str(value.url)
                else:
                    metadata[key] = str(value)
            except:
                metadata[key] = str(value)
    
    except ImportError:
        try:
            from mutagen.mp3 import MP3
            audio = MP3(filepath)
            if hasattr(audio, 'tags') and audio.tags:
                for key, value in audio.tags.items():
                    metadata[key] = str(value)
            metadata['duration'] = audio.info.length
            metadata['bitrate'] = audio.info.bitrate
        except:
            pass
    except Exception as e:
        pass
    
    return metadata

def read_soundfile_metadata(filepath):
    metadata = {}
    
    try:
        import soundfile as sf
        with sf.SoundFile(filepath) as f:
            metadata = {
                'channels': f.channels,
                'samplerate': f.samplerate,
                'duration': f.frames / f.samplerate,
                'format': f.format,
                'subtype': f.subtype
            }
    except:
        pass
    
    return metadata
