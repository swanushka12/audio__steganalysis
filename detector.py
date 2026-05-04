# detector.py 

import numpy as np
import librosa
import soundfile as sf
import time
import warnings
import json
import os
import sys
import argparse
import tempfile
import shutil
import subprocess
from pathlib import Path
warnings.filterwarnings('ignore')
from analysis.lsb import LSBDetector
from analysis.dsss import DSSSDetector
from analysis.phase import PhaseDetector
from analysis.utils import is_compressed, get_metadata
from analysis.echo import EchoDetector
from analysis.metadata import MetadataDetector
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from embedding import PhaseCoding, LSBCoding, DSSSCoding, EchoCoding, MetadataCoding

EMBED_METHOD = 'metadata'

class AudioStegDetector:
    def __init__(self, use_reports = False):
        self.use_reports = use_reports
        self.settings = {
            'enable_echo': True,
            'enable_phase': True,
            'enable_dsss': True,
            'enable_metadata': True,
            'enable_lsb': True
        }
        self.args = None
        
        if self.settings['enable_lsb']:
            self.lsb_detector = LSBDetector()
        
        if self.settings['enable_phase']:
            self.phase_detector = PhaseDetector()

        if self.settings['enable_echo']:
            self.echo_detector = EchoDetector()

        if self.settings['enable_dsss']:
            self.dsss_detector = DSSSDetector()
        
        if self.settings['enable_metadata']:
            self.metadata_detector = MetadataDetector()
  
        self.metadata_embedder = MetadataCoding()
        self.phase_embedder = PhaseCoding()
        self.lsb_embedder = LSBCoding()
        self.echo_embedder = EchoCoding()
        self.dsss_embedder = DSSSCoding()

    def load_audio(self, filepath, duration = 30):
        try:
            import scipy.io.wavfile as wav
            sr, audio_int16 = wav.read(filepath)
            
            audio_float = audio_int16.astype(np.float32) / 32767.0
            
            if len(audio_float.shape) > 1:
                audio_float = np.mean(audio_float, axis=1)
            
            return audio_float, sr
        except Exception as e:
            return librosa.load(filepath, sr = None, mono = True, duration = duration)

    def get_output_dir(self, audio_path):
        if self.args and hasattr(self.args, 'output_dir'):
            base_dir = self.args.output_dir
        else:
            base_dir = 'stego_output'
        
        ext = os.path.splitext(audio_path)[1].lower()
        
        if ext == '.mp3':
            output_dir = os.path.join(base_dir, 'mp3')
        elif ext == '.ogg':
            output_dir = os.path.join(base_dir, 'ogg')
        elif ext == '.flac':
            output_dir = os.path.join(base_dir, 'flac')
        else:
            output_dir = os.path.join(base_dir, 'wav')
        
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def save_audio_file(self, audio_data, sr, save_path, ext):
        try:
            if ext == '.wav':
                sf.write(save_path, audio_data, sr)
                return save_path
            
            tmp_wav = None
            try:
                tmp_wav = tempfile.NamedTemporaryFile(suffix = '.wav', delete = False)
                tmp_path = tmp_wav.name
                tmp_wav.close()
                
                sf.write(tmp_path, audio_data, sr)
                
                cmd = ['ffmpeg', '-y', '-i', tmp_path]
                
                if ext == '.mp3':
                    cmd += ['-codec:a', 'libmp3lame', '-b:a', '128k']
                elif ext == '.ogg':
                    cmd += ['-codec:a', 'libvorbis', '-b:a', '128k']
                elif ext == '.flac':
                    cmd += ['-codec:a', 'flac']
                
                cmd.append(save_path)
                
                result = subprocess.run(cmd, capture_output = True, timeout =30)
                
                if result.returncode != 0:
                    raise Exception(f"ffmpeg error: {result.stderr.decode()[:200]}")
                
                return save_path
                
            finally:
                if tmp_wav and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                        
        except Exception as e:
            print(f"Ошибка сохранения {ext}: {e}")
            return None

    def save_stego_file(self, audio_data, sr, original_path, method, output_dir = "stego_output"):
        ext = os.path.splitext(original_path)[1].lower()
        
        if ext == '.mp3':
            save_dir = os.path.join(output_dir, 'mp3')
        elif ext == '.ogg':
            save_dir = os.path.join(output_dir, 'ogg')
        else:
            save_dir = os.path.join(output_dir, method)
        
        os.makedirs(save_dir, exist_ok=True)
        
        original_name = os.path.basename(original_path)
        name_without_ext = os.path.splitext(original_name)[0]
        
        new_filename = f"{name_without_ext}_{method}{ext}"
        save_path = os.path.join(save_dir, new_filename)
        
        if method in ['lsb', 'echo']:
            try:
                audio_int16 = np.round(audio_data * 32767).astype(np.int16)
                import scipy.io.wavfile as wav
                wav.write(save_path, sr, audio_int16)
                print(f"Сохранён: {os.path.relpath(save_path)}")
                return save_path
            except Exception as e:
                print(f"Ошибка сохранения: {e}")
                return None
        else:
            saved = self.save_audio_file(audio_data, sr, save_path, ext)
            if saved:
                print(f"Сохранён: {os.path.relpath(saved)}")
            return saved

    def process_file(self, audio_path, embed_fraction = 0, embed_method = EMBED_METHOD, bits_per_sample = 1):
        audio_data = None
        sr = None
        
        if not os.path.exists(audio_path):
            print(f"Файл не существует: {audio_path}")
            return None
        
        # загрузка аудио
        if embed_method in ['lsb', 'echo'] and audio_path.lower().endswith('.wav'):
            try:
                audio_data, sr = self.load_audio(
                    audio_path, duration = 60 if embed_fraction > 0 else 0)
            except Exception as e:
                print(f"Ошибка загрузки: {e}")
                audio_data, sr = librosa.load(audio_path, sr = None, mono = True, duration = 30)
        else:
            try:
                audio_data, sr = librosa.load(audio_path, sr = None, mono = True, duration = 30)
            except Exception as e:
                try:
                    audio_data, sr = librosa.load(audio_path, sr = None, mono = True)
                    print(f"Загружен без duration: {os.path.basename(audio_path)}")
                except Exception as e2:
                    print(f"Ошибка загрузки {os.path.basename(audio_path)}: {str(e2)[:80]}")
                    return None
        
        original_audio = audio_data.copy()
        was_embedded = False
        embedded_bits = 0
        segment_size = 0
        total_capacity = 0
        first_capacity = 0
        bits_per_sample_used = bits_per_sample
        stego_metadata = None
        result = None
        stego_save_path = None
        stego_audio_data = None
        
        should_save_stego = (self.args and hasattr(self.args, 'save_stego') and self.args.save_stego)
        ext = os.path.splitext(audio_path)[1].lower()
        stego_output_dir = self.get_output_dir(audio_path)
        
        # встраивание
        if embed_fraction > 0:
            if embed_method == 'metadata':
                if hasattr(self, 'metadata_embedder'):
                    temp_dir = tempfile.mkdtemp()
                    base_name = os.path.splitext(os.path.basename(audio_path))[0]
                    temp_output = os.path.join(temp_dir, f"{base_name}_stego{ext}")
                    
                    result = self.metadata_embedder.embed_test_message(audio_path, embed_fraction, temp_output)
                    was_embedded = result.get('was_embedded', False)
                    embedded_bits = result.get('embedded_bytes', 0)
                    stego_metadata = result.get('stego_metadata', {})
                    
                    if was_embedded and result.get('output_path'):
                        print(f"Метаданные: встроено {embedded_bits} байт")
                        
                        if should_save_stego:
                            dest_path = os.path.join(
                                stego_output_dir, 
                                os.path.basename(temp_output)
                            )
                            shutil.copy2(temp_output, dest_path)
                            stego_save_path = dest_path
                            print(f"Сохранён: {os.path.relpath(dest_path)}")
                    else:
                        print(f"Метаданные: не удалось встроить")
                else:
                    print(f"Метаданные: встаривание не инициализировано")

            elif embed_method == 'phase':
                if hasattr(self, 'phase_embedder'):
                    result = self.phase_embedder.embed_test_message(original_audio, embed_fraction)
                    if result.get('was_embedded'):
                        was_embedded = True
                        embedded_bits = result.get('embedded_bits', 0)
                        stego_audio_data = result.get('audio_data', original_audio)
                        segment_size = result.get('segment_size_used', 0)
                        total_capacity = result.get('total_capacity_actual', 0)
                        first_capacity = result.get('first_capacity_actual', 0)
                        print(f"Фаза: встроено {embedded_bits} бит")
                        
                        if should_save_stego:
                            stego_save_path = self.save_stego_file(stego_audio_data, sr, audio_path, 'phase', os.path.dirname(stego_output_dir))
                    else:
                        print(f"Фаза: не удалось встроить")
                else:
                    print(f"Фаза: встраивание не инициализировано")
            
            elif embed_method == 'lsb':
                if hasattr(self, 'lsb_embedder'):
                    result = self.lsb_embedder.embed_test_message(original_audio, embed_fraction, bits_per_sample)
                    if result.get('was_embedded'):
                        was_embedded = True
                        embedded_bits = result.get('embedded_bits', 0)
                        stego_audio_data = result.get('audio_data', original_audio)
                        total_capacity = result.get('max_capacity', 0)
                        bits_per_sample_used = bits_per_sample
                        print(f"[Тест] LSB: встроено {embedded_bits} бит")
                        
                        if should_save_stego:
                            stego_save_path = self.save_stego_file(stego_audio_data, sr, audio_path, 'lsb', os.path.dirname(stego_output_dir))
                    else:
                        print(f"LSB: не удалось встроить")
                else:
                    print(f"LSB: встраивание не инициализировано")
            

            elif embed_method == 'echo':
                if hasattr(self, 'echo_embedder'):
                    result = self.echo_embedder.embed_test_message(original_audio, embed_fraction)
                    if result.get('was_embedded'):
                        was_embedded = True
                        embedded_bits = result.get('embedded_bits', 0)
                        stego_audio_data = result.get('audio_data', original_audio)
                        total_capacity = result.get('max_capacity', 0)
                        print(f"Эхо: встроено {embedded_bits} бит")
                        
                        if should_save_stego:
                            stego_save_path = self.save_stego_file(stego_audio_data, sr, audio_path, 'echo', os.path.dirname(stego_output_dir))
                    else:
                        print(f"Эхо: не удалось встроить")
                else:
                    print(f"Эхо: встраивание не инициализировано")

            elif embed_method == 'dsss':
                if hasattr(self, 'dsss_embedder'):
                    result = self.dsss_embedder.embed_test_message(original_audio, embed_fraction)
                    if result.get('was_embedded'):
                        was_embedded = True
                        embedded_bits = result.get('embedded_bits', 0)
                        stego_audio_data = result.get('audio_data', original_audio)
                        total_capacity = result.get('max_capacity', 0)
                        print(f"DSSS: встроено {embedded_bits} бит")
                        
                        if should_save_stego:
                            stego_save_path = self.save_stego_file(stego_audio_data, sr, audio_path, 'dsss', os.path.dirname(stego_output_dir))
                    else:
                        print(f"DSSS: не удалось встроить")
                else:
                    print(f"DSSS: встраивание не инициализировано")

            else:
                print(f"Неизвестный метод встраивания: {embed_method}")
        
        analyze_audio = stego_audio_data if (was_embedded and stego_audio_data is not None) else audio_data
        analyze_path = stego_save_path if (was_embedded and stego_save_path) else audio_path
        compressed = is_compressed(analyze_path)
        analysis_results = {}
        detected_method = None
        
        # метаданные
        if self.settings['enable_metadata']:
            try:
                metadata_detector = MetadataDetector()
                metadata_from_file = get_metadata(analyze_path)
                
                if metadata_from_file and len(metadata_from_file) > 0:
                    metadata_result = metadata_detector.analyze(metadata = metadata_from_file, filepath = analyze_path)
                    if metadata_result == 1:
                        analysis_results['metadata_analysis'] = 1
                        detected_method = 'metadata'
                        return self.create_result(analyze_path, analyze_audio, sr, compressed, analysis_results, detected_method, was_embedded, 
                            embedded_bits, segment_size, total_capacity, first_capacity)
            except Exception as e:
                print(f"Ошибка анализа : {e}")
        
        # LSB
        if not compressed and self.settings['enable_lsb']:
            try:
                lsb_result = self.lsb_detector.analyze(analyze_audio)
                if lsb_result == 1:
                    analysis_results['lsb_analysis'] = 1
                    detected_method = 'lsb'
                    return self.create_result(analyze_path, analyze_audio, sr, compressed, analysis_results, 
                        detected_method, was_embedded, embedded_bits, segment_size, total_capacity, first_capacity)
            except Exception as e:
                print(f"Ошибка LSB анализа: {e}")
        
        # фазовый анализ
        if self.settings['enable_phase']:
            try:
                phase_result = self.phase_detector.analyze(analyze_audio)
                if phase_result == 1:
                    analysis_results['phase_analysis'] = 1
                    detected_method = 'phase'
                    return self.create_result(analyze_path, analyze_audio, sr, compressed, analysis_results, detected_method, was_embedded, 
                        embedded_bits, segment_size, total_capacity, first_capacity)
            except Exception as e:
                print(f"Ошибка фазового анализа: {e}")

        # эхо
        if not compressed and self.settings['enable_echo']:
            try:
                echo_result = self.echo_detector.analyze(analyze_audio)
                if echo_result == 1:
                    analysis_results['echo_analysis'] = 1
                    detected_method = 'echo'
                    return self.create_result(analyze_path, analyze_audio, sr, compressed, analysis_results, 
                        detected_method, was_embedded, embedded_bits, segment_size, total_capacity, first_capacity)
            except Exception as e:
                print(f"Ошибка Echo анализа: {e}")

        # широкополосное кодирование
        if not compressed and self.settings['enable_dsss']:
            try:
                dsss_result = self.dsss_detector.analyze(signal = analyze_audio)
                if dsss_result == 1:
                    analysis_results['dsss_analysis'] = 1
                    detected_method = 'dsss'
                    return self.create_result(analyze_path, analyze_audio, sr, compressed, analysis_results, detected_method, was_embedded, 
                        embedded_bits, segment_size, total_capacity, first_capacity)
            except Exception as e:
                print(f"Ошибка DSSS анализа: {e}")
        
        # ничего не обнаружено
        return self.create_result(analyze_path, analyze_audio, sr, compressed, analysis_results, None,
            was_embedded, embedded_bits, segment_size, total_capacity, first_capacity)
    
    def create_result(self, audio_path, audio_data, sr, compressed, analysis_results, detected_method, was_embedded, 
        embedded_bits, segment_size, total_capacity, first_capacity):
        metadata = get_metadata(audio_path)
        
        return {
            'filename': audio_path,
            'is_compressed': compressed,
            'sample_rate': sr,
            'duration': len(audio_data) / sr if sr and len(audio_data) > 0 else 0,
            'metadata': metadata,
            'detected_method': detected_method,
            'is_stego': detected_method is not None,
            'was_embedded': was_embedded,
            'embedded_bits': embedded_bits,
            'segment_size': segment_size,
            'total_capacity': total_capacity,
            'first_capacity': first_capacity,
            'analysis_results': analysis_results
        }


def main():
    parser = argparse.ArgumentParser(description = 'Audio Steganography Detector')
    parser.add_argument('input', type = str, help = 'Путь к аудиофайлу или папке с файлами')
    parser.add_argument('--embed', type = float, default = 0, help = 'Доля для тестового встраивания (0-1)')    
    parser.add_argument('--recursive', action = 'store_true', help = 'Рекурсивный обход папки')
    parser.add_argument('--output', type = str, default = None, help = 'Сохранение результатов анализа в JSON')
    parser.add_argument('--save-stego', action = 'store_true', help = 'Сохранять стего-файлы в папку')
    parser.add_argument('--output-dir', type = str, default = 'stego_output', help = 'Папка для сохранения стего-файлов (по умолчанию stego_output)')
    
    args = parser.parse_args()
    detector = AudioStegDetector()
    detector.args = args
    start = time.time()

    # один файл
    if os.path.isfile(args.input):
        if args.embed > 0:
            print(f"Тестовое встраивание: {args.embed*100:.0f}% (метод: {EMBED_METHOD})")
            if args.save_stego:
                ext = os.path.splitext(args.input)[1].lower()
                subfolder = 'mp3' if ext == '.mp3' else ('ogg' if ext == '.ogg' else 'wav')
                print(f"Сохранение файлов в: {args.output_dir}/{subfolder}")
        
        result = detector.process_file(args.input, args.embed, EMBED_METHOD)
        if result:
            print(f"Длительность: {result['duration']:.2f} сек")
            if result['was_embedded']:
                print(f"Тест. встраивание: {result['embedded_bits']} бит")
            if result['is_stego']:
                print(f"Обнаружено, метод: {result['detected_method'].upper()}")
            else:
                print(f"Нe обнаружено")
            
            if args.output:
                with open(args.output, 'w', encoding = 'utf-8') as f:
                    json.dump(result, f, indent = 2, ensure_ascii = False, default = str)
                print(f"\nРезультаты сохранены: {args.output}")
        else:
            print(f"\nОшибка анализа")
    
    # папка
    elif os.path.isdir(args.input):
        if args.embed > 0:
            print(f"Тестовое встраивание: {args.embed*100:.0f}% (метод: {EMBED_METHOD})")
            if args.save_stego:
                print(f"Сохранение файлов в: {args.output_dir}")
        
        extensions = ('.wav', '.mp3', '.flac', '.ogg')
        audio_files = []
        
        if args.recursive:
            for root, _, files in os.walk(args.input):
                for f in files:
                    if f.lower().endswith(extensions):
                        audio_files.append(os.path.join(root, f))
        else:
            for f in os.listdir(args.input):
                if f.lower().endswith(extensions):
                    audio_files.append(os.path.join(args.input, f))
        
        audio_files.sort()
        print(f"Найдено файлов: {len(audio_files)}")
        
        results = []
        stego_count = 0
        clean_count = 0
        error_count = 0
        method_stats = {}
        
        for i, file_path in enumerate(audio_files, 1):
            file_name = os.path.basename(file_path)
            print(f"\n[{i}/{len(audio_files)}] {file_name}")
            result = detector.process_file(file_path, args.embed, EMBED_METHOD)
            
            if result:
                results.append(result)
                if result['is_stego']:
                    stego_count += 1
                    method = result['detected_method']
                    method_stats[method] = method_stats.get(method, 0) + 1
                    print(f"СТЕГО (метод: {method})")
                else:
                    clean_count += 1
                    print(f"ЧИСТЫЙ")
            else:
                error_count += 1
                print(f"Ошибка")
        
        end = time.time()
        duration = end - start
        
        # Cтатистика
        print(f"\nВсего файлов: {len(audio_files)}")
        print(f"Успешно: {len(results)}")
        print(f"Ошибок: {error_count}")
        
        if len(results) > 0:
            print(f"\nСтего: {stego_count} ({stego_count/len(results)*100:.1f}%)")
            print(f"Чистые: {clean_count} ({clean_count/len(results)*100:.1f}%)")
        
        if method_stats:
            print(f"\nПО МЕТОДАМ ОБНАРУЖЕНИЯ:")
            for method, count in sorted(method_stats.items(), key=lambda x: -x[1]):
                print(f"{method.upper()}: {count} ({count/stego_count*100:.1f}%)")
        
        print(f'Длительность: {duration:.2f} с')

        if args.output and results:
            clean_results = []
            for r in results:
                clean_r = {}
                for k, v in r.items():
                    if k == 'metadata':
                        if isinstance(v, dict):
                            clean_r[k] = {mk: str(mv)[:200] for mk, mv in v.items()}
                        else:
                            clean_r[k] = str(v)[:200]
                    elif isinstance(v, (str, int, float, bool, type(None))):
                        clean_r[k] = v
                    else:
                        clean_r[k] = str(v)
                clean_results.append(clean_r)
            
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(clean_results, f, indent=2, ensure_ascii=False, default=str)
            print(f"\nРезультаты сохранены: {args.output}")
    
    else:
        print(f"Путь не существует: {args.input}")
        sys.exit(1)

if __name__ == "__main__":
    main()