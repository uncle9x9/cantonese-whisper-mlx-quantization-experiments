#!/usr/bin/env python3
"""Run one local MLX Whisper model against a frozen Inflow evaluation manifest."""
import argparse
import ctypes as ct
import ctypes.util
import functools
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import sys
import time
import unicodedata
import wave

METRICS = ('raw', 'clean', 'common_script')


def edit_distance(reference, hypothesis):
    if len(hypothesis) > len(reference):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for row, ref_char in enumerate(reference, 1):
        current = [row]
        for col, hyp_char in enumerate(hypothesis, 1):
            current.append(min(previous[col] + 1, current[-1] + 1,
                               previous[col - 1] + (ref_char != hyp_char)))
        previous = current
    return previous[-1]


def clean(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    return ''.join(char for char in text
                   if not char.isspace() and not unicodedata.category(char).startswith('P'))


@functools.lru_cache(maxsize=1)
def script_converter():
    """Use the platform's ICU transform through CoreFoundation, when available."""
    library = ctypes.util.find_library('CoreFoundation')
    if library is None:
        return None
    cf = ct.CDLL(library)
    cf.CFStringCreateWithBytes.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_long, ct.c_uint32, ct.c_bool]
    cf.CFStringCreateWithBytes.restype = ct.c_void_p
    cf.CFStringCreateMutableCopy.argtypes = [ct.c_void_p, ct.c_long, ct.c_void_p]
    cf.CFStringCreateMutableCopy.restype = ct.c_void_p
    cf.CFStringTransform.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_bool]
    cf.CFStringTransform.restype = ct.c_bool
    cf.CFStringGetLength.argtypes = [ct.c_void_p]
    cf.CFStringGetLength.restype = ct.c_long
    cf.CFStringGetCString.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_long, ct.c_uint32]
    cf.CFStringGetCString.restype = ct.c_bool
    cf.CFRelease.argtypes = [ct.c_void_p]
    cf.CFRelease.restype = None
    utf8 = 0x08000100

    def cf_string(data):
        return cf.CFStringCreateWithBytes(None, data, len(data), utf8, False)

    def convert(text):
        if not text:
            return text
        original = cf_string(text.encode('utf-8'))
        mutable = transform = None
        try:
            if not original:
                raise ValueError('CoreFoundation could not read the text')
            mutable = cf.CFStringCreateMutableCopy(None, 0, original)
            transform = cf_string(b'Traditional-Simplified')
            if not (mutable and transform and cf.CFStringTransform(mutable, None, transform, False)):
                raise ValueError('CoreFoundation script conversion failed')
            buffer = ct.create_string_buffer(4 * cf.CFStringGetLength(mutable) + 1)
            if not cf.CFStringGetCString(mutable, buffer, len(buffer), utf8):
                raise ValueError('CoreFoundation could not return the converted text')
            return buffer.value.decode('utf-8')
        finally:
            for pointer in (transform, mutable, original):
                if pointer:
                    cf.CFRelease(pointer)

    return convert


def score(reference, hypothesis, to_simplified=None):
    if reference is None:
        return None
    clean_ref, clean_hyp = clean(reference), clean(hypothesis)
    views = {'raw': (reference, hypothesis), 'clean': (clean_ref, clean_hyp)}
    if to_simplified is not None:
        views['common_script'] = (to_simplified(clean_ref), to_simplified(clean_hyp))
    result = {}
    for name, (ref, hyp) in views.items():
        edits = edit_distance(ref, hyp)
        result[name] = {'edits': edits, 'reference_chars': len(ref),
                        'hypothesis_chars': len(hyp), 'cer': edits / len(ref) if ref else None}
    denominator = max(len(clean_ref), len(clean_hyp))
    result['clean_symmetric'] = result['clean']['edits'] / denominator if denominator else 0.0
    return result


def aggregate(rows):
    result = {}
    for metric in METRICS:
        items = [row['scores'][metric] for row in rows
                 if row.get('scores') and metric in row['scores']]
        if not items:
            continue
        edits = sum(item['edits'] for item in items)
        chars = sum(item['reference_chars'] for item in items)
        rates = [item['cer'] for item in items if item['cer'] is not None]
        result[metric] = {'scored': len(items), 'edits': edits, 'reference_chars': chars,
                          'corpus_cer': edits / chars if chars else None,
                          'macro_cer': statistics.mean(rates) if rates else None}
    symmetric = [row['scores']['clean_symmetric'] for row in rows if row.get('scores')]
    if symmetric:
        result['clean_symmetric_macro'] = statistics.mean(symmetric)
    return result


def read_audio(path, np, limit_seconds, expected_sha256=None):
    with path.open('rb') as source:
        audio_sha256 = hashlib.file_digest(source, 'sha256').hexdigest()
        if expected_sha256 is not None and audio_sha256 != expected_sha256.lower():
            raise ValueError('audio SHA256 differs from the manifest')
        source.seek(0)
        with wave.open(source, 'rb') as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
                raise ValueError('expected 16 kHz mono signed 16-bit PCM WAV')
            frames = wav.getnframes()
            if not 0 < frames <= 16000 * limit_seconds:
                raise ValueError(f'expected audio duration in (0, {limit_seconds}] seconds')
            data = wav.readframes(frames)
            if len(data) != frames * 2:
                raise ValueError('WAV data ended before its declared frame count')
    return np.frombuffer(data, dtype='<i2').astype(np.float32) / 32768, frames / 16000, audio_sha256


def segment_metadata(segment):
    metadata = {}
    for key in ('start', 'end', 'temperature', 'avg_logprob', 'no_speech_prob', 'compression_ratio'):
        value = segment.get(key)
        metadata[key] = float(value) if value is not None and math.isfinite(value) else None
    metadata['token_count'] = len(segment.get('tokens', []))
    return metadata


def process_memory_reader():
    if sys.platform == 'darwin':
        # ABI defined by <sys/resource.h> struct rusage_info_v0 and <libproc.h>.
        class RusageInfo(ct.Structure):
            _fields_ = [('uuid', ct.c_uint8 * 16)] + [
                (name, ct.c_uint64) for name in
                ('user_time', 'system_time', 'package_idle_wakeups', 'interrupt_wakeups',
                 'pageins', 'wired_size', 'resident_size', 'physical_footprint',
                 'start_time', 'exit_time')
            ]

        try:
            query = ct.CDLL('/usr/lib/libproc.dylib').proc_pid_rusage
        except (OSError, AttributeError):
            pass
        else:
            query.argtypes = [ct.c_int, ct.c_int, ct.c_void_p]
            query.restype = ct.c_int

            def read_darwin():
                info = RusageInfo()
                if query(os.getpid(), 0, ct.byref(info)) != 0:
                    return {}
                return {'process_rss_bytes': info.resident_size,
                        'process_physical_footprint_bytes': info.physical_footprint}

            return read_darwin
    try:
        import psutil
    except ImportError:
        return lambda: {}
    process = psutil.Process()
    return lambda: {'process_rss_bytes': process.memory_info().rss}


def memory_snapshot(mx, read_process_memory):
    snapshot = {'mlx_peak_bytes': mx.get_peak_memory(),
                'mlx_active_bytes': mx.get_active_memory(),
                'mlx_cache_bytes': mx.get_cache_memory(),
                'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * (1 if sys.platform == 'darwin' else 1024)}
    snapshot.update(read_process_memory())
    return snapshot


def aggregate_group(rows):
    successful = [row for row in rows if row.get('error') is None]
    times = sorted(row['decode_seconds'] for row in successful)
    p95 = times[(95 * len(times) + 99) // 100 - 1] if times else None
    audio_seconds = sum(row['duration_seconds'] for row in successful)
    return {
        'samples': len(rows),
        'successful': len(successful),
        'failed': len(rows) - len(successful),
        'empty': sum(not row['text'].strip() for row in successful),
        'possible_sample_len_cap': sum(row['possible_sample_len_cap'] for row in successful),
        'audio_seconds': audio_seconds,
        'decode_seconds': sum(times),
        'median_decode_seconds': statistics.median(times) if times else None,
        'p95_decode_seconds_nearest_rank': p95,
        'real_time_factor': sum(times) / audio_seconds if audio_seconds else None,
        'scores': aggregate(successful),
    }


def write_row(output, row):
    output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    output.flush()


def load_manifest(path):
    data = path.read_bytes()
    samples = json.loads(data)['samples']
    if not samples or len({sample['id'] for sample in samples}) != len(samples):
        raise ValueError('manifest must have at least one sample and unique sample IDs')
    return samples, hashlib.sha256(data).hexdigest()


def run(args, prompt=None):
    start = time.perf_counter()
    model_path = str(args.model.resolve(strict=True))
    samples, manifest_sha256 = load_manifest(args.manifest)
    summary_path = args.output.with_suffix('.summary.json')
    if args.output == summary_path or args.output.exists() or summary_path.exists():
        raise FileExistsError('choose a fresh output path distinct from its summary')
    os.environ.update(HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1')
    convert = script_converter()
    import_start = time.perf_counter()
    import mlx.core as mx
    import numpy as np
    whisper = importlib.import_module('mlx_whisper.transcribe')
    import_seconds = time.perf_counter() - import_start
    read_process_memory = process_memory_reader()
    mx.set_cache_limit(args.cache_mib * 1024**2)
    mx.reset_peak_memory()
    mx.synchronize()
    settings = {
        'language': 'yue',
        'temperature': 0.0,
        'condition_on_previous_text': False,
        'word_timestamps': False,
        'initial_prompt': prompt,
        'fp16': True,
        'sample_len': args.sample_len,
        'task': 'transcribe',
        'verbose': None,
    }
    with args.output.open('x', encoding='utf-8') as output:
        load_start = time.perf_counter()
        model = whisper.ModelHolder.get_model(model_path, mx.float16)
        mx.eval(model.parameters())
        mx.synchronize()
        load_seconds = time.perf_counter() - load_start
        after_load = memory_snapshot(mx, read_process_memory)
        warmup_start = time.perf_counter()
        whisper.transcribe(np.zeros(16000, dtype=np.float32), path_or_hf_repo=model_path, **settings)
        mx.synchronize()
        warmup_seconds = time.perf_counter() - warmup_start
        after_warmup = memory_snapshot(mx, read_process_memory)
        rows = []
        for sample in samples:
            audio_path = (args.manifest.parent / sample['audio']).resolve()
            row = {'id': sample['id'], 'group': sample['group'],
                   'reference_kind': sample['reference_kind'], 'error': None}
            try:
                audio, seconds, sha256 = read_audio(
                    audio_path, np, args.max_audio_seconds, sample.get('audio_sha256'))
                row.update(duration_seconds=seconds, audio_sha256=sha256)
                mx.synchronize()
                decode_start = time.perf_counter()
                result = whisper.transcribe(audio, path_or_hf_repo=model_path, **settings)
                mx.synchronize()
                decode_seconds = time.perf_counter() - decode_start
                text = result['text']
                reference = sample.get('reference')
                segments = [segment_metadata(segment) for segment in result.get('segments', [])]
                row.update(text=text, decode_seconds=decode_seconds,
                           real_time_factor=decode_seconds / seconds, language=result.get('language'),
                           segments=segments,
                           possible_sample_len_cap=any(
                               segment['token_count'] >= args.sample_len for segment in segments),
                           scores=score(reference, text, convert))
            except Exception as exc:
                row['error'] = f'{type(exc).__name__}: {exc}'
            row['memory'] = memory_snapshot(mx, read_process_memory)
            write_row(output, row)
            rows.append(row)
    groups = sorted({row['group'] for row in rows})
    kinds = sorted({row['reference_kind'] for row in rows})
    summary = {
        'model': model_path, 'manifest': str(args.manifest.resolve()), 'manifest_sha256': manifest_sha256,
        'mode': args.mode,
        'settings': {key: value for key, value in settings.items() if key != 'initial_prompt'},
        'initial_prompt_used': prompt is not None,
        'max_audio_seconds': args.max_audio_seconds,
        'mlx_cache_limit_bytes': args.cache_mib * 1024**2,
        'platform': platform.platform(), 'python_version': platform.python_version(),
        'packages': {pkg: importlib.metadata.version(pkg) for pkg in ('mlx', 'mlx-whisper', 'numpy')},
        'common_script_normalization': 'CoreFoundation Traditional-Simplified on both clean strings'
        if convert else None,
        'timing_seconds': {'imports': import_seconds, 'model_load': load_seconds,
                           'warmup': warmup_seconds, 'run_wall': time.perf_counter() - start},
        'memory': {'after_load': after_load, 'after_warmup': after_warmup,
                   'final': memory_snapshot(mx, read_process_memory)},
        'all': aggregate_group(rows),
        'groups': {group: aggregate_group([row for row in rows if row['group'] == group])
                   for group in groups},
        'reference_kinds': {kind: aggregate_group([row for row in rows if row['reference_kind'] == kind])
                            for kind in kinds},
    }
    with summary_path.open('x', encoding='utf-8') as output:
        json.dump(summary, output, ensure_ascii=False, allow_nan=False, indent=2)
        output.write('\n')
    return 1 if summary['all']['failed'] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('prompted', 'unprompted'), required=True)
    parser.add_argument('--prompt-file', type=Path,
                        help='UTF-8 text to supply as the initial prompt in prompted mode')
    parser.add_argument('--cache-mib', type=int, default=64)
    parser.add_argument('--sample-len', type=int, default=224)
    parser.add_argument('--max-audio-seconds', type=int, default=120)
    args = parser.parse_args()
    if args.cache_mib < 0 or not 0 < args.sample_len <= 224 or args.max_audio_seconds <= 0:
        parser.error('expected nonnegative cache MiB, sample length 1–224, and positive max audio seconds')
    if args.mode == 'unprompted' and args.prompt_file is not None:
        parser.error('--prompt-file is only accepted in prompted mode')
    if args.mode == 'prompted' and args.prompt_file is None:
        parser.error('--prompt-file is required in prompted mode')
    prompt = None
    if args.prompt_file is not None:
        try:
            prompt = args.prompt_file.read_text(encoding='utf-8').strip()
        except (OSError, UnicodeError):
            parser.error('--prompt-file must be a readable UTF-8 text file')
        if not prompt:
            parser.error('--prompt-file must contain a nonempty prompt')
    return run(args, prompt)


if __name__ == '__main__':
    raise SystemExit(main())
