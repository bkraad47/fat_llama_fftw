# fat_llama_fftw — Current State

Regenerable factblock snapshot of the codebase, produced by the `review-current-state` skill. Do not hand-edit — regenerate on demand instead. Scope: whole repository (tracked files only; build/venv/cache directories excluded).

## File tree

```
fat_llama_fftw/  (repo root)
├── .github/
│   └── workflows/
│       ├── deploy.yml
│       ├── issue-branch-resolve.yml
│       ├── issue-release-comment.yml
│       └── tests.yml
├── fat_llama_fftw/
│   ├── __init__.py
│   ├── audio_fattener/
│   │   ├── __init__.py
│   │   └── feed.py
│   └── tests/
│       ├── __init__.py
│       └── test_feed.py
├── .gitignore
├── LICENSE
├── Manifest.in
├── README.md
├── analysis.py
├── example.py
├── input_test.flac
├── input_test.mp3
├── output_test.flac
├── requirements.txt
└── setup.py
```

## fat_llama_fftw/audio_fattener/feed.py

### `read_audio(file_path, format) -> tuple`
**File:** fat_llama_fftw/audio_fattener/feed.py:17
**Kind:** function
**Description:** Reads an audio file of the given format, extracts its raw samples as a numpy array, its sample rate, and (where the format's tag library supports it) its bitrate. Falls back to computing an approximate bitrate from sample count/duration for formats mutagen doesn't report bitrate for directly.
**Parameters:**
- `file_path` (`str`): path to the input audio file. Raises `FileNotFoundError` if it doesn't exist.
- `format` (`str`): one of `'mp3'`, `'flac'`, `'ogg'`, `'wav'` (or any other format `pydub`/ffmpeg can decode, with an approximated bitrate).
**Returns:** `(sample_rate: int, samples: np.ndarray, bitrate: float|None, audio: pydub.AudioSegment)` — `samples` is reshaped to `(-1, 2)` for stereo input.
**Usage:**
```python
sample_rate, samples, bitrate, audio = read_audio('input_test.mp3', format='mp3')
```

### `write_audio(file_path, sample_rate, data, format) -> None`
**File:** fat_llama_fftw/audio_fattener/feed.py:47
**Kind:** function
**Description:** Writes float32 PCM sample data to a FLAC or WAV file via `soundfile`, at 24-bit depth. Raises `ValueError` for any other target format.
**Parameters:**
- `file_path` (`str`): output file path.
- `sample_rate` (`int`): sample rate to write.
- `data` (`np.ndarray`): sample data (any dtype; cast to `float32`).
- `format` (`str`): `'flac'` or `'wav'`.
**Returns:** `None`.
**Usage:**
```python
write_audio('output_test.flac', 44100, upscaled_samples, 'flac')
```

### `new_interpolation_algorithm(data, upscale_factor) -> np.ndarray`
**File:** fat_llama_fftw/audio_fattener/feed.py:55
**Kind:** function
**Description:** Zero-order-hold upsampling: repeats each input sample `upscale_factor` times in a plain Python double loop to expand the signal length, ahead of IST refinement. This is the interpolation step named in README's Algorithm Explanation step 3.
**Parameters:**
- `data` (`np.ndarray`): 1-D channel samples.
- `upscale_factor` (`int`): how many times to repeat each sample.
**Returns:** `np.ndarray` (`float32`), length `len(data) * upscale_factor`.
**Usage:**
```python
expanded = new_interpolation_algorithm(channel, upscale_factor=4)
```

### `initialize_ist(data, threshold) -> np.ndarray`
**File:** fat_llama_fftw/audio_fattener/feed.py:68
**Kind:** function
**Description:** First IST step: zeroes out every sample whose magnitude is at/below `threshold`, keeping only values already above it as the starting point for the FFT-domain refinement loop.
**Parameters:**
- `data` (`np.ndarray`): input samples (typically the interpolated/expanded channel).
- `threshold` (`float`): magnitude cutoff.
**Returns:** `np.ndarray` — same shape as `data`, thresholded.
**Usage:**
```python
data_thres = initialize_ist(expanded_channel, threshold=0.6)
```

### `perform_ist_iteration(data_thres, threshold) -> np.ndarray`
**File:** fat_llama_fftw/audio_fattener/feed.py:73
**Kind:** function
**Description:** One IST refinement pass: FFT the current estimate (via `pyfftw.interfaces.numpy_fft`), zero out frequency bins at/below `threshold`, inverse-FFT back to the time domain, and keep the real part. This is the core FFT/IST step described in README's "Why FFT and IST?" and `.claude/rules/project-mission.md`.
**Parameters:**
- `data_thres` (`np.ndarray`): current time-domain estimate.
- `threshold` (`float`): frequency-domain magnitude cutoff.
**Returns:** `np.ndarray` — refined time-domain estimate (real-valued).
**Usage:**
```python
data_thres = perform_ist_iteration(data_thres, threshold=0.6)
```

### `iterative_soft_thresholding(data, max_iter, threshold) -> np.ndarray`
**File:** fat_llama_fftw/audio_fattener/feed.py:80
**Kind:** function
**Description:** Runs `perform_ist_iteration` `max_iter` times via a `ThreadPoolExecutor`, submitting all `max_iter` iterations against the *same* initial `data_thres` concurrently and keeping whichever result completes last in `as_completed` order — note this means iterations are not chained sequentially on each other's output; each worker independently re-derives from `initialize_ist`'s output, and the final returned estimate is simply whichever future happened to finish last, not the result of `max_iter` sequential refinements.
**Parameters:**
- `data` (`np.ndarray`): input samples for `initialize_ist`.
- `max_iter` (`int`): number of IST passes to submit.
- `threshold` (`float`): magnitude/frequency cutoff used throughout.
**Returns:** `np.ndarray` — the IST "changes" to add back onto the interpolated signal (see `upscale_channels`).
**Usage:**
```python
ist_changes = iterative_soft_thresholding(expanded_channel, max_iter=300, threshold=0.6)
```

### `upscale_channels(channels, upscale_factor, max_iter, threshold) -> np.ndarray`
**File:** fat_llama_fftw/audio_fattener/feed.py:90
**Kind:** function
**Description:** Runs interpolation + IST per channel (looping over `channels.T`), adding each channel's IST result back onto its interpolated version, then stacks the processed channels back into a single 2-D array.
**Parameters:**
- `channels` (`np.ndarray`): shape `(n_samples, n_channels)`.
- `upscale_factor` (`int`): passed to `new_interpolation_algorithm`.
- `max_iter` (`int`): passed to `iterative_soft_thresholding`.
- `threshold` (`float`): passed to both `initialize_ist`/IST iterations.
**Returns:** `np.ndarray`, shape `(n_samples * upscale_factor, n_channels)`.
**Usage:**
```python
upscaled = upscale_channels(channels, upscale_factor=4, max_iter=300, threshold=0.6)
```

### `normalize_signal(signal) -> np.ndarray`
**File:** fat_llama_fftw/audio_fattener/feed.py:104
**Kind:** function
**Description:** Peak-normalizes a signal to `[-1, 1]` by dividing by its max absolute value.
**Parameters:**
- `signal` (`np.ndarray`): input samples.
**Returns:** `np.ndarray`, same shape, peak-normalized.
**Usage:**
```python
normalized = normalize_signal(channel)
```

### `upscale(input_file_path, output_file_path, source_format, target_format='flac', max_iterations=800, threshold_value=0.6, target_bitrate_kbps=1411) -> None`
**File:** fat_llama_fftw/audio_fattener/feed.py:107
**Kind:** function
**Description:** The package's public entry point (README's documented API). Reads the source file, computes an upscale factor from the target vs. original bitrate, runs interpolation+IST per channel via `upscale_channels`, auto-scales each channel's amplitude back to the original's peak level, normalizes, and writes the result at `sample_rate * upscale_factor`. Note: this signature has no `toggle_*` flags (normalize/autoscale/adaptive-filter) and no LMS adaptive filter step — auto-scaling and normalization always run unconditionally as the last two stages.
**Parameters:**
- `input_file_path` (`str`): source audio file path.
- `output_file_path` (`str`): destination file path.
- `source_format` (`str`): `'mp3'`, `'wav'`, `'ogg'`, or `'flac'`.
- `target_format` (`str`): `'flac'` or `'wav'`. Default `'flac'`.
- `max_iterations` (`int`): IST iteration count. Default `800`.
- `threshold_value` (`float`): IST threshold. Default `0.6`.
- `target_bitrate_kbps` (`int`): target bitrate; validated against a per-format range (`flac`: 800-1411, `wav`: 800-6444). Default `1411`.
**Returns:** `None` (writes the output file as a side effect).
**Usage:**
```python
# from example.py
from fat_llama_fftw.audio_fattener.feed import upscale

upscale(
    input_file_path='input_test.mp3',
    output_file_path='output_test.flac',
    source_format='mp3',
    target_format='flac',
    max_iterations=1000,
    threshold_value=0.6,
    target_bitrate_kbps=1400
)
```

## fat_llama_fftw/tests/test_feed.py

### `TestFeed`
**File:** fat_llama_fftw/tests/test_feed.py:13
**Kind:** class
**Description:** `unittest.TestCase` covering `feed.py`'s helper functions with mocked I/O (`read_audio`, `write_audio`) and small synthetic-array checks (`new_interpolation_algorithm`, `initialize_ist`, `upscale_channels`, `normalize_signal`). No test exercises the full `upscale()` entry point or asserts on real audio content/quality — see Open items below.
**Usage:**
```python
python -m unittest discover -s fat_llama_fftw/tests
```

### `TestFeed.test_read_audio(self, mock_exists, mock_mp3, mock_from_file)`
**File:** fat_llama_fftw/tests/test_feed.py:18
**Kind:** method
**Description:** Mocks `AudioSegment.from_file` and `MP3` to verify `read_audio` returns the mocked sample rate and correctly reshapes interleaved stereo samples into `(n, 2)`.
**Returns:** `None` (assertion-based).
**Usage:** run via `unittest`/`pytest`, not called directly.

### `TestFeed.test_write_audio(self, mock_write)`
**File:** fat_llama_fftw/tests/test_feed.py:30
**Kind:** method
**Description:** Mocks `soundfile.write` to verify `write_audio` calls it with the right path, sample rate, data, format, and `PCM_24` subtype for a FLAC target.
**Returns:** `None` (assertion-based).

### `TestFeed.test_new_interpolation_algorithm(self)`
**File:** fat_llama_fftw/tests/test_feed.py:42
**Kind:** method
**Description:** Verifies the zero-order-hold repeat behavior on a 4-sample array with `upscale_factor=2`.
**Returns:** `None` (assertion-based).

### `TestFeed.test_initialize_ist(self)`
**File:** fat_llama_fftw/tests/test_feed.py:49
**Kind:** method
**Description:** Verifies thresholding zeroes out samples at/below the threshold.
**Returns:** `None` (assertion-based).

### `TestFeed.test_upscale_channels(self)`
**File:** fat_llama_fftw/tests/test_feed.py:56
**Kind:** method
**Description:** Verifies `upscale_channels`' output shape only (`(4, 2)` for a 2x2 input at `upscale_factor=2`) — does not assert on the actual sample values/content produced.
**Returns:** `None` (assertion-based).

### `TestFeed.test_normalize_signal(self)`
**File:** fat_llama_fftw/tests/test_feed.py:64
**Kind:** method
**Description:** Verifies peak-normalization divides by the max absolute value.
**Returns:** `None` (assertion-based).

## example.py

### Module-level script
**File:** example.py:1
**Kind:** script (no functions/classes)
**Description:** The README-documented usage example — calls `upscale()` once against the repo-root `input_test.mp3` → `output_test.flac`, with `max_iterations=1000` (README's own documented example value; note this differs from `.claude/agents/rules/audio-quality.md`'s pinned baseline of `max_iterations=300` — the rules file's baseline is intentionally lower for a faster, comparable CI run, not a copy of this example).
**Usage:**
```python
python example.py
```

## analysis.py

### `read_mp3(file_path) -> tuple`
**File:** analysis.py:8
**Kind:** function
**Description:** Reads an MP3 via `pydub`, downmixes stereo to mono by averaging channels.
**Parameters:**
- `file_path` (`str`): path to the MP3.
**Returns:** `(data: np.ndarray, sample_rate: int)`.
**Usage:**
```python
mp3, sr = read_mp3('input_test.mp3')
```

### `read_flac(file_path) -> tuple`
**File:** analysis.py:16
**Kind:** function
**Description:** Reads a FLAC via `soundfile`, downmixes stereo to mono by averaging channels.
**Parameters:**
- `file_path` (`str`): path to the FLAC.
**Returns:** `(data: np.ndarray, sample_rate: int)`.
**Usage:**
```python
flac, sr = read_flac('output_test.flac')
```

### `normalize(signal) -> np.ndarray`
**File:** analysis.py:22
**Kind:** function
**Description:** Peak-normalizes a signal to `[-1, 1]` — duplicate of `feed.py`'s `normalize_signal`, kept as a separate copy in this analysis script.
**Parameters:**
- `signal` (`np.ndarray`): input samples.
**Returns:** `np.ndarray`, peak-normalized.

### `compare_signals(mp3, flac, sample_rate) -> None`
**File:** analysis.py:25
**Kind:** function
**Description:** Produces a battery of comparison plots/metrics between an MP3 and a FLAC signal: waveform overlay, difference signal, MSE, spectrogram comparison (this is the routine `.claude/agents/rules/audio-quality.md` says to reuse for the spectrogram comparison image), cross-correlation, and frequency-domain (FFT) comparison. **Note: imports `cupy as cp` at module level (analysis.py:6) and uses it for the cross-correlation and FFT steps (`cp.correlate`, `cp.fft.fft`) — this is a CUDA/CuPy dependency in this CPU/FFTW-only package** (per `.claude/rules/project-mission.md`'s CPU-only focus). On a machine without a CUDA-capable GPU/CuPy installed, importing this module at all will fail, which would break anything relying on its `read_mp3`/`read_flac`/`compare_signals` functions (including the spectrogram-comparison step `audio-quality.md` assigns to `audio-quality-checker`). Flagged here as a factual observation only — fixing it is `generate-code`'s job, not this skill's.
**Parameters:**
- `mp3` (`np.ndarray`): mono MP3 samples.
- `flac` (`np.ndarray`): mono FLAC samples.
- `sample_rate` (`int`): shared sample rate.
**Returns:** `None` (shows plots, prints MSE).
**Usage:**
```python
# illustrative — see analysis.py's own __main__ block
mp3, sr1 = read_mp3('input_test.mp3')
flac, sr2 = read_flac('output_test.flac')
compare_signals(mp3, flac, sr1)
```

## Open items / observations

- No test exercises `upscale()` end-to-end, and `test_upscale_channels` only checks output shape, not content — matches `.claude/agents/rules/audio-quality.md`'s own note that test coherence needs checking.
- `analysis.py` still imports `cupy` (GPU-only) despite this package (`fat_llama_fftw`) being the CPU/FFTW-only variant — a latent environment-breaking dependency in source that predates and is separate from the `.claude/`/`.github/workflows/` config already updated to CPU/FFTW.
- `upscale()`'s actual signature has no `toggle_normalize`/`toggle_autoscale`/`toggle_adaptive_filter` flags and `feed.py` has no `lms_filter` function — `.claude/agents/rules/audio-quality.md`'s pinned baseline config and runtime-estimate section reference these anyway (apparently carried over from the CUDA sibling package's more elaborate `feed.py`). This is a `.claude/` rules-file inaccuracy, out of scope for this skill (and for `iterate-fat-llama`'s own write scope) to fix directly — flagged here for a human or a future permitted edit to reconcile.
- `iterative_soft_thresholding`'s `ThreadPoolExecutor` submits all `max_iter` iterations concurrently against the same starting `data_thres` rather than chaining them sequentially (see its factblock above) — worth double-checking against the intended IST semantics if audio quality/performance work touches this function.
