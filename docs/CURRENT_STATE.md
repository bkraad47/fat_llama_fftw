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
├── docs/
│   ├── CURRENT_STATE.md
│   └── images/
│       └── spectrogram_comparison.png
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
**File:** fat_llama_fftw/audio_fattener/feed.py:18
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
**File:** fat_llama_fftw/audio_fattener/feed.py:49
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
**File:** fat_llama_fftw/audio_fattener/feed.py:60
**Kind:** function
**Description:** Zero-order-hold upsampling: repeats each input sample `upscale_factor` times in a plain Python double loop to expand the signal length, ahead of IST refinement. This is the interpolation step named in README's Algorithm Explanation step 3. Being zero-order-hold, it images the original spectrum above the original Nyquist frequency — that's why `upscale()` now runs `apply_nyquist_cutoff` as its final stage (see below).
**Parameters:**
- `data` (`np.ndarray`): 1-D channel samples.
- `upscale_factor` (`int`): how many times to repeat each sample.
**Returns:** `np.ndarray` (`float32`), length `len(data) * upscale_factor`.
**Usage:**
```python
expanded = new_interpolation_algorithm(channel, upscale_factor=4)
```

### `initialize_ist(data, threshold) -> np.ndarray`
**File:** fat_llama_fftw/audio_fattener/feed.py:74
**Kind:** function
**Description:** First IST step: zeroes out every sample whose magnitude is at/below `threshold * peak(data)`, keeping only values already above that as the starting point for the FFT-domain refinement loop. Cycle-2 fix: `threshold` is now a fraction (0-1) of the array's own peak magnitude rather than an absolute cutoff — the old absolute comparison kept ~99.8% of raw int16-scale samples at the default `0.6`, making IST a near-identity with no added detail. Returns an all-zero array (via an explicit `peak == 0` guard) for a silent input.
**Parameters:**
- `data` (`np.ndarray`): input samples (typically the interpolated/expanded channel).
- `threshold` (`float`): fraction of `data`'s own peak magnitude to use as the cutoff.
**Returns:** `np.ndarray` — same shape as `data`, thresholded.
**Usage:**
```python
data_thres = initialize_ist(expanded_channel, threshold=0.6)
```

### `perform_ist_iteration(data_thres, threshold) -> np.ndarray`
**File:** fat_llama_fftw/audio_fattener/feed.py:93
**Kind:** function
**Description:** One IST refinement pass: FFT the current estimate (via `pyfftw.interfaces.numpy_fft`, cached via `pyfftw.interfaces.cache.enable()` at module load), zero out frequency bins at/below `threshold * peak(|FFT|)`, **always also excludes the DC bin (`mask[0] = False`) regardless of whether it clears the threshold** (cycle-3 fix — the DC bin carries only a constant offset, never spectral detail; an asymmetric transient can make it the single loudest raw bin, and keeping it injected a literal DC offset across the whole output, the root cause of a cycle-2 regression), inverse-FFT back to the time domain, and keep the real part.
**Parameters:**
- `data_thres` (`np.ndarray`): current time-domain estimate (one block's worth, when called via the blocked path below).
- `threshold` (`float`): fraction of the current FFT magnitude's own peak to use as the cutoff.
**Returns:** `np.ndarray` — refined time-domain estimate (real-valued).
**Usage:**
```python
data_thres = perform_ist_iteration(data_thres, threshold=0.6)
```

### `_ist_chain(data, max_iter, threshold, convergence_tol) -> np.ndarray`
**File:** fat_llama_fftw/audio_fattener/feed.py:127
**Kind:** function (private helper)
**Description:** New in cycle 3 — factored out of `iterative_soft_thresholding` unchanged: runs `perform_ist_iteration` in a genuine sequential chain (cycle-1 fix — each pass feeds its output into the next; previously a non-chaining `ThreadPoolExecutor` made this equivalent to a single iteration regardless of `max_iter`), with an early exit once a pass changes the estimate by less than `convergence_tol` relative to its own scale (hard-threshold IST is a fixed-point projection, so further passes past that point are provably no-ops). Called either once (short/whole-signal path) or once per block (long-signal blocked path) by `iterative_soft_thresholding`.
**Parameters:**
- `data` (`np.ndarray`): input samples (a whole signal or a single block).
- `max_iter` (`int`): upper bound on passes.
- `threshold` (`float`): magnitude/frequency cutoff.
- `convergence_tol` (`float`): relative early-exit tolerance.
**Returns:** `np.ndarray` — the converged (or `max_iter`-capped) IST estimate.

### `iterative_soft_thresholding(data, max_iter, threshold, convergence_tol=1e-6, block_size=8192) -> np.ndarray`
**File:** fat_llama_fftw/audio_fattener/feed.py:160
**Kind:** function
**Description:** Cycle-3 rewrite, cycle-4 WOLA fix. For `len(data) <= block_size`, behaves exactly as before (a single `_ist_chain` call over the whole array). Above `block_size`, splits the signal into 50%-overlapping **sqrt-Hann**-windowed blocks (STFT-style), runs `_ist_chain` independently on each block (so the peak-relative threshold reflects each block's own *local* dynamics rather than one whole-file global peak, fixing a cycle-2 regression), and reconstructs via **WOLA** (weighted overlap-add): the same sqrt-Hann window is applied a second time to each block's *output* before accumulation (synthesis, not just analysis), with the accumulation weight summing `window**2`. Cycle-4 fix: `_ist_chain` is a nonlinear FFT hard-threshold projection that does not preserve an analysis-windowed frame's edge taper on the way out (measured: edges tapered to ~1e-4 of peak going in, back up to ~8-11% of peak coming out) — analysis-only windowing (COLA, cycle 3) let that untapered edge content enter the overlap-add sum at full weight at every hop boundary, producing a periodic broadband artifact at the block-hop rate (~75 Hz for a real file). Synthesis windowing forces each block's own contribution back toward ~0 at its edges regardless of what the nonlinear operation did there, eliminating the discontinuity; sqrt-Hann (not plain Hann) on both sides is required so the analysis×synthesis window product itself still sums to a flat constant at 50% hop. Blocking/WOLA does not reintroduce the cycle-1 non-chaining bug or lose its convergence early-exit — each block still runs its own full chained/early-exiting `_ist_chain`. **Cycle-5 fix:** the synthesis-window multiplication itself was found to leak a small DC/near-DC component back into each windowed frame even though `perform_ist_iteration` already excludes the DC bin (the window's spectrum isn't a perfect delta at 0Hz, so windowing is a convolution that reintroduces some near-DC content) — each windowed frame's own mean is now explicitly re-subtracted before accumulation, removing that reintroduced term without affecting any other frequency or reopening the cycle-4 edge-taper regression. Measured on real `input_test.mp3` programme material: relative DC dropped from 9.59e-3 (cycle 4) to 7.29e-8 (cycle 5).
**Parameters:**
- `data` (`np.ndarray`): input samples (typically the interpolated/expanded channel).
- `max_iter` (`int`): upper bound on IST passes per block/chain.
- `threshold` (`float`): fraction-of-peak magnitude cutoff, applied per-block once above `block_size`.
- `convergence_tol` (`float`): relative early-exit tolerance. Default `1e-6`.
- `block_size` (`int`): blocking threshold/window length; not exposed on `upscale()`'s public signature. Default `8192`.
**Returns:** `np.ndarray` — the IST "changes" to add back onto the interpolated signal (see `upscale_channels`).
**Usage:**
```python
ist_changes = iterative_soft_thresholding(expanded_channel, max_iter=300, threshold=0.6)
```

### `upscale_channels(channels, upscale_factor, max_iter, threshold) -> np.ndarray`
**File:** fat_llama_fftw/audio_fattener/feed.py:122
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
**File:** fat_llama_fftw/audio_fattener/feed.py:139
**Kind:** function
**Description:** Peak-normalizes a signal to `[-1, 1]` by dividing by its max absolute value.
**Parameters:**
- `signal` (`np.ndarray`): input samples.
**Returns:** `np.ndarray`, same shape, peak-normalized.
**Usage:**
```python
normalized = normalize_signal(channel)
```

### `apply_nyquist_cutoff(signal, sample_rate, original_nyquist) -> np.ndarray`
**File:** fat_llama_fftw/audio_fattener/feed.py:275
**Kind:** function
**Description:** New in cycle 1 — zeroes every FFT bin above `original_nyquist` (via `pyfftw` `rfft`/`irfft`) and returns the filtered signal, enforcing `.claude/rules/project-mission.md`'s hard constraint. Cycle-5 reorder: now runs after amplitude auto-scaling but *before* `upscale()`'s final `normalize_signal` pass, not strictly last — a brick-wall FFT filter can overshoot its own input's peak (Gibbs-phenomenon ripple), so normalizing must come after the cutoff to guarantee the written signal never exceeds full scale (cycle-5 fix for ~4e-6 clipping fraction that occurred when normalization ran first). Measured: peak above-Nyquist content on `input_test.mp3`'s output dropped from -52.1 dB to -145.7 dB relative to the output's own peak (cycle 1).
**Parameters:**
- `signal` (`np.ndarray`): 1-D channel samples at the *upscaled* sample rate.
- `sample_rate` (`int`): the upscaled signal's own sample rate (used to compute FFT bin frequencies).
- `original_nyquist` (`float`): the cutoff — the original source file's sample rate / 2.
**Returns:** `np.ndarray` (`float32`), same length as `signal`, with all content above `original_nyquist` removed.
**Usage:**
```python
filtered = apply_nyquist_cutoff(upscaled_channel, new_sample_rate, original_sample_rate / 2.0)
```

### `upscale(input_file_path, output_file_path, source_format, target_format='flac', max_iterations=800, threshold_value=0.6, target_bitrate_kbps=1411) -> None`
**File:** fat_llama_fftw/audio_fattener/feed.py:296
**Kind:** function
**Description:** The package's public entry point (README's documented API). Reads the source file, computes an upscale factor from the target vs. original bitrate, runs interpolation+IST per channel via `upscale_channels`, auto-scales each channel's amplitude back to the original's peak level, then (cycle-5 reorder) runs `apply_nyquist_cutoff` per channel, and *finally* normalizes — normalization moved after the cutoff (previously it ran before, i.e. last) because the cutoff's brick-wall filtering can overshoot the already-normalized peak and get silently clipped on write; putting normalization strictly last guarantees the written signal peaks at exactly full scale with no clipping. Note: this signature still has no `toggle_*` flags (normalize/autoscale/adaptive-filter) and no LMS adaptive filter step — auto-scaling, the Nyquist cutoff, and normalization always run unconditionally as the last stages, in that order.
**Parameters:**
- `input_file_path` (`str`): source audio file path.
- `output_file_path` (`str`): destination file path.
- `source_format` (`str`): `'mp3'`, `'wav'`, `'ogg'`, or `'flac'`.
- `target_format` (`str`): `'flac'` or `'wav'`. Default `'flac'`.
- `max_iterations` (`int`): IST iteration upper bound (see `iterative_soft_thresholding`'s early-exit behavior). Default `800`.
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
**File:** fat_llama_fftw/tests/test_feed.py:16
**Kind:** class
**Description:** `unittest.TestCase` covering `feed.py`'s helper functions. Cycle 1 strengthened several previously shape-only assertions and added coverage for the IST chaining/convergence fix and the new `apply_nyquist_cutoff`. Still no test exercises the full `upscale()` entry point end-to-end (see Open items).
**Usage:**
```python
python -m unittest discover -s fat_llama_fftw/tests
```

### `TestFeed.test_read_audio(self, mock_exists, mock_mp3, mock_from_file)`
**File:** fat_llama_fftw/tests/test_feed.py:21
**Kind:** method
**Description:** Mocks `AudioSegment.from_file` and `MP3` to verify `read_audio` returns the mocked sample rate, correctly reshapes interleaved stereo samples into `(n, 2)`, and (cycle-1 strengthening) also asserts on the returned `bitrate`, `audio` object identity, sample content/ordering, and implied duration — previously these return values were discarded without assertions.
**Returns:** `None` (assertion-based).

### `TestFeed.test_write_audio(self, mock_write)`
**File:** fat_llama_fftw/tests/test_feed.py:43
**Kind:** method
**Description:** Mocks `soundfile.write` to verify `write_audio` calls it with the right path, sample rate, data, format, and `PCM_24` subtype for a FLAC target.
**Returns:** `None` (assertion-based).

### `TestFeed.test_new_interpolation_algorithm(self)`
**File:** fat_llama_fftw/tests/test_feed.py:55
**Kind:** method
**Description:** Verifies the zero-order-hold repeat behavior on a 4-sample array with `upscale_factor=2`, plus (cycle-1 strengthening) output dtype and length.
**Returns:** `None` (assertion-based).

### `TestFeed.test_initialize_ist(self)`
**File:** fat_llama_fftw/tests/test_feed.py:64
**Kind:** method
**Description:** Verifies thresholding zeroes out samples at/below the threshold.
**Returns:** `None` (assertion-based).

### `TestFeed.test_upscale_channels(self)`
**File:** fat_llama_fftw/tests/test_feed.py:71
**Kind:** method
**Description:** Verifies `upscale_channels`' output shape, plus (cycle-1 strengthening) finiteness (no NaN/Inf) and that each output column stays within its own source channel's dynamic range — previously shape-only.
**Returns:** `None` (assertion-based).

### `TestFeed.test_upscale_channels_thresholded_out_is_pure_interpolation(self)`
**File:** fat_llama_fftw/tests/test_feed.py:86
**Kind:** method
**Description:** New in cycle 1. With an IST threshold above every sample's magnitude, IST contributes exactly zero, so the result must equal the zero-order-hold interpolation alone — isolates `upscale_channels`' interpolation path from its IST path.
**Returns:** `None` (assertion-based).

### `TestFeed.test_iterative_soft_thresholding_chains_across_iterations(self)`
**File:** fat_llama_fftw/tests/test_feed.py:95
**Kind:** method
**Description:** New in cycle 1 — the regression test for the fixed chaining bug. With `convergence_tol=0.0` (isolating the chaining claim from the convergence claim), asserts `iterative_soft_thresholding`'s 3-pass output matches manually calling `perform_ist_iteration` 3 times in sequence.
**Returns:** `None` (assertion-based).

### `TestFeed.test_iterative_soft_thresholding_stops_early_once_converged(self)`
**File:** fat_llama_fftw/tests/test_feed.py:114
**Kind:** method
**Description:** New in cycle 1 — spies on `perform_ist_iteration` call count to assert `iterative_soft_thresholding` stops well before `max_iter=300` once converged, and that the early-stopped result matches the full (non-early-stopping, `convergence_tol=0.0`) run.
**Returns:** `None` (assertion-based).

### `TestFeed.test_apply_nyquist_cutoff_removes_image_content(self)`
**File:** fat_llama_fftw/tests/test_feed.py:144
**Kind:** method
**Description:** New in cycle 1. Constructs a zero-order-hold-imaged tone (via `new_interpolation_algorithm`) and verifies `apply_nyquist_cutoff` reduces energy above the original Nyquist to <0.01% of its pre-filter value while in-band energy survives.
**Returns:** `None` (assertion-based).

### `TestFeed.test_initialize_ist_scales_with_data_magnitude(self)`
**File:** fat_llama_fftw/tests/test_feed.py:99
**Kind:** method
**Description:** New in cycle 2 — regression test for the absolute-vs-relative threshold bug. Verifies the same fractional `threshold` keeps the same *proportion* of samples regardless of the data's absolute numeric scale (small-scale array vs. the same array x8192, i.e. int16-range), and that int16-scale data isn't simply "keep everything" (the actual pre-fix bug).
**Returns:** `None` (assertion-based).

### `TestFeed.test_initialize_ist_zero_signal_stays_zero(self)`
**File:** fat_llama_fftw/tests/test_feed.py:119
**Kind:** method
**Description:** New in cycle 2 — guards the `peak == 0` short-circuit added for the relative-threshold fix.
**Returns:** `None` (assertion-based).

### `TestFeed.test_ist_adds_content_distinct_from_input_at_realistic_scale(self)`
**File:** fat_llama_fftw/tests/test_feed.py:202
**Kind:** method
**Description:** New in cycle 2 — the direct regression test for the "no measurable added detail" finding. On an int16-scale synthetic signal, asserts `iterative_soft_thresholding`'s output is finite, non-zero, and differs from the input by more than 5% of the input's peak (the pre-fix bug's signature was landing within ~1.95e-3 of the input's own peak — an effective no-op).
**Returns:** `None` (assertion-based).

### `TestFeed.test_upscale_wires_apply_nyquist_cutoff(self, mock_read_audio, mock_write_audio)`
**File:** fat_llama_fftw/tests/test_feed.py:264
**Kind:** method
**Description:** New in cycle 2 — end-to-end regression test (mocked I/O) proving `upscale()` itself calls `apply_nyquist_cutoff` once per channel with the correct upscaled sample rate and original Nyquist, and that the filtered result is what actually gets written — closes the cycle-1 gap where the cutoff was only unit-tested in isolation.
**Returns:** `None` (assertion-based).

### `TestFeed.test_perform_ist_iteration_never_keeps_dc_bin(self)`
**File:** fat_llama_fftw/tests/test_feed.py
**Kind:** method
**Description:** New in cycle 3 — direct regression test for the DC-bin exclusion fix, using a reproduced asymmetric-transient scenario where the DC bin was the single loudest raw FFT bin pre-fix.
**Returns:** `None` (assertion-based).

### `TestFeed.test_iterative_soft_thresholding_output_has_no_dc_offset(self)`
**File:** fat_llama_fftw/tests/test_feed.py
**Kind:** method
**Description:** New in cycle 3 — end-to-end regression test (above `block_size`, exercising the blocked path) asserting `iterative_soft_thresholding`'s output carries no DC offset.
**Returns:** `None` (assertion-based).

### `TestFeed.test_iterative_soft_thresholding_blocks_restore_multiple_bands(self)`
**File:** fat_llama_fftw/tests/test_feed.py
**Kind:** method
**Description:** New in cycle 3 — multi-segment, multi-block synthetic signal asserting all 4 target frequency bands show >1% of peak magnitude after IST (previously 3 of 4 landed at ~0 under the pre-fix whole-file-global threshold).
**Returns:** `None` (assertion-based).

### `TestFeed.test_normalize_signal(self)`
**File:** fat_llama_fftw/tests/test_feed.py:313
**Kind:** method
**Description:** Verifies peak-normalization divides by the max absolute value, plus (cycle-1 strengthening) that the result peaks at exactly 1.0 and stays within `[-1, 1]`.
**Returns:** `None` (assertion-based).

## example.py

### Module-level script
**File:** example.py:1
**Kind:** script (no functions/classes)
**Description:** The README-documented usage example — calls `upscale()` once against the repo-root `input_test.mp3` → `output_test.flac`, with `max_iterations=1000` (README's own documented example value; note this differs from `.claude/agents/rules/audio-quality.md`'s pinned baseline of `max_iterations=300` — the rules file's baseline is intentionally lower for a faster, comparable CI run, not a copy of this example; both are now upper bounds on IST passes rather than exact counts, per `iterative_soft_thresholding`'s cycle-1 convergence early-exit).
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
**Description:** Produces a battery of comparison plots/metrics between an MP3 and a FLAC signal: waveform overlay, difference signal, MSE, spectrogram comparison (this is the routine `.claude/agents/rules/audio-quality.md` says to reuse for the spectrogram comparison image), cross-correlation, and frequency-domain (FFT) comparison. **Still imports `cupy as cp` at module level (analysis.py:6) and uses it for the cross-correlation and FFT steps (`cp.correlate`, `cp.fft.fft`) — unchanged in cycle 1** (`generate-code` explicitly declined this fix — see Open items). This is a CUDA/CuPy dependency in this CPU/FFTW-only package (per `.claude/rules/project-mission.md`'s CPU-only focus); on a machine without CuPy installed, importing this module fails entirely, which affects anything relying on `read_mp3`/`read_flac`/`compare_signals` (including `audio-quality-checker`'s spectrogram-comparison step, which worked around it this run via a `sys.modules` stub rather than a real fix).
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

- **Fixed in cycle 1:** `iterative_soft_thresholding`'s non-chaining `ThreadPoolExecutor` loop (was equivalent to always running a single IST pass regardless of `max_iter`) — now a real sequential loop with a convergence early-exit. **Fixed in cycle 1:** no FFT-domain cutoff above the original Nyquist frequency — `apply_nyquist_cutoff` now runs as `upscale()`'s final stage.
- **Fixed in cycle 2, regressed, then fixed properly in cycle 3:** no measurable added detail below the original Nyquist frequency. Cycle 2's fix (peak-relative thresholding) traded the original bug for a new one — thresholding one whole-file FFT against its own global peak meant almost nothing outside the single loudest moment ever cleared the cutoff, and an asymmetric transient could make the DC bin that loudest moment, injecting a DC offset + drone (cycle-3-discovered regression). Cycle 3 fixed both: `perform_ist_iteration` now always excludes the DC bin, and `iterative_soft_thresholding` processes long signals in 50%-overlap Hann-windowed blocks so thresholding reflects local dynamics. **Fixed in cycle 2:** `upscale()`'s wiring to `apply_nyquist_cutoff` is now covered by an end-to-end mocked-I/O test (previously only unit-tested in isolation).
- **Fixed in cycle 4:** the block/windowed IST reconstruction from cycle 3 injected a periodic broadband artifact at the block-hop rate (nonlinear per-block operation broke the COLA/analysis-only-windowing assumption). Fixed via WOLA — sqrt-Hann windowing at both analysis and synthesis, weight accumulating `window**2`.
- **Fixed in cycle 5:** cycle 4's WOLA fix itself leaked a small DC/near-DC component per block (synthesis windowing is a convolution, not a perfect delta at 0Hz) — measured on real programme material as a ~+40dB DC regression. Fixed by re-zeroing each windowed frame's own mean before accumulation. **Also fixed in cycle 5:** minor write-stage clipping (~4e-6 of samples) caused by `apply_nyquist_cutoff`'s brick-wall filter overshooting an already-normalized peak — fixed by reordering `upscale()` so normalization runs after the cutoff, not before.
- **Still open (deferred, not a cycle-5 priority):** a ~6dB LF-to-HF tonal tilt observed in cycle 5's testing (measured against the pre-DC-leak-fix state — may partially resolve as a side effect of the DC-leak fix, unverified/needs re-testing) and a `write_audio` real-file-roundtrip test coverage gap (existing tests mock `sf.write`; no test verifies a real written file reads back correctly, though this was checked manually).
- **Candidate for a future cycle (deliberately deferred in cycle 4, not attempted for risk/scope reasons with only one cycle left):** replace `new_interpolation_algorithm`'s zero-order-hold upsampling with bandlimited (FFT-zero-padding/sinc) interpolation — per the CUDA sibling repo `bkraad47/fat_llama`'s own equivalent fix, this prevents spectral imaging above the original Nyquist at the source (their measurement: ~1e-8 relative energy above Nyquist immediately after this step) rather than only cleaning it up post hoc via `apply_nyquist_cutoff`, and may free up genuine headroom for IST to add real detail. Do not port the sibling repo's "harmonic-reconstruction term" (removed by them after 3 rounds of new artifacts) or its `toggle_*`-flag public API (out of scope here).
- **Note for a human/future cycle:** `.claude/agents/rules/scientific-coding.md`'s Priority 1 expects an algorithm-level DSP change like cycle 3/4's block/windowed IST to be reflected in README.md's Algorithm Explanation, but README.md is outside every skill/agent's write scope in this pipeline (`generate-code` is restricted to `fat_llama_fftw/**`; `iterate-fat-llama` itself is restricted to `CHANGELOG.md`/`setup.py`'s version field) — flagged by `generate-code` in cycles 2, 3, and 4, still unresolved. README's Algorithm Explanation should be updated to mention DC-bin exclusion and WOLA block/windowed IST processing for long signals.
- **Still open, out of `generate-code`'s write scope:** `analysis.py` (repo root, outside `fat_llama_fftw/**`) still imports `cupy` (GPU-only) despite this package being the CPU/FFTW-only variant.
- **Still open, out of scope for any skill/agent to edit directly:** `upscale()`'s actual signature has no `toggle_normalize`/`toggle_autoscale`/`toggle_adaptive_filter` flags and `feed.py` has no `lms_filter` function — `.claude/agents/rules/audio-quality.md`'s pinned baseline config and runtime-estimate section reference these anyway (apparently carried over from the CUDA sibling package's more elaborate `feed.py`). Needs a human or a future permitted `.claude/` edit to reconcile.
- **Still open, methodology/human decision, not a source bug:** the repo-root reference asset `input_test.flac` is itself a stale output of this same pipeline from *before* the cycle-1 Nyquist-cutoff fix, so it still carries above-Nyquist imaging — `audio-quality-checker`'s spectral-deviation score partly measures agreement with this defective reference rather than fidelity to an independent lossless master. Regenerating it needs a human/methodology decision (e.g. sourcing a true lossless master), not a `generate-code` edit.
- **Still open, environment issue, not a source bug:** a stale `pip install`-ed copy of `fat_llama_fftw` exists in this environment's `venv/Lib/site-packages/`, which can shadow the working tree's own package when a script is run with the working tree not first on `sys.path` — `audio-quality-checker` hit this on its first cycle-2 attempt (silently ran pre-fix code). Needs `pip install -e .` / reinstall in this environment; not a file for any skill/agent to edit.
- `upscale()` end-to-end coverage is now present (`test_upscale_wires_apply_nyquist_cutoff`, cycle 2) but only for the Nyquist-cutoff wiring specifically — no test yet exercises the full `upscale()` call against a real (or fully synthetic) audio file end-to-end for output correctness generally.
