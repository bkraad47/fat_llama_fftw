# fat_llama_fftw — Current State

Regenerable factblock snapshot of the codebase, produced by the `review-current-state` skill. Do not hand-edit — regenerate on demand instead. Scope: whole repository (tracked files only; build/venv/cache directories excluded).

Snapshot taken against candidate `v-2.0.0/iter-2` — after `iterate-fat-llama`'s cycle 2 fix, which corrected a real-audio regression cycle 1's port (frequency-selective + envelope-gated IST peak-capping, ported from the CUDA sibling repo `bkraad47/fat_llama`'s `v-2.0.0` branch) introduced into `_cap_ist_changes_to_baseline_peak`.

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
│       ├── logo.jpg
│       ├── spectrogram_comparison.png
│       └── theory.png
├── fat_llama_fftw/
│   ├── __init__.py
│   ├── audio_fattener/
│   │   ├── __init__.py
│   │   └── feed.py
│   └── tests/
│       ├── __init__.py
│       └── test_feed.py
├── .gitignore
├── CHANGELOG.md
├── LICENSE
├── Manifest.in
├── README.md
├── analysis.py
├── example.py
├── input_test.flac
├── input_test.mp3
├── output_test.flac
├── requirements.txt
├── setup.py
└── test_output.txt
```

## fat_llama_fftw/audio_fattener/feed.py

### `_fft_thread_count(n) -> int`
**File:** fat_llama_fftw/audio_fattener/feed.py:55
**Kind:** function
**Description:** Returns how many FFTW threads to request for a transform of length `n` — `1` below `_MULTI_THREAD_FFT_MIN_SAMPLES` (200,000; multi-threading is a measured net loss on small transforms — 1.6x-9x slower — because thread dispatch overhead dominates), otherwise `os.cpu_count()`. Used by `apply_nyquist_cutoff`'s whole-signal transforms; `perform_ist_iteration`'s fixed-size per-block transforms stay single-threaded always.
**Parameters:**
- `n` (`int`): transform length.
**Returns:** `int` — thread count to pass as `pyfftw`'s `threads=` kwarg.
**Usage:**
```python
threads = _fft_thread_count(len(signal))
```

### `read_audio(file_path, format) -> tuple`
**File:** fat_llama_fftw/audio_fattener/feed.py:67
**Kind:** function
**Description:** Reads an audio file via `soundfile.read` (returns samples normalized to `[-1, 1]`, shaped `(frames,)` mono or `(frames, channels)` for any channel count) and looks up its bitrate via the matching `mutagen` reader (`MP3`/`FLAC`/`OggVorbis`/`WAVE`). For any other/uncatalogued format, estimates bitrate from `len(samples) * 8 / duration_seconds` instead of a container tag.
**Parameters:**
- `file_path` (`str` or file-like object): input audio file. Raises `FileNotFoundError` if a path string doesn't exist.
- `format` (`str`): `'mp3'`, `'flac'`, `'ogg'`, `'wav'`, or anything else `soundfile`/`libsndfile` can decode (falls to the duration-estimate branch).
**Returns:** `(sample_rate: int, samples: np.ndarray, bitrate: float|None)`.
**Usage:**
```python
sample_rate, samples, bitrate = read_audio('input_test.mp3', format='mp3')
```

### `write_audio(file_path, sample_rate, data, format) -> None`
**File:** fat_llama_fftw/audio_fattener/feed.py:115
**Kind:** function
**Description:** Writes `data` (cast to `float32`) to a FLAC or WAV file via `soundfile`, at 24-bit (`PCM_24`) depth. Raises `ValueError` for any other target format.
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
**File:** fat_llama_fftw/audio_fattener/feed.py:126
**Kind:** function
**Description:** Bandlimited (FFT zero-padding / ideal-sinc) interpolation: `rfft` the channel, zero-pad its one-sided spectrum out to the upscaled length's own `rfft` size (halving the original Nyquist bin first for even-length inputs, the standard real-FFT correction), `irfft` back, and rescale by `upscale_factor` to compensate `irfft`'s larger normalization divisor. Replaces an earlier zero-order-hold (repeat-each-sample) implementation, which imaged the original spectrum above the original Nyquist frequency (measured -34.7dB relative on real material, vs. this method's -153.3dB — pure float noise) and left every downstream stage (IST's peak-relative threshold, the peak cap) operating on a signal shaped by that imaging. `upscale_factor==1` and empty input are explicit identity/passthrough short-circuits.
**Parameters:**
- `data` (`np.ndarray`): 1-D channel samples.
- `upscale_factor` (`int`): how many times to expand the sample count.
**Returns:** `np.ndarray` (`float32`), length `len(data) * upscale_factor`.
**Usage:**
```python
expanded = new_interpolation_algorithm(channel, upscale_factor=4)
```

### `initialize_ist(data, threshold) -> np.ndarray`
**File:** fat_llama_fftw/audio_fattener/feed.py:207
**Kind:** function
**Description:** First IST step: zeroes every sample at/below `threshold * max(abs(data))` (a fraction of the array's own peak, not an absolute cutoff — makes `threshold` scale-invariant regardless of `data`'s absolute numeric range). Returns an all-zero array for a silent (`peak == 0`) input.
**Parameters:**
- `data` (`np.ndarray`): input samples (typically the interpolated/expanded channel).
- `threshold` (`float`): fraction (0-1) of `data`'s own peak magnitude to use as the cutoff.
**Returns:** `np.ndarray` — same shape as `data`, thresholded.
**Usage:**
```python
data_thres = initialize_ist(expanded_channel, threshold=0.6)
```

### `perform_ist_iteration(data_thres, threshold) -> np.ndarray`
**File:** fat_llama_fftw/audio_fattener/feed.py:220
**Kind:** function
**Description:** One IST refinement pass: FFT the current estimate (`pyfftw.interfaces.numpy_fft`, cached via `pyfftw.interfaces.cache.enable()` at module load), zero out FFT bins at/below `threshold * max(abs(fft))`, **always excludes the DC bin (`mask[0] = False`)** regardless of whether it clears the threshold (an asymmetric transient can otherwise make DC the single loudest bin, injecting a spurious constant offset into every subsequent pass), then inverse-FFTs back and keeps the real part.
**Parameters:**
- `data_thres` (`np.ndarray`): current time-domain estimate (one block's worth, when called via the blocked path).
- `threshold` (`float`): fraction of the current FFT magnitude's own peak to use as the cutoff.
**Returns:** `np.ndarray` — refined time-domain estimate (real-valued).
**Usage:**
```python
data_thres = perform_ist_iteration(data_thres, threshold=0.6)
```

### `_ist_chain(data, max_iter, threshold, convergence_tol) -> np.ndarray`
**File:** fat_llama_fftw/audio_fattener/feed.py:254
**Kind:** function (private helper)
**Description:** Runs `perform_ist_iteration` in a genuine sequential chain (each pass feeds its own output into the next) with an early exit once a pass's change falls below `convergence_tol` relative to the initial post-threshold scale — hard-threshold IST is a fixed-point projection, so passes past that point are provably no-ops. Called once for a short/whole-signal input, or once per block for the blocked path in `iterative_soft_thresholding`.
**Parameters:**
- `data` (`np.ndarray`): input samples (a whole signal or a single block).
- `max_iter` (`int`): upper bound on passes.
- `threshold` (`float`): fraction-of-peak magnitude cutoff.
- `convergence_tol` (`float`): relative early-exit tolerance.
**Returns:** `np.ndarray` — the converged (or `max_iter`-capped) IST estimate.

### `iterative_soft_thresholding(data, max_iter, threshold, convergence_tol=1e-6, block_size=8192) -> np.ndarray`
**File:** fat_llama_fftw/audio_fattener/feed.py:287
**Kind:** function
**Description:** For `len(data) <= block_size`, a single `_ist_chain` call over the whole array. Above `block_size`, splits the signal into 50%-overlapping **sqrt-Hann**-windowed blocks and runs `_ist_chain` independently on each (so the peak-relative threshold reflects each block's own local dynamics rather than one whole-file global peak), reconstructing via **WOLA** (weighted overlap-add — the same sqrt-Hann window applied again on the synthesis side before accumulation, weight summing `window**2`; needed because `_ist_chain`'s nonlinear FFT-threshold projection does not preserve an analysis-windowed frame's own edge taper on the way out, so synthesis-side windowing is required to keep the discontinuity out of the overlap-add sum). Each windowed block's own synthesis-side DC/near-DC leak (windowing is a convolution with the window's own spectrum, not a perfect delta at 0Hz) is removed before accumulation via a taper-preserving correction (`windowed_result - window * (sum(windowed_result) / sum(window))`, not a flat scalar subtraction, which would re-introduce edge discontinuities).
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

### `_local_peak_envelope(signal, block_size=8192) -> np.ndarray`
**File:** fat_llama_fftw/audio_fattener/feed.py:400
**Kind:** function (private helper)
**Description:** **New this cycle** (ported from the CUDA sibling repo's `v-2.0.0` branch, translated cupy→numpy/pyfftw). Estimates a smooth, WOLA-based local-peak-magnitude envelope of `signal`: each `block_size`-sample analysis window's own peak (`max(abs(.))` of that windowed block) is broadcast across the block and overlap-added using the same 50%-hop sqrt-Hann window-shape and `window**2` weighting `iterative_soft_thresholding` uses for its own reconstruction — implemented as a separate, self-contained function (not a refactor of that one) so it carries zero risk to `iterative_soft_thresholding`'s already-tested WOLA/DC-leak-correction path. For `len(signal) <= block_size`, returns a constant array at the signal's own overall peak. Used by `_cap_ist_changes_to_baseline_peak` to gate its frequency-selective correction so quiet/onset regions (never responsible for a peak overshoot) receive little to none of it.
**Parameters:**
- `signal` (`np.ndarray`): the signal to estimate an envelope for (typically `expanded_channel`, the pre-IST baseline).
- `block_size` (`int`): analysis/synthesis window length in samples. Default `8192`.
**Returns:** `np.ndarray` (`float32`), same length as `signal`, non-negative.
**Usage:**
```python
envelope = _local_peak_envelope(expanded_channel)
gate = np.clip(envelope / np.max(np.abs(expanded_channel)), 0.0, 1.0)
```

### `_cap_ist_changes_to_baseline_peak(expanded_channel, ist_changes, max_rounds=20, dominant_band_ratio=0.002, safety_margin=1.2, onset_gate_ratio=0.3) -> np.ndarray`
**File:** fat_llama_fftw/audio_fattener/feed.py:453
**Kind:** function (private helper)
**Description:** Bounds how much `perform_ist_iteration`'s peak-relative boost of a channel's dominant (usually low-frequency) content is allowed to inflate the combined (interpolation + IST) signal's peak above the pre-IST baseline's own peak — an inflated peak here is not "corrected" by `upscale()`'s later autoscale/normalize stages (each a per-channel scalar, together mathematically inert on the final output), it determines how hard *every* frequency, not just what IST touched, gets divided down at final normalization. Cycle 1 (this run) rewrote this on top of the original whole-signal uniform-scalar shrink (still present as the fallback path) with two refinements ported from the CUDA sibling repo `bkraad47/fat_llama`'s `v-2.0.0` branch (pure numpy/FFT logic, no CUDA-specific mechanism): (1) **frequency-selective dominant/residual split** — `ist_changes`'s own `rfft` spectrum is split into a "dominant" component (bins `>= dominant_band_ratio` of the spectrum's own peak magnitude) and a "residual" component; only the dominant component is iteratively shrunk (the original bounded multiplicative-shrink loop, reused, scoped to this component) toward bounding the combined peak, leaving residual (e.g. genuinely-added quiet/high-frequency detail) untouched; (2) **envelope-gated correction** — the dominant-band correction is tapered by a gate derived from `_local_peak_envelope(expanded_channel)`, so a genuinely non-stationary channel's quiet onset/fade-in receives little to none of the correction. **Cycle 2 (this run) fix, `onset_gate_ratio` (new):** cycle 1's gate (`clip(envelope / baseline_peak, 0, 1)`) was measured on real audio to sit below 1.0 for ~99.6% of samples in an ordinary loud passage (real music's crest factor keeps a block's local peak well below the channel's single loudest instant almost everywhere, not just at genuine quiet/onset regions), suppressing the intended correction nearly everywhere and letting `ist_changes` retain ~24-29% of its own uncapped peak instead of the ~7-13% the original uniform-shrink cap was tuned for — measured as a +1.0 to +1.55dB low-frequency boost that the final normalize then paid for out of every other band (coherence 9.9→7.0 on real material). Fix: `gate = clip((envelope / baseline_peak) / onset_gate_ratio, 0, 1)` — saturates to `1.0` (full correction) throughout any region whose local envelope is within `onset_gate_ratio` of the channel's own peak (i.e. ordinary loud passages), while still gating down to ~0 only in genuinely quiet/onset regions well below that floor. Falls back to one additional whole-signal uniform-shrink pass (the original method, gated the same way) on top of the frequency-selective candidate only if that candidate's own combined peak still exceeds `baseline_peak * safety_margin`. `max_rounds=20`, `dominant_band_ratio=0.002`, `safety_margin=1.2` unchanged; `onset_gate_ratio=0.3` empirically chosen (cycle 2) from a real audio slice — verified in the range ~[0.05, 0.35].
**Parameters:**
- `expanded_channel` (`np.ndarray`): the pre-IST interpolated baseline for this channel.
- `ist_changes` (`np.ndarray`): IST's contribution, as returned by `iterative_soft_thresholding`.
- `max_rounds` (`int`): bound on shrink-loop iterations (both the dominant-band pass and the safety-net fallback). Default `20`.
- `dominant_band_ratio` (`float`): fraction of the spectrum's own peak magnitude a bin must reach to be classified "dominant". Default `0.002`.
- `safety_margin` (`float`): the safety-net fallback only engages if the frequency-selective candidate's own combined peak still exceeds `baseline_peak * safety_margin`. Default `1.2`.
- `onset_gate_ratio` (`float`): **new this cycle.** Fraction of the channel's own peak below which the envelope gate starts tapering the dominant-band correction down; at/above it the gate saturates to `1.0` (full correction applied). Default `0.3`.
**Returns:** `np.ndarray` — `ist_changes`, reshaped/rescaled per the above (unchanged if no capping was needed).
**Usage:**
```python
capped = _cap_ist_changes_to_baseline_peak(expanded_channel, ist_changes)
```

### `_process_channel(channel, upscale_factor, max_iter, threshold) -> np.ndarray`
**File:** fat_llama_fftw/audio_fattener/feed.py:659
**Kind:** function (private helper)
**Description:** The full per-channel pipeline — interpolate (`new_interpolation_algorithm`) → IST (`iterative_soft_thresholding`) → peak-cap (`_cap_ist_changes_to_baseline_peak`) → combine — factored out so `upscale_channels` can dispatch it either sequentially or on a worker thread. Reads/writes only its own `channel` argument (no shared state), so channels are provably safe to run concurrently.
**Parameters:**
- `channel` (`np.ndarray`): 1-D single-channel samples.
- `upscale_factor` (`int`), `max_iter` (`int`), `threshold` (`float`): passed through to the pipeline stages.
**Returns:** `np.ndarray` (`float32`) — this channel's fully processed (interpolated + IST + capped) samples.

### `upscale_channels(channels, upscale_factor, max_iter, threshold) -> np.ndarray`
**File:** fat_llama_fftw/audio_fattener/feed.py:675
**Kind:** function
**Description:** Runs `_process_channel` per channel and stacks the results back into a single 2-D array. For multi-channel (e.g. stereo) input, dispatches each channel's independent work across a `ThreadPoolExecutor` (numpy/pyfftw release the GIL during vectorized FFT work) instead of a sequential loop — measured 1.24x wall-clock speedup on real stereo `input_test.mp3`, verified bit-identical output. Mono input skips the thread pool (no benefit for one unit of work).
**Parameters:**
- `channels` (`np.ndarray`): shape `(n_samples, n_channels)`.
- `upscale_factor` (`int`): passed to `new_interpolation_algorithm`.
- `max_iter` (`int`): passed to `iterative_soft_thresholding`.
- `threshold` (`float`): passed to `initialize_ist`/IST iterations.
**Returns:** `np.ndarray`, shape `(n_samples * upscale_factor, n_channels)`.
**Usage:**
```python
upscaled = upscale_channels(channels, upscale_factor=4, max_iter=300, threshold=0.6)
```

### `normalize_signal(signal) -> np.ndarray`
**File:** fat_llama_fftw/audio_fattener/feed.py:720
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
**File:** fat_llama_fftw/audio_fattener/feed.py:724
**Kind:** function
**Description:** Zeroes every FFT bin above `original_nyquist` (via `pyfftw` `rfft`/`irfft`, multi-threaded per `_fft_thread_count` for large signals), enforcing `.claude/rules/project-mission.md`'s hard "no content above the original Nyquist frequency" constraint. Runs after amplitude auto-scaling but *before* `upscale()`'s final `normalize_signal` pass — a brick-wall FFT filter can overshoot its own input's peak (Gibbs ripple), so normalizing after the cutoff guarantees the written signal never exceeds full scale.
**Parameters:**
- `signal` (`np.ndarray`): 1-D channel samples at the *upscaled* sample rate.
- `sample_rate` (`int`): the upscaled signal's own sample rate.
- `original_nyquist` (`float`): the cutoff — original source sample rate / 2.
**Returns:** `np.ndarray` (`float32`), same length as `signal`, with all content above `original_nyquist` removed.
**Usage:**
```python
filtered = apply_nyquist_cutoff(upscaled_channel, new_sample_rate, original_sample_rate / 2.0)
```

### `upscale(input_file_path, output_file_path, source_format, target_format='flac', max_iterations=800, threshold_value=0.6, target_bitrate_kbps=1411) -> None`
**File:** fat_llama_fftw/audio_fattener/feed.py:750
**Kind:** function
**Description:** The package's public entry point (README's documented API). Reads the source file, computes an `upscale_factor` from `target_bitrate_kbps` vs. the source's own bitrate (`round(target_bitrate / bitrate)`, floored at `1` so a source whose own bitrate already exceeds the target — e.g. a lossless WAV/FLAC source — never rounds to 0/crashes, just skips the sample-rate increase), runs `upscale_channels` per channel, auto-scales each channel back to its original peak, runs `apply_nyquist_cutoff`, and finally normalizes (in that order — normalization strictly last so the cutoff's own Gibbs overshoot can't push samples above full scale on write). No `toggle_*` flags exist on this signature — auto-scaling, the Nyquist cutoff, and normalization always run unconditionally.
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
    max_iterations=600,
    threshold_value=0.75,
    target_bitrate_kbps=1400
)
```

## fat_llama_fftw/tests/test_feed.py

### `TestFeed`
**File:** fat_llama_fftw/tests/test_feed.py:35
**Kind:** class
**Description:** `unittest.TestCase` covering every function in `feed.py`, 50 test methods as of this cycle. Notable groups: I/O (`read_audio`/`write_audio`, including a real on-disk roundtrip test), interpolation (bandlimited-ness, identity/passthrough edge cases, no-imaging-above-Nyquist), IST core (thresholding, chaining, convergence early-exit, DC-bin exclusion, WOLA block/no-discontinuity behavior, scale-invariance), `_local_peak_envelope` (3 direct unit tests, new cycle 2), the peak cap (`_cap_ist_changes_to_baseline_peak` — 6 tests, see below), 2 strengthened net-attenuation tests (now also bound a low-frequency *boost*, cycle 2), the Nyquist cutoff (removal + multi-threading), and full `upscale()` wiring/edge cases (bitrate clamping, never-exceeds-full-scale, a real end-to-end on-disk FLAC write).
**Usage:**
```python
python -m unittest discover -s fat_llama_fftw/tests
```

### `TestFeed.test_cap_ist_changes_to_baseline_peak_noop_when_not_needed(self)`
**File:** fat_llama_fftw/tests/test_feed.py:508
**Kind:** method
**Description:** Asserts the cap leaves `ist_changes` unmodified when the combined peak never exceeds the baseline (zero baseline peak, or an out-of-phase addition that can only reduce the combined amplitude).
**Returns:** `None` (assertion-based).

### `TestFeed.test_cap_ist_changes_to_baseline_peak_bounds_combined_peak(self)`
**File:** fat_llama_fftw/tests/test_feed.py:519
**Kind:** method
**Description:** Regression test for the original attenuation fix: on an in-phase-dominant synthetic signal, asserts the cap meaningfully reduces the combined-peak overshoot without collapsing `ist_changes` to (near) zero.
**Returns:** `None` (assertion-based).

### `TestFeed.test_cap_ist_changes_to_baseline_peak_preserves_quiet_band(self)`
**File:** fat_llama_fftw/tests/test_feed.py:567
**Kind:** method
**Description:** **New this cycle.** Loud-low/quiet-high two-tone fixture: asserts the quiet high-frequency band's own FFT magnitude survives the (new, frequency-selective) cap at >90% of its uncapped value, while the combined peak is still bounded close to baseline — the direct regression test for the frequency-selective dominant/residual split (the old uniform shrink left this band at ~4% survival).
**Returns:** `None` (assertion-based).

### `TestFeed.test_cap_ist_changes_to_baseline_peak_pathological_ratio_falls_back(self)`
**File:** fat_llama_fftw/tests/test_feed.py:622
**Kind:** method
**Description:** **New this cycle.** A `dominant_band_ratio` high enough that nothing classifies as dominant; asserts the safety-net whole-signal uniform-shrink fallback still bounds the combined peak at least as well as the pre-existing (pre-cycle) cap did.
**Returns:** `None` (assertion-based).

### `TestFeed.test_cap_ist_changes_to_baseline_peak_zero_baseline(self)`
**File:** fat_llama_fftw/tests/test_feed.py:645
**Kind:** method
**Description:** Asserts the cap returns `ist_changes` unchanged when `expanded_channel` is silent (nothing to cap against).
**Returns:** `None` (assertion-based).

### `TestFeed.test_cap_ist_changes_to_baseline_peak_envelope_gates_quiet_onset(self)`
**File:** fat_llama_fftw/tests/test_feed.py:654
**Kind:** method
**Description:** Broadband, raised-cosine fade-in synthetic channel run through the real `new_interpolation_algorithm` → `iterative_soft_thresholding` → `_cap_ist_changes_to_baseline_peak` pipeline; asserts the onset window's post-cap RMS elevation vs. the pre-IST interpolation baseline stays under a small bound — the direct regression test for `_local_peak_envelope`'s gating. Re-verified this cycle (2) against the `onset_gate_ratio` fix: still passes (~0.03dB elevation, well under bound), confirming the fix didn't regress onset protection while fixing the real-material over-suppression bug.
**Returns:** `None` (assertion-based).

### `TestFeed.test_upscale_channels_caps_combined_peak_close_to_baseline_stereo(self)`
**File:** fat_llama_fftw/tests/test_feed.py:718
**Kind:** method
**Description:** End-to-end (stereo, 2 distinct-content channels) wiring check that `upscale_channels` itself — not just the cap helper in isolation — keeps each channel's combined peak close to its own pre-IST interpolation baseline.
**Returns:** `None` (assertion-based).

### `TestFeed.test_local_peak_envelope_empty_signal(self)` / `test_local_peak_envelope_short_signal_returns_constant_peak(self)` / `test_local_peak_envelope_tracks_loud_and_quiet_regions(self)`
**File:** fat_llama_fftw/tests/test_feed.py:508, 520, 530
**Kind:** method (3 tests)
**Description:** **New this cycle (2).** Direct unit coverage for `_local_peak_envelope`, closing a gap `audio-quality-checker` found (the function was imported in cycle 1 but never referenced by any test): empty-signal edge case, short-signal (`<= block_size`) constant-peak passthrough, and that the envelope actually tracks distinct loud vs. quiet regions of a synthetic signal.
**Returns:** `None` (assertion-based, each).

### `TestFeed.test_upscale_channels_ist_does_not_net_attenuate_untouched_band(self)` / `test_upscale_channels_ist_does_not_net_attenuate_real_material(self)`
**File:** fat_llama_fftw/tests/test_feed.py:809, 906
**Kind:** method (2 tests, strengthened this cycle)
**Description:** Pre-existing net-attenuation regression tests, extended this cycle (2) with a symmetric low-frequency/low-band **boost** bound (`assertLess(lf_ratio_db, 1.0)` / a new 20-300Hz `assertLess(lb_ratio_db, 0.5)` check) calibrated against the actual cycle-1-regression measurement, so a future recurrence of the "onset_gate_ratio"-class bug (excess low-frequency content surviving the peak cap) fails the unit suite directly instead of requiring a full `audio-quality-checker` real-material run to catch it.
**Returns:** `None` (assertion-based, each).

### Other test methods (unchanged since cycle 1)
**File:** fat_llama_fftw/tests/test_feed.py
**Kind:** method (39 additional tests)
**Description:** `test_read_audio`, `test_read_audio_unsupported_format_computes_bitrate_from_duration`, `test_read_audio_does_not_corrupt_channel_counts_above_stereo`, `test_write_audio`, `test_write_audio_wav_uses_wav_container`, `test_write_audio_rejects_unsupported_format`, `test_write_audio_roundtrip_preserves_audio_properties_on_disk`, `test_new_interpolation_algorithm(_identity_at_upscale_factor_one|_odd_length_passthrough|_no_imaging_above_original_nyquist)`, `test_initialize_ist(_scales_with_data_magnitude|_zero_signal_stays_zero)`, `test_upscale_channels(_thresholded_out_is_pure_interpolation|_parallel_matches_sequential_per_channel|_mono_skips_thread_pool)`, `test_iterative_soft_thresholding_chains_across_iterations`, `test_iterative_soft_thresholding_stops_early_once_converged`, `test_ist_adds_content_distinct_from_input_at_realistic_scale`, `test_ist_pipeline_is_scale_invariant_within_float32_precision`, `test_perform_ist_iteration_never_keeps_dc_bin`, `test_iterative_soft_thresholding_output_has_no_dc_offset`, `test_ist_no_dc_offset_on_real_programme_material`, `test_iterative_soft_thresholding_blocks_restore_multiple_bands`, `test_iterative_soft_thresholding_no_block_boundary_discontinuity`, `test_feed_block_boundary_test_fixture_is_representative`, `test_no_block_hop_boundary_curvature_bump`, `test_apply_nyquist_cutoff_removes_image_content`, `test_fft_thread_count_scoped_to_large_transforms`, `test_apply_nyquist_cutoff_requests_multiple_threads_for_large_signal`, `test_upscale_wires_apply_nyquist_cutoff`, `test_upscale_clamps_upscale_factor_to_one_for_high_bitrate_source`, `test_upscale_factor_one_from_moderately_high_bitrate_source_no_warning`, `test_upscale_output_never_exceeds_full_scale`, `test_upscale_writes_valid_flac_end_to_end_on_disk`, `test_normalize_signal`. All still pass against cycle 2's `onset_gate_ratio` fix.
**Returns:** `None` (assertion-based, each).

## example.py

### Module-level script
**File:** example.py:1
**Kind:** script (no functions/classes)
**Description:** The README-documented usage example — calls `upscale()` once against the repo-root `input_test.mp3` → `output_test.flac` with `max_iterations=600, threshold_value=0.75, target_bitrate_kbps=1400`. Not affected by this cycle's `_cap_ist_changes_to_baseline_peak` change since `upscale()`'s own public signature gained no new parameters (the new `dominant_band_ratio`/`safety_margin` kwargs are internal to the capping helper, not exposed here).
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
**Description:** Peak-normalizes a signal to `[-1, 1]` — a separate copy of `feed.py`'s `normalize_signal`, kept standalone in this analysis script.
**Parameters:**
- `signal` (`np.ndarray`): input samples.
**Returns:** `np.ndarray`, peak-normalized.

### `compare_signals(mp3, flac, sample_rate) -> None`
**File:** analysis.py:24
**Kind:** function
**Description:** Produces comparison plots/metrics between an MP3 and a FLAC signal: waveform overlay, difference signal, MSE, spectrogram comparison (the routine `.claude/agents/rules/audio-quality.md` reuses for the spectrogram comparison image), cross-correlation, and FFT comparison — plain `numpy` throughout (no `cupy`), consistent with this CPU/FFTW-only package.
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

- **This run's port (iterate-fat-llama, DIRECTIVES: bring over `bkraad47/fat_llama`'s `v-2.0.0` non-CUDA changes):** compared that sibling repo's `v-2.0.0` branch against its own `main` — only `feed.py`/`test_feed.py` (plus changelog/readme/version) differed. This repo was already at parity or ahead on peak-relative thresholding, DC-bin exclusion, convergence early-exit, and WOLA blocking (this repo's own WOLA has a DC-leak correction the sibling's does not). The one real gap — `_cap_ist_changes_to_baseline_peak`'s frequency-selective + envelope-gated refinement — was ported in cycle 1. Did not port the sibling's separately-investigated-and-rejected `threshold_value=0.15` experiment (their own changelog logged it as "investigated, no change made").
- **Cycle 1→2 regression, now fixed:** cycle 1's straight port introduced a real regression measurable only on real (non-synthetic) audio — `audio-quality-checker` measured coherence 9.9→7.0 and spectral_deviation 9.8→9.2 at the pinned baseline config. Root cause: the envelope gate (cycle 1) was scaled against the channel's single global peak, so it suppressed the frequency-selective correction almost everywhere in an ordinary loud passage, not just at genuine quiet onsets — a crest-factor effect the synthetic fixtures used to write cycle 1's own unit tests didn't exercise. Cycle 2 added `onset_gate_ratio=0.3` (see factblock) to fix this, verified directly on a real audio slice. This is a concrete illustration of why `audio-quality-checker`'s real-material measurement is load-bearing, not redundant with the unit suite.
- **Environment hygiene finding (informational, not a source bug):** `audio-quality-checker` found a stale `fat_llama_fftw` 1.0.4 pip-installed into `venv/Lib/site-packages`, which silently shadows the repo source for any script run from outside the repo root (`python <script>` puts the script's own directory, not the repo root, at `sys.path[0]`). Not fixed this run (out of any skill's write scope) — flagged so a human or a future permitted step can `pip uninstall`/reinstall in editable mode.
- **Still open, flagged by `generate-code` (out of its `fat_llama_fftw/**` write scope):** README.md's Algorithm Explanation (step 4) still describes the peak cap as a flat rescale, with no mention of the new frequency-selective/envelope-gated refinement — needs a human or a permitted direct edit to reconcile, same recurring gap prior cycles have flagged for other algorithm changes.
- **Still open, out of scope for any skill/agent to edit directly:** `upscale()`'s actual signature has no `toggle_normalize`/`toggle_autoscale`/`toggle_adaptive_filter` flags and `feed.py` has no `lms_filter` function — `.claude/agents/rules/audio-quality.md`'s pinned baseline config and runtime-estimate section reference these anyway (apparently carried over from the CUDA sibling package's more elaborate `feed.py`). Needs a human or a future permitted `.claude/` edit to reconcile.
- **Still open, methodology/human decision, not a source bug:** the repo-root reference asset `input_test.flac` is itself a stale output of this same pipeline from before the original Nyquist-cutoff fix, so it still carries above-Nyquist imaging — `audio-quality-checker`'s spectral-deviation score partly measures agreement with this defective reference rather than fidelity to an independent lossless master.
- `upscale()` end-to-end coverage exists (`test_upscale_wires_apply_nyquist_cutoff`, `test_upscale_writes_valid_flac_end_to_end_on_disk`) including a real on-disk write/readback test.
