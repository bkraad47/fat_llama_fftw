# fat_llama_fftw — Current State

Regenerable factblock snapshot of the codebase, produced by the `review-current-state` skill. Do not hand-edit — regenerate on demand instead. Scope: whole repository (tracked files only; build/venv/cache directories excluded).

Snapshot taken against candidate `v-2.0.0/iter-6` — after an `iterate-fat-llama` cycle 3 investigation (no behavior change): a 4th independent attempt at the "no IST-attributable added detail below the original Nyquist" gap tested two genuinely new hypotheses (the peak-cap's dominant/residual split bottlenecking HF content; convergence/`max_iterations` differing by band) and disproved both with direct measurement against the lossless `input_test.flac` reference. No source logic changed — only docstring/comment additions recording the investigation, plus 2 new permanent regression tests proving the root cause (real-material blocks never classify any bin `>=2kHz` as "dominant," and the kept-bin mask is identical pass-over-pass, so neither the cap nor iteration count is what's discarding HF content — nothing is ever produced there in the first place).

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
**Description:** Bandlimited (FFT zero-padding / ideal-sinc) interpolation: `rfft` the channel, zero-pad its one-sided spectrum out to the upscaled length's own `rfft` size (halving the original Nyquist bin first for even-length inputs, the standard real-FFT correction), `irfft` back, and rescale by `upscale_factor` to compensate `irfft`'s larger normalization divisor. Replaces an earlier zero-order-hold (repeat-each-sample) implementation, which imaged the original spectrum above the original Nyquist frequency. `upscale_factor==1` and empty input are explicit identity/passthrough short-circuits.
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
**Description:** One IST refinement pass: FFT the current estimate (`pyfftw.interfaces.numpy_fft`, cached via `pyfftw.interfaces.cache.enable()` at module load), zero out FFT bins at/below `threshold * max(abs(fft))`, **always excludes the DC bin (`mask[0] = False`)** regardless of whether it clears the threshold, then inverse-FFTs back and keeps the real part. A per-band (rather than whole-spectrum) peak-relative thresholding variant was investigated in a prior cycle as a direct attempt at the "no measurable added detail above ~800Hz" gap — it did raise high-frequency band energy on real material, but a direct envelope-correlation check against the lossless reference showed the gain was NOT more reference-correlated (stayed at ~0.02-0.08 regardless of band count) — i.e. noise, not detail — and it also reopened an existing block-boundary-discontinuity regression. Reverted; function body unchanged since. **This cycle's investigation (comment-only addition):** directly measured, on a real programme-material block, that the kept-bin mask is bit-identical between pass 0 and pass 1, and that 0 of thousands of possible bins `>=2kHz` are ever kept — confirming the single hard peak-relative threshold reaches its fixed point in one pass and structurally never selects HF content on real material, regardless of `max_iterations`. Function body still unchanged.
**Parameters:**
- `data_thres` (`np.ndarray`): current time-domain estimate (one block's worth, when called via the blocked path).
- `threshold` (`float`): fraction of the current FFT magnitude's own peak to use as the cutoff.
**Returns:** `np.ndarray` — refined time-domain estimate (real-valued).
**Usage:**
```python
data_thres = perform_ist_iteration(data_thres, threshold=0.6)
```

### `_ist_chain(data, max_iter, threshold, convergence_tol) -> np.ndarray`
**File:** fat_llama_fftw/audio_fattener/feed.py:302
**Kind:** function (private helper)
**Description:** Runs `perform_ist_iteration` in a genuine sequential chain (each pass feeds its own output into the next) with an early exit once a pass's change falls below `convergence_tol` relative to the initial post-threshold scale — hard-threshold IST is a fixed-point projection, so passes past that point are provably no-ops. Called once for a short/whole-signal input, or once per block for the blocked path in `iterative_soft_thresholding`.
**Parameters:**
- `data` (`np.ndarray`): input samples (a whole signal or a single block).
- `max_iter` (`int`): upper bound on passes.
- `threshold` (`float`): fraction-of-peak magnitude cutoff.
- `convergence_tol` (`float`): relative early-exit tolerance.
**Returns:** `np.ndarray` — the converged (or `max_iter`-capped) IST estimate.

### `iterative_soft_thresholding(data, max_iter, threshold, convergence_tol=1e-6, block_size=8192) -> np.ndarray`
**File:** fat_llama_fftw/audio_fattener/feed.py:335
**Kind:** function
**Description:** For `len(data) <= block_size`, a single `_ist_chain` call over the whole array. Above `block_size`, splits the signal into 50%-overlapping **sqrt-Hann**-windowed blocks and runs `_ist_chain` independently on each (so the peak-relative threshold reflects each block's own local dynamics rather than one whole-file global peak), reconstructing via **WOLA** (weighted overlap-add). Each windowed block's own synthesis-side DC/near-DC leak is removed before accumulation via a taper-preserving correction (`windowed_result - window * (sum(windowed_result) / sum(window))`).
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
**File:** fat_llama_fftw/audio_fattener/feed.py:448
**Kind:** function (private helper)
**Description:** Smooth, WOLA-based local-peak-magnitude envelope of `signal`: each `block_size`-sample analysis window's own peak (`max(abs(.))`) is broadcast across the block and overlap-added using the same 50%-hop sqrt-Hann window-shape and `window**2` weighting `iterative_soft_thresholding` uses, implemented as a separate function so it carries zero risk to that function's own tested path. For `len(signal) <= block_size`, returns a constant array at the signal's own overall peak. Kept for any future peak-envelope need and still has its own direct unit tests — `_cap_ist_changes_to_baseline_peak`'s gates use `_local_rms_envelope` instead.
**Parameters:**
- `signal` (`np.ndarray`): the signal to estimate an envelope for.
- `block_size` (`int`): analysis/synthesis window length in samples. Default `8192`.
**Returns:** `np.ndarray` (`float32`), same length as `signal`, non-negative.
**Usage:**
```python
envelope = _local_peak_envelope(expanded_channel)
```

### `_local_rms_envelope(signal, block_size=2048) -> np.ndarray`
**File:** fat_llama_fftw/audio_fattener/feed.py:501
**Kind:** function (private helper)
**Description:** Same WOLA machinery as `_local_peak_envelope`, but each block's own scalar is its RMS (`sqrt(mean(x**2))`) instead of its peak. Kept as a separate function for the same zero-risk-to-existing-behavior reason `_local_peak_envelope` is separate from `iterative_soft_thresholding`'s own WOLA loop. Used by both of `_cap_ist_changes_to_baseline_peak`'s gates (the "helps or hurts" gate and this cycle's new silence gate) — RMS rather than peak because a peak-based version of the "helps or hurts" gate was measured to be noisy at a quiet, oscillating signal's zero crossings, mis-classifying several hundred samples right at a fade-in's quietest point.
**Parameters:**
- `signal` (`np.ndarray`): the signal to estimate an RMS envelope for.
- `block_size` (`int`): analysis/synthesis window length in samples. Default `2048`.
**Returns:** `np.ndarray` (`float32`), same length as `signal`, non-negative.
**Usage:**
```python
env = _local_rms_envelope(combined_candidate, block_size=2048)
```

### `_cap_ist_changes_to_baseline_peak(expanded_channel, ist_changes, max_rounds=20, dominant_band_ratio=0.002, safety_margin=1.2, gate_block_size=2048, gate_steepness=10.0, silence_floor_ratio=0.005, silence_gate_power=6.0) -> np.ndarray`
**File:** fat_llama_fftw/audio_fattener/feed.py:562
**Kind:** function (private helper)
**Description:** Bounds how much `perform_ist_iteration`'s peak-relative boost of a channel's dominant content is allowed to inflate the combined (interpolation + IST) signal's peak above the pre-IST baseline's own peak — an inflated peak here is what determines how hard every frequency gets divided down at final normalization, not just what IST touched. Splits `ist_changes`'s own `rfft` spectrum into a "dominant" component (bins `>= dominant_band_ratio` of the spectrum's own peak) and a "residual" component; only the dominant component is iteratively shrunk (bounded multiplicative-shrink loop, `max_rounds`) toward bounding the combined peak, leaving residual (genuinely-added quiet/high-frequency detail) untouched. Falls back to one additional whole-signal uniform-shrink pass if the frequency-selective candidate's own combined peak still exceeds `baseline_peak * safety_margin`.

How much of that dominant-band correction actually applies is decided by **two independent gates**, applied in sequence:

1. **"Helps or hurts" gate** (replaces a retired `onset_gate_ratio` global-peak-ratio design). Computes `_local_rms_envelope` (block size `gate_block_size`) of both the fully-uncorrected and fully-corrected combined signal; `local_ratio = env_corrected / env_uncorrected`; `gate = clip(1 - gate_steepness * (local_ratio - 1), 0, 1)` — `ratio <= 1` (correction doesn't raise local level) saturates the gate to `1` (full correction); `ratio > 1` (correction raises local level — the "unmasking" signature of breaking a real fade-in's near-cancellation) ramps the gate toward `0`. This directly measures, per local region, whether applying the correction helps or hurts, rather than inferring it from a single global envelope/peak ratio that could not distinguish "quiet because dominant/residual genuinely cancel" (a real fade-in) from "quiet with real uncancelled excess" (an ordinary quiet passage) — both situations sit at a similarly low envelope/peak ratio on real material. Measured on real `input_test.mp3`: quiet-passage elevation (vs. a no-IST control) dropped from mean +2.0/max +2.5dB to mean +0.45/max +1.9dB, while the original synthetic fade-in regression bound still holds at ~0.42dB elevation (well under its 3dB bound).
2. **Silence gate** (added the prior cycle). The "helps or hurts" gate above only ever *chooses between* two non-zero candidates — it cannot suppress `ist_changes` altogether, and `residual_component` is never touched by it at all. On a genuine digital-silence region (e.g. a track's true lead-in, not just "quiet"), *both* candidates still carry non-zero low-level content from `perform_ist_iteration`'s own hard-threshold FFT ringing on a block whose real content is at/below the noise floor — neither of the gate's two choices is actually correct there. This gate is scoped to whether there is any genuine signal at all, measured from `expanded_channel` itself (not from either candidate): `silence_gate = clip(local_rms(expanded_channel) / baseline_peak / silence_floor_ratio, 0, 1) ** silence_gate_power`, multiplying the *whole* `candidate_result` (both components). `silence_floor_ratio=0.005` sits strictly below the fade-in fixture's own smallest ratio (0.005685) so it cannot interfere with the "helps or hurts" gate's onset protection; `silence_gate_power=6.0` (vs. a linear ramp, which only got the measured lead-in elevation from +3.7dB to +2.3dB) reduces it to ~+0.17dB. Same gating is applied to the safety-net fallback's own correction, for consistency.

**This cycle (comment-only, no logic change):** two more investigations of the dominant/residual split itself — whether it, rather than `perform_ist_iteration`'s own retention, was the bottleneck for genuine high-frequency detail — were tested and disproven directly against real material: 0 of 457,051 real-block `rfft` bins `>=2kHz` are ever classified "dominant" (so the split isn't discarding HF content — there's none in the dominant bucket to discard), and the untouched residual bucket's HF content is measurably uncorrelated with the lossless reference (`|corr| < 0.2`, measured ~0.01-0.04) — i.e. unshrunk leakage, not genuine detail either. See `perform_ist_iteration`'s own factblock above for the companion finding (kept-bin mask never selects HF bins at all).

**Returns:** `np.ndarray` — `ist_changes`, reshaped/rescaled per the above (unchanged if no capping was needed).
**Parameters:**
- `expanded_channel` (`np.ndarray`): the pre-IST interpolated baseline for this channel.
- `ist_changes` (`np.ndarray`): IST's contribution, as returned by `iterative_soft_thresholding`.
- `max_rounds` (`int`): bound on shrink-loop iterations. Default `20`.
- `dominant_band_ratio` (`float`): fraction of the spectrum's own peak magnitude a bin must reach to be classified "dominant". Default `0.002`.
- `safety_margin` (`float`): the safety-net fallback only engages if the frequency-selective candidate's own combined peak still exceeds `baseline_peak * safety_margin`. Default `1.2`.
- `gate_block_size` (`int`): WOLA block size for both gates' local-RMS estimates. Default `2048`.
- `gate_steepness` (`float`): how sharply the "helps or hurts" gate ramps from `1` to `0` as `local_ratio` exceeds `1`. Default `10.0`.
- `silence_floor_ratio` (`float`): **new this cycle.** Local-RMS-to-peak ratio floor below which the silence gate engages. Default `0.005`.
- `silence_gate_power` (`float`): **new this cycle.** Exponent shaping the silence gate's falloff below the floor. Default `6.0`.
**Usage:**
```python
capped = _cap_ist_changes_to_baseline_peak(expanded_channel, ist_changes)
```

### `_limit_combined_peak_to_baseline(combined, baseline_peak, block_size=2048) -> np.ndarray`
**File:** fat_llama_fftw/audio_fattener/feed.py:1022
**Kind:** function (private helper)
**Description:** **New this cycle** (iterate-fat-llama, targeting audio-quality-checker's `test_no_net_band_attenuation_vs_resample_control` finding). Locally trims the rare residual peak overshoot that `_cap_ist_changes_to_baseline_peak`'s bounded shrink loop (`max_rounds=20`, deliberately not fully converged) leaves behind — only ~0.003-0.01% of samples per channel exceed `baseline_peak` after that cap, but the mandatory final `normalize_signal` divides the *entire* channel by whatever its actual peak is, so that sparse excess becomes a uniform ~0.36-0.39dB attenuation tax on every sample, including mid/high bands IST never boosted. Reuses `_local_peak_envelope`'s existing WOLA machinery (no new mechanism) to compute a per-sample gain of `min(1, baseline_peak / local_envelope)` — exactly `1.0` (bit-identical passthrough) everywhere the local envelope doesn't exceed `baseline_peak`, and reduced only in the rare regions that do. Applied to `combined` (`expanded_channel + capped ist_changes`) in `_process_channel`, after and outside `_cap_ist_changes_to_baseline_peak`, so that function's own frequency-selective split/gates are unchanged.
**Parameters:**
- `combined` (`np.ndarray`): the interpolated + IST-added per-channel signal, pre-trim.
- `baseline_peak` (`float`): `expanded_channel`'s own pre-IST peak absolute value; the ceiling this function trims `combined` back toward.
- `block_size` (`int`): WOLA envelope block size, passed through to `_local_peak_envelope`. Default `2048`.
**Returns:** `np.ndarray` (`float32`), same shape as `combined`, with local peaks trimmed to `baseline_peak` (passthrough where already within bounds).
**Usage:**
```python
combined = _limit_combined_peak_to_baseline(combined, baseline_peak)
```

### `_process_channel(channel, upscale_factor, max_iter, threshold) -> np.ndarray`
**File:** fat_llama_fftw/audio_fattener/feed.py:1093
**Kind:** function (private helper)
**Description:** The full per-channel pipeline — interpolate (`new_interpolation_algorithm`) → IST (`iterative_soft_thresholding`) → peak-cap (`_cap_ist_changes_to_baseline_peak`) → local peak trim (`_limit_combined_peak_to_baseline`, new this cycle) → combine — factored out so `upscale_channels` can dispatch it either sequentially or on a worker thread. Reads/writes only its own `channel` argument (no shared state), so channels are provably safe to run concurrently.
**Parameters:**
- `channel` (`np.ndarray`): 1-D single-channel samples.
- `upscale_factor` (`int`), `max_iter` (`int`), `threshold` (`float`): passed through to the pipeline stages.
**Returns:** `np.ndarray` (`float32`) — this channel's fully processed (interpolated + IST + capped + peak-trimmed) samples.

### `upscale_channels(channels, upscale_factor, max_iter, threshold) -> np.ndarray`
**File:** fat_llama_fftw/audio_fattener/feed.py:1112
**Kind:** function
**Description:** Runs `_process_channel` per channel and stacks the results back into a single 2-D array. For multi-channel (e.g. stereo) input, dispatches each channel's independent work across a `ThreadPoolExecutor` instead of a sequential loop. Mono input skips the thread pool.
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
**File:** fat_llama_fftw/audio_fattener/feed.py:1157
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
**File:** fat_llama_fftw/audio_fattener/feed.py:1161
**Kind:** function
**Description:** Zeroes every FFT bin above `original_nyquist` (via `pyfftw` `rfft`/`irfft`, multi-threaded per `_fft_thread_count` for large signals), enforcing `.claude/rules/project-mission.md`'s hard "no content above the original Nyquist frequency" constraint. Runs after amplitude auto-scaling but *before* `upscale()`'s final `normalize_signal` pass.
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
**File:** fat_llama_fftw/audio_fattener/feed.py:1187
**Kind:** function
**Description:** The package's public entry point (README's documented API). Reads the source file, computes an `upscale_factor` from `target_bitrate_kbps` vs. the source's own bitrate (floored at `1`), runs `upscale_channels` per channel, auto-scales each channel back to its original peak, runs `apply_nyquist_cutoff`, and finally normalizes (normalization strictly last so the cutoff's own Gibbs overshoot can't push samples above full scale on write). No `toggle_*` flags exist on this signature — auto-scaling, the Nyquist cutoff, and normalization always run unconditionally. Unchanged this cycle.
**Parameters:**
- `input_file_path` (`str`): source audio file path.
- `output_file_path` (`str`): destination file path.
- `source_format` (`str`): `'mp3'`, `'wav'`, `'ogg'`, or `'flac'`.
- `target_format` (`str`): `'flac'` or `'wav'`. Default `'flac'`.
- `max_iterations` (`int`): IST iteration upper bound. Default `800`.
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
**File:** fat_llama_fftw/tests/test_feed.py:37
**Kind:** class
**Description:** `unittest.TestCase` covering every function in `feed.py`, 64 test methods (+2 this cycle: permanent regression tests documenting a 6th/7th independent investigation of the added-detail-above-4kHz gap, both refuted — see below). Notable groups: I/O, interpolation, IST core, `_local_peak_envelope` (3 tests), `_local_rms_envelope` (4 tests), the peak cap (`_cap_ist_changes_to_baseline_peak` — 7 tests), `_limit_combined_peak_to_baseline` (3 direct + 2 wiring/real-material), 2 net-attenuation/boost-bound tests, the added-detail investigation regression tests (2), the Nyquist cutoff, and full `upscale()` wiring/edge cases.
**Usage:**
```python
python -m unittest discover -s fat_llama_fftw/tests
```

### `TestFeed.test_local_rms_envelope_empty_signal(self)` / `test_local_rms_envelope_short_signal_returns_constant_rms(self)` / `test_local_rms_envelope_tracks_loud_and_quiet_regions(self)` / `test_local_rms_envelope_differs_from_peak_envelope_on_bursty_signal(self)`
**File:** fat_llama_fftw/tests/test_feed.py:562, 572, 581, 611
**Kind:** method (4 tests, unchanged this cycle)
**Description:** Direct unit coverage for `_local_rms_envelope`: empty-signal edge case, short-signal constant-RMS passthrough, tracking distinct loud/quiet regions, and a direct RMS-vs-peak disagreement test (a brief loud burst inside an otherwise-silent block: peak envelope is dominated by the burst, RMS envelope stays low) proving the two are not interchangeable proxies.
**Returns:** `None` (assertion-based, each).

### `TestFeed.test_cap_ist_changes_to_baseline_peak_noop_when_not_needed(self)` / `..._bounds_combined_peak(self)` / `..._preserves_quiet_band(self)` / `..._pathological_ratio_falls_back(self)` / `..._zero_baseline(self)`
**File:** fat_llama_fftw/tests/test_feed.py:628, 639, 687, 742, 765
**Kind:** method (5 tests, unchanged this cycle)
**Description:** No-op/zero-baseline short-circuits, the original attenuation-fix regression bound, the frequency-selective dominant/residual split's quiet-band-survival guard, and the pathological-`dominant_band_ratio` safety-net fallback. All still pass unmodified against this cycle's new silence gate (which only engages when the pre-IST baseline itself has essentially no signal — none of these fixtures' baselines are silent).
**Returns:** `None` (assertion-based, each).

### `TestFeed.test_cap_ist_changes_to_baseline_peak_envelope_gates_quiet_onset(self)`
**File:** fat_llama_fftw/tests/test_feed.py:774
**Kind:** method
**Description:** Broadband, raised-cosine fade-in synthetic channel run through the real pipeline; asserts onset-window RMS elevation vs. the pre-IST baseline stays under 3dB — the regression guard for genuine near-cancelling-onset protection. **Re-verified this cycle** against the new silence gate: still passes at ~0.42dB, unchanged — this fixture's own local-RMS-to-peak ratio (0.005685) sits just above `silence_floor_ratio` (0.005), the value chosen specifically so the silence gate cannot engage on this fixture.
**Returns:** `None` (assertion-based).

### `TestFeed.test_cap_ist_changes_to_baseline_peak_does_not_elevate_quiet_real_material(self)`
**File:** fat_llama_fftw/tests/test_feed.py:849
**Kind:** method
**Description:** **Restructured this cycle** into 2 `subTest` cases against real `input_test.mp3` slices at the pinned baseline config (`upscale_factor=7, max_iterations=300, threshold_value=0.6`), each processing a properly long, WOLA-blocked slice and measuring broadband RMS elevation vs. a no-IST interpolation-only control in a specific window: (1) *"ordinary quiet passage (0.5-1.0s)"* — measures the whole processed window, bound 1.5dB (the prior cycle's fix target; still passes at ~0.24-0.29dB). (2) *New this cycle: "genuine digital-silence lead-in (0.0-0.02s)"* — processes from the true start of the file but measures only the first 0.02s of output (where the residual concentrated), bound 1.0dB (passes at ~0.17dB with the new silence gate; was ~+3.7dB before it).
**Returns:** `None` (assertion-based; both subTest cases must pass).

### `TestFeed.test_limit_combined_peak_to_baseline_passes_through_when_not_needed(self)` / `..._zero_baseline_passthrough(self)` / `..._bounds_a_brief_spike(self)`
**File:** fat_llama_fftw/tests/test_feed.py:1201, 1214, 1221
**Kind:** method (3 tests, new this cycle)
**Description:** Direct unit coverage for `_limit_combined_peak_to_baseline`: bit-identical passthrough when the local envelope never exceeds `baseline_peak`, a `baseline_peak<=0` short-circuit, and confirmation that a brief synthetic spike above `baseline_peak` gets bounded while the rest of the signal is untouched.
**Returns:** `None` (assertion-based, each).

### `TestFeed.test_process_channel_wires_peak_limit_after_cap(self)`
**File:** fat_llama_fftw/tests/test_feed.py:1252
**Kind:** method (new this cycle)
**Description:** End-to-end wiring check via `upscale_channels` that `_limit_combined_peak_to_baseline` is actually applied after `_cap_ist_changes_to_baseline_peak` in `_process_channel` — a tight `baseline_peak*1.01` bound, tighter than the pre-existing `test_upscale_channels_caps_combined_peak_close_to_baseline_stereo`'s looser `*1.25` bound.
**Returns:** `None` (assertion-based).

### `TestFeed.test_upscale_channels_does_not_net_attenuate_mid_band_real_material(self)`
**File:** fat_llama_fftw/tests/test_feed.py:1280
**Kind:** method (new this cycle)
**Description:** Real-material regression test against `input_test.mp3` at the pinned baseline config, same slice/methodology as the pre-existing `test_upscale_channels_ist_does_not_net_attenuate_real_material`, scoped specifically to the 400-4000Hz band that `audio-quality-checker`'s `test_no_net_band_attenuation_vs_resample_control` flagged. Asserts `mid_ratio_db > -0.05dB` against a plain-resample control — the fix measured +0.01 to +0.07dB on this exact slice.
**Returns:** `None` (assertion-based).

### `TestFeed.test_cap_ist_changes_dominant_split_does_not_bottleneck_hf_content(self)`
**File:** fat_llama_fftw/tests/test_feed.py:1352
**Kind:** method (new this cycle)
**Description:** Cycle-3 investigation regression test, against real `input_test.mp3`/`input_test.flac` at the pinned baseline config. Proves `_cap_ist_changes_to_baseline_peak`'s dominant/residual FFT split is not the bottleneck for genuine high-frequency detail: 0 of 457,051 real-block `rfft` bins `>=2kHz` are ever classified "dominant" (so nothing HF is being shrunk there), and the untouched residual bucket's own HF content is measurably uncorrelated with the lossless reference (`|corr| < 0.2`, measured ~0.01-0.04) — i.e. it's unshrunk leakage, not genuine recoverable detail either.
**Returns:** `None` (assertion-based).

### `TestFeed.test_perform_ist_iteration_excludes_hf_and_converges_in_one_pass_on_real_block(self)`
**File:** fat_llama_fftw/tests/test_feed.py:1483
**Kind:** method (new this cycle)
**Description:** Cycle-3 investigation regression test, companion to the one above. Proves on a real programme-material block that `perform_ist_iteration`'s kept-bin mask is bit-for-bit identical between pass 0 and pass 1 (i.e. the fixed point is reached in a single pass), and that 0 of thousands of possible bins `>=2kHz` are ever kept — so `max_iterations`/convergence behavior cannot recover HF content either, since none is ever selected in the first place.
**Returns:** `None` (assertion-based).

### `TestFeed.test_iterative_soft_thresholding_block_size_does_not_change_hf_correlation_with_reference(self)`
**File:** fat_llama_fftw/tests/test_feed.py:1565
**Kind:** method (new this cycle)
**Description:** 6th independent investigation regression test (this iterate-fat-llama run's cycle 2, targeting the added-detail-above-4kHz gap with a genuinely new angle: block/WOLA size). Sweeps `iterative_soft_thresholding(..., block_size=X)` at `{2048, 8192, 32768}` through the full real pipeline tail against real `input_test.mp3`/`input_test.flac`; asserts per-band gain measurably shifts with `block_size` (proving the sweep isn't vacuous) while 8-19kHz envelope correlation with the lossless reference stays low (`<0.2`) at every size — block size is not the bottleneck either.
**Returns:** `None` (assertion-based).

### `TestFeed.test_iterative_soft_thresholding_noise_floor_relative_threshold_adds_noise_not_detail(self)`
**File:** fat_llama_fftw/tests/test_feed.py:1675
**Kind:** method (new this cycle)
**Description:** 7th independent investigation regression test (this run's cycle 2, second new angle: a two-pass per-bin noise-floor-relative threshold, self-contained/never wired into production). Against real material, asserts this alternative mechanism measurably raises the 16-20kHz band (`>3dB`, proving it's a real effect) while its HF content's correlation with the lossless reference stays low (`<0.2`) — i.e. it indiscriminately amplifies noise rather than recovering genuine detail, the strongest evidence yet that the MP3 decoder's HF content sits at its own near-zero dequantization floor (~5.6e5x smaller than the LF floor) rather than being quiet-but-recoverable signal.
**Returns:** `None` (assertion-based).

### `TestFeed.test_upscale_channels_caps_combined_peak_close_to_baseline_stereo(self)`
**File:** fat_llama_fftw/tests/test_feed.py:944
**Kind:** method
**Description:** End-to-end (stereo, 2 distinct-content channels) wiring check that `upscale_channels` itself keeps each channel's combined peak close to its own pre-IST interpolation baseline. Unchanged this cycle.
**Returns:** `None` (assertion-based).

### `TestFeed.test_local_peak_envelope_empty_signal(self)` / `test_local_peak_envelope_short_signal_returns_constant_peak(self)` / `test_local_peak_envelope_tracks_loud_and_quiet_regions(self)`
**File:** fat_llama_fftw/tests/test_feed.py:510, 522, 532
**Kind:** method (3 tests, unchanged this cycle)
**Description:** Direct unit coverage for `_local_peak_envelope` (still used elsewhere/kept for future use, though `_cap_ist_changes_to_baseline_peak`'s own gates use `_local_rms_envelope` instead): empty-signal edge case, short-signal constant-peak passthrough, tracking loud vs. quiet regions.
**Returns:** `None` (assertion-based, each).

### `TestFeed.test_upscale_channels_ist_does_not_net_attenuate_untouched_band(self)` / `test_upscale_channels_ist_does_not_net_attenuate_real_material(self)`
**File:** fat_llama_fftw/tests/test_feed.py:983, 1080
**Kind:** method (2 tests, unchanged this cycle)
**Description:** Net-attenuation regression tests with a symmetric low-frequency/low-band boost bound (from a prior cycle's `onset_gate_ratio` regression). Still pass against this cycle's silence-gate addition.
**Returns:** `None` (assertion-based, each).

### Other test methods (unchanged since prior cycles)
**File:** fat_llama_fftw/tests/test_feed.py
**Kind:** method (39 additional tests)
**Description:** `test_read_audio`, `test_read_audio_unsupported_format_computes_bitrate_from_duration`, `test_read_audio_does_not_corrupt_channel_counts_above_stereo`, `test_write_audio`, `test_write_audio_wav_uses_wav_container`, `test_write_audio_rejects_unsupported_format`, `test_write_audio_roundtrip_preserves_audio_properties_on_disk`, `test_new_interpolation_algorithm(_identity_at_upscale_factor_one|_odd_length_passthrough|_no_imaging_above_original_nyquist)`, `test_initialize_ist(_scales_with_data_magnitude|_zero_signal_stays_zero)`, `test_upscale_channels(_thresholded_out_is_pure_interpolation|_parallel_matches_sequential_per_channel|_mono_skips_thread_pool)`, `test_iterative_soft_thresholding_chains_across_iterations`, `test_iterative_soft_thresholding_stops_early_once_converged`, `test_ist_adds_content_distinct_from_input_at_realistic_scale`, `test_ist_pipeline_is_scale_invariant_within_float32_precision`, `test_perform_ist_iteration_never_keeps_dc_bin`, `test_iterative_soft_thresholding_output_has_no_dc_offset`, `test_ist_no_dc_offset_on_real_programme_material`, `test_iterative_soft_thresholding_blocks_restore_multiple_bands`, `test_iterative_soft_thresholding_no_block_boundary_discontinuity`, `test_feed_block_boundary_test_fixture_is_representative`, `test_no_block_hop_boundary_curvature_bump`, `test_apply_nyquist_cutoff_removes_image_content`, `test_fft_thread_count_scoped_to_large_transforms`, `test_apply_nyquist_cutoff_requests_multiple_threads_for_large_signal`, `test_upscale_wires_apply_nyquist_cutoff`, `test_upscale_clamps_upscale_factor_to_one_for_high_bitrate_source`, `test_upscale_factor_one_from_moderately_high_bitrate_source_no_warning`, `test_upscale_output_never_exceeds_full_scale`, `test_upscale_writes_valid_flac_end_to_end_on_disk`, `test_normalize_signal`. All still pass against this cycle's changes.
**Returns:** `None` (assertion-based, each).

## example.py

### Module-level script
**File:** example.py:1
**Kind:** script (no functions/classes)
**Description:** The README-documented usage example — calls `upscale()` once against the repo-root `input_test.mp3` → `output_test.flac` with `max_iterations=600, threshold_value=0.75, target_bitrate_kbps=1400`. A comment (added when v2.0.0 shipped) notes the IST peak-cap's behavior changed internally; `upscale()`'s own public signature has gained no new parameters across any of these cycles.
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

- **This cycle (cycle 3, no behavior change):** a 4th independent investigation of the "no added detail below Nyquist" gap (see below) — comment-only additions plus 2 new permanent regression tests; no source logic changed.
- **Prior cycle's fix (cycle 2):** `_cap_ist_changes_to_baseline_peak` gained a second, independent "silence gate" (new `silence_floor_ratio`/`silence_gate_power` params) closing a residual audio-quality-checker finding left over after cycle 1's "helps or hurts" gate fix — a genuine digital-silence lead-in (the track's very first ~0.02s) was still measurably elevated (~+3.7dB) because the existing gate could only choose between two non-zero candidates, never suppress `ist_changes` outright. Measured fix: same window now ~+0.17dB, with the fade-in-onset regression test and every other existing regression guard unaffected.
- **Now considered a settled, closed investigation (not a "still open" item to keep re-attempting):** no IST-attributable added detail below the original Nyquist frequency. Four independent investigations across two `iterate-fat-llama` runs — a `threshold_value=0.15` experiment, a per-band FFT-thresholding variant, and (this run) whether the peak cap's dominant/residual split or IST's convergence/iteration-count behavior was the bottleneck — have all converged on the same conclusion, the last two demonstrated directly against the lossless reference: IST's single hard peak-relative threshold reaches its fixed point in one pass and structurally never selects high-frequency content on real programme material, so there is no genuine detail anywhere downstream to recover — closing this without violating the no-synthesized-content constraint is not achievable via any peak-relative/hard-threshold IST variant. Only a fundamentally different DSP mechanism would be worth a future attempt.
- **Still open (per DIRECTIVES, explicitly out of scope for any skill/agent):** `.claude/agents/rules/audio-quality.md`'s pinned baseline config still references `toggle_normalize`/`toggle_autoscale`/`toggle_adaptive_filter` kwargs and an `lms_filter`/~20-minute runtime estimate that don't match this package's actual `upscale()` signature or current (~3-4 second) runtime. This file is under `.claude/`, a hard exclusion — "edited by a human, never by a running skill or agent" — so no automated run can fix it; needs a direct human edit.
- **Environment hygiene finding (informational, not a source bug, repeatedly re-flagged):** a stale `fat_llama_fftw` 1.0.4 is pip-installed into `venv/Lib/site-packages`, shadowing the repo source for any script run from outside the repo root. Not fixed (out of any skill's write scope) — a human should `pip uninstall`/reinstall in editable mode.
- **Still open, flagged by prior cycles (out of `fat_llama_fftw/**` write scope):** README.md's Algorithm Explanation still describes the peak cap in terms that predate the frequency-selective/envelope-gated refinement and the two gate mechanisms now in place — needs a human or a permitted direct edit to reconcile.
- **Still open, methodology/human decision, not a source bug:** the repo-root reference asset `input_test.flac` is itself a stale output of this same pipeline from before the original Nyquist-cutoff fix, so it still carries above-Nyquist imaging — spectral-deviation scoring against it partly measures agreement with this defective reference rather than fidelity to an independent lossless master.
- `upscale()` end-to-end coverage exists (`test_upscale_wires_apply_nyquist_cutoff`, `test_upscale_writes_valid_flac_end_to_end_on_disk`) including a real on-disk write/readback test.
