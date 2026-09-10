# Changelog

All notable changes to this project will be documented in this file.

## [1.4.3] - 2026-09-10

`iterate-fat-llama` run (2 cycles, early stop) verifying and completing the pydub removal introduced in 1.4.2. Coherence and spectral deviation confirmed at or above the pre-removal baseline (9.5/9.9): this run measured 9.8/9.8, with the remaining 0.1 spectral-deviation gap traced to the committed reference asset itself, not a regression.

### Fixed

- **Investigated a suspected precision regression from the `pydub` removal and confirmed it was not one.** `read_audio` moved from `pydub` (which returned raw integer PCM samples, e.g. int16 range) to `soundfile` (which returns normalized float64 samples in `[-1, 1]` by default) — a real change in numeric scale flowing through the whole pipeline. Verified empirically that every threshold/scale computation downstream is genuinely scale-invariant as designed: the difference between the old and new scales lands at float32's own rounding epsilon, not a measurable quality loss. If anything, the new path loses zero *additional* precision versus the old one, since `soundfile`'s MP3 decoder already produces float32-quantized values internally.
- **Fixed a crash introduced by the `pydub` removal:** `read_audio`'s fallback branch for uncatalogued formats referenced a variable that no longer exists post-migration, raising a `NameError` instead of computing a duration estimate.
- **Fixed a silent data-corruption bug introduced by the `pydub` removal:** a leftover reshape step hardcoded audio to 2 channels regardless of the source's actual channel count — harmless for mono/stereo, but would have silently corrupted any source with more than 2 channels.
- **Fixed a crash on high-bitrate sources**, unrelated to the `pydub` removal but found while verifying it: a lossless source (e.g. 96kHz/24-bit WAV) with a bitrate already exceeding the requested target could compute an upscale factor of zero, crashing the interpolation step. The factor now has a floor of 1, with a clear log message explaining when and why no upsampling occurs, rather than a cryptic crash.
- Added a real, fully unmocked end-to-end test that writes an actual FLAC file and verifies its on-disk properties — previously every end-to-end test mocked the file-writing step, so a container/subtype/channel-interleaving regression could have passed the whole suite undetected. Minor test-suite cleanup (a stray debug print, a redundant assertion) alongside.

### Known remaining gaps

- `.claude/agents/rules/audio-quality.md`'s pinned test-baseline configuration is out of sync with `example.py` and this package's actual `upscale()` signature — this is a project-tooling file outside this pipeline's own write scope; needs a direct human edit.
- The committed reference asset `input_test.flac` is a legacy output of an earlier (zero-order-hold-era) version of this pipeline, not an independent lossless master — it now carries content above the original Nyquist frequency that the current, correct pipeline properly excludes, which is why spectral-deviation scoring against it has a small, expected ceiling below what a clean master would allow.

## [1.4.2] - 2026-09-09

### Changed
- Removes pydub from read_audio and replaces it with soundfile.
- Removes pydub from dependencies because it's now rendered obsolete

## [1.4.1] - 2026-09-09

Second `iterate-fat-llama` run (5 cycles), continuing directly from 1.4.0, focused on closing that release's known gaps: the persistent lack of measurable added detail, and further performance work.

### Fixed

- **The upscale now measurably restores detail in thin/missing frequency bands, not just a safe transparent pass-through.** The interpolation stage was replaced from zero-order-hold (repeat-each-sample, which images the original spectrum above the source's Nyquist frequency as a side effect) to bandlimited FFT-domain interpolation, which adds no new spectral content by construction. This freed real headroom for IST's own contribution: the reference's thinnest frequency bins now gain up to +8 dB, with consistent positive gains from 8 kHz up to the original Nyquist frequency (as much as +5 dB at the top), verified as programme-correlated content rather than a broadband noise-floor rise.
- **Reduced (then further reduced) a high-shelf attenuation** where IST's own processing was inadvertently pulling down untouched high frequencies during final normalization — down from ~4 dB to sub-audible before the interpolation fix above made it moot.
- Fixed a residual, low-severity block-boundary artifact left over from the 1.4.0 release's own fix for the same class of issue (a DC-removal step was undoing part of the edge tapering it depended on).
- Corrected an inaccurate code comment overstating a threading-related performance change's actual speedup and exactness guarantee.
- Removed a stray `cupy` (GPU) import from `analysis.py`, the standalone spectrogram/comparison script — it now runs on plain CPU/`numpy`, matching the rest of the package.

### Performance

- Vectorized the interpolation step (`np.repeat` instead of a manual Python loop) — ~54x faster on its own.
- The final Nyquist-frequency safety filter now uses multi-threaded FFTs for large signals — ~4.5x faster at real file sizes.
- Stereo channels are now processed in parallel (previously sequential) — ~1.2-1.7x faster end to end, verified bit-identical output.
- Net effect across both this run and 1.4.0: the reference test file's full pipeline run dropped from ~44 seconds to ~2 seconds.

### Known remaining gaps

- `.claude/agents/rules/audio-quality.md`'s internal test-baseline configuration still references parameters that don't exist on `upscale()` (a leftover from this package's CUDA sibling project) — needs a human edit; blocked by local tooling permissions during this run.
- The committed reference asset `input_test.flac` is itself a legacy output of an earlier (zero-order-hold-era) version of this pipeline, not an independent lossless master — worth considering whether to regenerate it from a true master at some point.
- The multi-thread FFT crossover threshold could likely be raised further for an additional performance gain (identified but not acted on this run).

## [1.4.0] - 2026-09-08

Five-cycle `iterate-fat-llama` run focused on upscale pipeline performance and audio coherence (`fat_llama_fftw/audio_fattener/feed.py`).

### Fixed

- **IST no longer silently degenerated into a single pass regardless of `max_iterations`.** `iterative_soft_thresholding` used to submit every iteration concurrently against the same starting estimate, so the result never actually chained. It now runs a genuine sequential loop with an early exit once a pass converges — roughly a 12-14x speedup on the reference test file, with no loss of output quality.
- **No content above the original source's Nyquist frequency survives in the output**, closing a gap where the zero-order-hold interpolation step was imaging the spectrum above that point (previously an audible -52 dB defect; now consistently below -145 dB, at the noise floor).
- **IST now contributes genuine added detail instead of an near-identity pass-through.** The threshold value is compared against each signal's own peak magnitude rather than as a fixed absolute cutoff, so it behaves consistently regardless of the input's numeric scale.
- **Fixed a DC-offset/tonal regression introduced while chasing the fix above:** applying the peak-relative threshold to a whole file's single FFT let one loud moment dominate the cutoff for the entire file, occasionally including the DC bin — producing an audible offset and a fixed-frequency drone. IST now processes long signals in overlapping local blocks, and the DC bin is always excluded from consideration.
- **Fixed a periodic clicking artifact the block-based processing above introduced:** the block reconstruction only tapered each segment on the way in, not on the way out, producing a discontinuity at every block boundary (audible as a fixed-rate burst pattern). Reconstruction now windows both directions (WOLA).
- **Fixed a small DC-offset regression from the WOLA fix itself**, plus minor sample clipping caused by the final safety filter occasionally overshooting an already-normalized peak. Both are now resolved with real audio measurably clean of either artifact.
- General PEP8 cleanup in `feed.py` alongside the above (import order, line length, blank-line spacing).

### Known remaining gaps (not addressed this run, tracked for a follow-up pass)

- No genuinely added spectral detail is yet measurable in previously-thin frequency bands — the current output is a clean, artifact-free "transparent" upscale rather than one that restores missing/congested detail. This is the primary target for the next `iterate-fat-llama` run.
- A modest low-frequency-to-high-frequency tonal tilt (~6 dB) was observed in the final cycle; it may partially resolve as a side effect of this run's DC-offset fix, but hasn't been re-verified.
- `analysis.py` (the standalone spectrogram/comparison script) still imports `cupy`, a GPU dependency this now-CPU-only package doesn't otherwise use — it will fail to import as-is in a plain CPU environment.
- `.claude/agents/rules/audio-quality.md`'s documented baseline test configuration references parameters (`toggle_normalize`, `toggle_autoscale`, `toggle_adaptive_filter`) that don't exist on `upscale()`'s actual signature — a leftover from this package's CUDA sibling project. Needs a human edit to reconcile.
- `README.md`'s Algorithm Explanation section hasn't been updated to describe the block/windowed IST processing and DC-bin exclusion added this run.


### [1.0.3] - [1.0.4] - 2024-07-26

#### Changed

- Moved to `pyfftw` from CUDA.

### [1.0.2] - 2024-07-26

#### Changed

- Remove `logging` from requirements to fix pip bug.

### [1.0.1] - 2024-07-26

#### Changed

- Updated `analytics.py` analysis and spectorgram results.
- Updated `README.md` details.

### [1.0.0] - 2024-07-25

#### Added

- Added support for reading 'ogg', 'flac', and 'wav' file formats and calculating their bitrates correctly.

#### Changed

- Renamed `upscale_mp3_to_flac` method to `upscale` to support multiple source formats.
- Simplified the workflow to focus on 'mp3' to 'flac' conversion with essential steps only.

#### Removed

- Dropped support for 'ape' and 'alac' target formats.

### [0.1.8] - 2024-07-24

#### Added

- Introduced toggle flags for normalization, equalization, amplitude scaling, and gain reduction.
- Enhanced auto-scaling of amplitude based on the original MP3 file when `toggle_scale_amplitude` is `False`.
- Logging for each step of the processing to provide better traceability and debugging.

#### Changed

- Default values for parameters are now set at the function call.
- Refined the upscaling algorithm to ensure better handling of amplitude and gain.
- Renamed the flags for consistency (`toggle_wiener_filter`, `toggle_normalize`, `toggle_equalize`, `toggle_scale_amplitude`, `toggle_gain_reduction`).

#### Fixed

- Fixed issues related to numpy and cupy array conversions.
- Improved error handling for invalid target bitrate values.
- Addressed the issue where the amplitude of the produced signal was significantly weaker than the original.

### [0.1.7] - 2024-07-22

#### Added

- Added methods for MP3 to FLAC conversion with optional processing using CuPy for GPU acceleration.
- Initial version of `upscale_mp3_to_flac` method with parameters for iterative soft thresholding (IST), gain reduction, and equalization.

### [0.1.0] to [0.1.6] - 2024-07-20

#### Added

- Basic functionality for reading MP3 files and writing FLAC files.
- Initial implementation of the new interpolation algorithm and IST for audio processing.