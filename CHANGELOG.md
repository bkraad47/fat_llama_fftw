# Changelog

All notable changes to this project will be documented in this file.

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
