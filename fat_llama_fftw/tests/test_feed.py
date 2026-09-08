import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from fat_llama_fftw.audio_fattener.feed import (
    read_audio,
    write_audio,
    new_interpolation_algorithm,
    initialize_ist,
    perform_ist_iteration,
    iterative_soft_thresholding,
    upscale_channels,
    normalize_signal,
    apply_nyquist_cutoff,
    upscale
)
import fat_llama_fftw.audio_fattener.feed as feed_module

# Repo-root reference asset, already used by example.py/README's documented
# usage. A handful of new cycle-5 regression tests below need genuine
# programme material (not a synthetic fixture) to exercise the WOLA
# synthesis-window DC leak and the Nyquist-cutoff/normalize ordering bug -
# both were found by measuring against this file directly, and a
# white-noise-plus-spikes synthetic fixture provably does not trigger the
# former (see test_ist_no_dc_offset_on_real_programme_material's own
# docstring).
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
_INPUT_TEST_MP3 = os.path.join(_REPO_ROOT, 'input_test.mp3')

class TestFeed(unittest.TestCase):

    @patch('fat_llama_fftw.audio_fattener.feed.AudioSegment.from_file')
    @patch('fat_llama_fftw.audio_fattener.feed.MP3')
    @patch('os.path.exists', return_value=True)
    def test_read_audio(self, mock_exists, mock_mp3, mock_from_file):
        mock_audio = MagicMock()
        mock_audio.frame_rate = 44100
        mock_audio.channels = 2
        mock_audio.get_array_of_samples.return_value = np.arange(
            44100 * 4, dtype=np.int16)
        mock_from_file.return_value = mock_audio
        mock_mp3.return_value.info.bitrate = 1411000

        sample_rate, samples, bitrate, audio = read_audio('test.mp3', 'mp3')
        self.assertEqual(sample_rate, 44100)
        self.assertEqual(samples.shape, (44100 * 4 // 2, 2))
        # The bitrate must be the one reported by the MP3 tag reader, not None.
        self.assertEqual(bitrate, 1411000)
        # The returned AudioSegment must be the decoded object itself.
        self.assertIs(audio, mock_audio)
        # Stereo de-interleaving must preserve sample content and ordering:
        # the flat array is [L0, R0, L1, R1, ...].
        expected = np.arange(44100 * 4, dtype=np.int16).reshape((-1, 2))
        np.testing.assert_array_equal(samples, expected)
        # Duration implied by the returned samples must match the source.
        self.assertAlmostEqual(samples.shape[0] / sample_rate, 2.0, places=6)

    @patch('fat_llama_fftw.audio_fattener.feed.sf.write')
    def test_write_audio(self, mock_write):
        # float64 in - the writer's contract is to hand soundfile float32
        # samples, so the downcast must be asserted, not assumed.
        data = np.random.rand(44100 * 10).astype(np.float64)
        write_audio('output.flac', 44100, data, 'flac')
        mock_write.assert_called_once()
        args, kwargs = mock_write.call_args
        self.assertEqual(args[1].dtype, np.float32)
        np.testing.assert_array_equal(args[1], data.astype(np.float32))
        self.assertEqual(args[0], 'output.flac')
        self.assertEqual(args[2], 44100)
        self.assertEqual(kwargs['format'], 'FLAC')
        self.assertEqual(kwargs['subtype'], 'PCM_24')

    @patch('fat_llama_fftw.audio_fattener.feed.sf.write')
    def test_write_audio_wav_uses_wav_container(self, mock_write):
        data = np.random.rand(1000).astype(np.float64)
        write_audio('output.wav', 48000, data, 'wav')
        args, kwargs = mock_write.call_args
        self.assertEqual(args[0], 'output.wav')
        self.assertEqual(args[2], 48000)
        self.assertEqual(kwargs['format'], 'WAV')
        self.assertEqual(kwargs['subtype'], 'PCM_24')
        self.assertEqual(args[1].dtype, np.float32)

    def test_write_audio_rejects_unsupported_format(self):
        with self.assertRaises(ValueError):
            write_audio('output.mp3', 44100, np.zeros(10, dtype=np.float32),
                        'mp3')

    def test_new_interpolation_algorithm(self):
        data = np.array([1, 2, 3, 4])
        upscale_factor = 2
        expected_output = np.array([1, 1, 2, 2, 3, 3, 4, 4], dtype=np.float32)
        output = new_interpolation_algorithm(data, upscale_factor)
        np.testing.assert_array_equal(output, expected_output)
        self.assertEqual(output.dtype, np.float32)
        self.assertEqual(len(output), len(data) * upscale_factor)

    def test_initialize_ist(self):
        # threshold is a fraction of the array's own peak magnitude (cycle-2
        # fix), not an absolute cutoff - a raw threshold like the documented
        # default 0.6 is meaningless compared directly against int16-scale
        # audio samples (peaks in the tens of thousands), so it must scale
        # with the data itself. Peak here is 4, so threshold=0.6 keeps only
        # values > 0.6*4 = 2.4.
        data = np.array([1, 2, 3, 4])
        threshold = 0.6
        expected_output = np.array([0, 0, 3, 4])
        output = initialize_ist(data, threshold)
        np.testing.assert_array_equal(output, expected_output)

    def test_initialize_ist_scales_with_data_magnitude(self):
        # The same fractional threshold must keep the same *proportion* of
        # samples regardless of the data's absolute numeric scale - this is
        # the regression test for the bug where a fixed absolute threshold
        # kept ~100% of int16-scale samples (root cause probed in cycle 2:
        # threshold_value=0.6 against raw int16-scale data kept 99.35% of
        # samples, making IST a near-identity operation with no added
        # detail).
        small_scale = np.array([1, 2, 3, 4], dtype=np.float64)
        int16_scale = small_scale * 8192.0  # e.g. [8192, 16384, 24576, 32768]
        threshold = 0.6

        small_mask = initialize_ist(small_scale, threshold) != 0
        large_mask = initialize_ist(int16_scale, threshold) != 0
        np.testing.assert_array_equal(small_mask, large_mask)
        # And it isn't simply "keep everything" at int16 scale, which was
        # the actual bug (a fixed absolute 0.6 threshold against values in
        # the thousands never zeroes anything).
        self.assertTrue(np.any(large_mask == False))  # noqa: E712

    def test_initialize_ist_zero_signal_stays_zero(self):
        # Guards the peak==0 short-circuit added for the relative-threshold
        # fix - without it, dividing/comparing against a zero peak would be
        # a degenerate no-op mask rather than an explicit all-zero return.
        data = np.zeros(5)
        output = initialize_ist(data, 0.6)
        np.testing.assert_array_equal(output, np.zeros(5))

    def test_upscale_channels(self):
        # threshold must be a *fraction* of peak magnitude (cycle-2
        # semantics). This test previously passed threshold=2.5, which
        # under fractional semantics means 250% of peak - nothing survives
        # thresholding, IST contributes exactly zero, and the test silently
        # degenerated into a duplicate of
        # test_upscale_channels_thresholded_out_is_pure_interpolation while
        # its only remaining assertion (a loose "within 4x the source peak"
        # bound) passed trivially. Use an in-range threshold so the
        # IST-contributing path is actually exercised, and assert the exact
        # expected samples rather than a bound.
        #
        # Cycle-3 fix: perform_ist_iteration now always excludes the FFT's
        # DC bin (index 0) from surviving thresholding (see its own
        # docstring comment - a cycle-2 regression let DC dominate the
        # kept mask for asymmetric/transient content, injecting a literal
        # DC offset into ist_changes). For this array
        # (interpolated to [1,1,3,3] per channel), initialize_ist keeps
        # only the loud half [0,0,3,3], whose FFT is [6, -3+3i, 0, -3-3i];
        # excluding bin 0 leaves only the conjugate pair (bins 1 and 3),
        # whose ifft is the new fixed point [-1.5,-1.5,1.5,1.5] - a
        # different, DC-free result than the old (DC-inclusive) fixed
        # point, with the same 1.5x-of-source-peak scale-invariant relationship
        # for both channels since peak-relative thresholding scales with
        # the input.
        channels = np.array([[1, 2], [3, 4]], dtype=np.float32)
        upscale_factor = 2
        threshold = 0.6
        max_iter = 10
        output = upscale_channels(channels, upscale_factor, max_iter,
                                  threshold)
        self.assertEqual(output.shape, (4, 2))
        # Output must be usable audio, not NaN/Inf.
        self.assertTrue(np.all(np.isfinite(output)))
        expected = np.array([[-0.5, 0], [-0.5, 0], [4.5, 6], [4.5, 6]],
                            dtype=np.float32)
        np.testing.assert_allclose(output, expected, atol=1e-6)
        # IST must have contributed something here - otherwise this test
        # would be indistinguishable from pure interpolation.
        interpolated = np.array([[1, 2], [1, 2], [3, 4], [3, 4]],
                                dtype=np.float32)
        self.assertGreater(float(np.max(np.abs(output - interpolated))), 0.0)
        # Column order must be preserved: column i still derives from
        # channel i, so each column stays proportional to its own source
        # channel and never picks up the other channel's values. The
        # DC-free fixed point's peak lands at 1.5x each channel's own
        # source peak (scale-invariant, since peak-relative thresholding
        # produces a proportional result regardless of the input's
        # absolute scale).
        for i, src in enumerate(channels.T):
            self.assertAlmostEqual(float(np.max(np.abs(output[:, i]))),
                                   1.5 * float(np.max(np.abs(src))),
                                   places=5)

    def test_upscale_channels_thresholded_out_is_pure_interpolation(self):
        # threshold is a fraction of peak magnitude, so any value > 1.0
        # sits above every sample: IST contributes exactly zero and the
        # result must be the zero-order-hold interpolation of the input.
        channels = np.array([[1, 2], [3, 4]], dtype=np.float32)
        output = upscale_channels(channels, upscale_factor=2, max_iter=5,
                                  threshold=1.5)
        expected = np.array([[1, 2], [1, 2], [3, 4], [3, 4]], dtype=np.float32)
        np.testing.assert_allclose(output, expected, atol=1e-6)

    def test_iterative_soft_thresholding_chains_across_iterations(self):
        data = np.array([0.9, 0.1, -0.8, 0.05, 0.7, -0.2, 0.3, -0.6],
                        dtype=np.float32)
        threshold = 0.05

        # Ground truth for what "N chained IST iterations" must mean:
        # each pass refines the previous pass's own output. Disable the
        # early-convergence exit here (convergence_tol=0) so this proves
        # the chaining mechanism itself runs the full N passes, not that
        # it happens to stop early.
        expected = initialize_ist(data, threshold)
        for _ in range(3):
            expected = perform_ist_iteration(expected, threshold)

        actual = iterative_soft_thresholding(data, max_iter=3,
                                             threshold=threshold,
                                             convergence_tol=0.0)
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_iterative_soft_thresholding_stops_early_once_converged(self):
        # Hard-threshold IST is a fixed-point projection (fft/ifft are
        # exact inverses of each other), so it provably stabilizes after
        # the first pass, up to float32 rounding. Running the caller's
        # full max_iter passes regardless would burn compute on iterations
        # that cannot change the result - the loop must detect that and
        # stop early rather than always running to max_iter.
        np.random.seed(1)
        n = 2000
        t = np.arange(n) / 44100.0
        sig = (0.5 * np.sin(2 * np.pi * 440 * t)
              + 0.1 * np.random.randn(n)).astype(np.float32)

        import fat_llama_fftw.audio_fattener.feed as feed_module
        with patch.object(feed_module, 'perform_ist_iteration',
                          wraps=feed_module.perform_ist_iteration) as spy:
            early = feed_module.iterative_soft_thresholding(
                sig, max_iter=300, threshold=0.05)
            call_count = spy.call_count

        self.assertLess(call_count, 300)

        # The early-stopped result must match running the full max_iter
        # passes (convergence_tol=0 disables early stop) - proving the
        # early exit only skips passes that were already no-ops, not that
        # it changes the algorithm's output.
        full = feed_module.iterative_soft_thresholding(
            sig, max_iter=300, threshold=0.05, convergence_tol=0.0)
        np.testing.assert_allclose(early, full, atol=1e-4)

    def test_ist_adds_content_distinct_from_input_at_realistic_scale(self):
        # Regression test for the cycle-2 "no measurable added detail"
        # finding: at int16-scale amplitudes (real decoded audio samples
        # run up to ~32767), the pre-fix absolute threshold_value=0.6 kept
        # ~99.8% of time-domain samples and ~99.9998% of FFT bins, making
        # perform_ist_iteration an effectively lossless fft/ifft identity -
        # ist_changes was almost a duplicate of the input, contributing no
        # genuine spectral difference (upscale_channels' `expanded +
        # ist_changes` was ~= 2*expanded, which normalize_signal then
        # divided straight back out). With peak-relative thresholding, the
        # same default threshold_value=0.6 must produce a result that
        # differs substantially from the input, not a near-copy of it.
        n = 2000
        t = np.arange(n) / 44100.0
        sig = (12000 * np.sin(2 * np.pi * 440 * t)
              + 6000 * np.sin(2 * np.pi * 2000 * t)).astype(np.float32)

        ist_changes = iterative_soft_thresholding(sig, max_iter=50,
                                                   threshold=0.6)

        self.assertTrue(np.all(np.isfinite(ist_changes)))
        # IST must contribute something (not an all-zero no-op)...
        self.assertGreater(np.max(np.abs(ist_changes)), 0)
        # ...and must not simply reproduce the input almost exactly - the
        # pre-fix bug's signature was ist_changes landing within ~1.95e-3
        # of the input's own peak.
        max_diff = np.max(np.abs(ist_changes - sig))
        self.assertGreater(max_diff, 0.05 * np.max(np.abs(sig)))

    def test_perform_ist_iteration_never_keeps_dc_bin(self):
        # Regression test for the cycle-2 DC-offset regression found by
        # audio-quality-checker in cycle 3 (+41.7 dB DC offset vs. the
        # reference on a real 15s output file). Root cause: asymmetric,
        # transient-heavy content (e.g. a kick-drum-like spike whose
        # positive excursion outweighs its negative one) concentrates a
        # large share of broadband energy at FFT bin 0 (DC) - sometimes
        # enough to make it the single loudest bin, which the old
        # peak-relative threshold then kept outright. perform_ist_iteration
        # must now always zero the DC bin regardless of its magnitude.
        rng = np.random.default_rng(3)
        n = 4000
        sig = (2000 * rng.standard_normal(n)).astype(np.float64)
        spike_idx = rng.choice(n, size=3, replace=False)
        sig[spike_idx] += rng.uniform(15000, 30000, size=3)

        data_thres = initialize_ist(sig, 0.6)
        # Confirm this input actually reproduces the bug's precondition -
        # the DC bin must be at (or very near) the loudest bin in the raw,
        # un-fixed FFT, otherwise this isn't exercising the regression.
        raw_fft = np.fft.fft(data_thres)
        self.assertAlmostEqual(
            float(np.abs(raw_fft[0])), float(np.max(np.abs(raw_fft))),
            delta=1e-6 * float(np.max(np.abs(raw_fft)))
        )

        result = perform_ist_iteration(data_thres, 0.6)
        result_fft = np.fft.fft(result)
        # DC must be exactly excluded, not just reduced.
        self.assertAlmostEqual(float(np.abs(result_fft[0])), 0.0, places=6)
        # And the pass must still have kept other content (not degenerated
        # into an all-zero no-op by removing the one bin that dominated).
        self.assertGreater(np.max(np.abs(result)), 0)

    def test_iterative_soft_thresholding_output_has_no_dc_offset(self):
        # End-to-end version of the DC regression, through the public
        # iterative_soft_thresholding entry point (long enough - above
        # the default block_size - to also exercise the blocked/windowed
        # path, since each block's own DC bin must stay excluded too).
        rng = np.random.default_rng(3)
        n = 40000
        sig = (2000 * rng.standard_normal(n)).astype(np.float64)
        spike_idx = rng.choice(n, size=6, replace=False)
        sig[spike_idx] += rng.uniform(15000, 30000, size=6)

        ist_changes = iterative_soft_thresholding(sig, max_iter=50,
                                                   threshold=0.6)
        self.assertTrue(np.all(np.isfinite(ist_changes)))
        peak = np.max(np.abs(ist_changes))
        self.assertGreater(peak, 0)
        # The old (pre-fix) whole-file behavior put the DC bin at the very
        # top of the kept mask (see test_perform_ist_iteration_never_keeps_
        # dc_bin) - the fixed output's own mean, relative to its peak
        # amplitude, must be far below the cycle-2 regression's measured
        # +1.9897e-3 (-54.0 dBFS) offset.
        relative_dc = abs(float(np.mean(ist_changes))) / peak
        self.assertLess(relative_dc, 1e-3)
        # Note: this white-noise-plus-spikes fixture does NOT exercise the
        # cycle-4 WOLA synthesis-window DC leak fixed in cycle 5 (see
        # test_ist_no_dc_offset_on_real_programme_material below) -
        # measured directly, this fixture's relative_dc
        # stays under ~4e-5 with or without that fix, because the leak is
        # a block-to-block *programme-correlated* effect (an infrasonic
        # drift built from many blocks' own windowed-DC leakage summing
        # coherently) that white noise's block-to-block independence does
        # not produce. A real/tonal fixture is required to catch it.

    def test_ist_no_dc_offset_on_real_programme_material(self):
        # Regression test for the cycle-5 finding: cycle 4's WOLA fix
        # (synthesis windowing to eliminate the block-hop-rate click)
        # itself reintroduced a DC leak. perform_ist_iteration always
        # excludes the FFT DC bin, so each block's raw frame_result has an
        # (numerically) exact-zero mean - but multiplying by the
        # synthesis window is a frequency-domain convolution with the
        # window's own spectrum (not a perfect delta at 0 Hz), which
        # leaks frame_result's own near-DC content into the windowed
        # frame's DC bin. On real programme material this measured as a
        # +40 dB, programme-correlated (infrasonic-drift-shaped) DC
        # regression - a whole order of magnitude above the 1e-3 bound
        # test_iterative_soft_thresholding_output_has_no_dc_offset's own
        # (non-representative) white-noise fixture asserts.
        #
        # This uses the repo's own committed reference asset
        # (input_test.mp3, already relied on by example.py/README) rather
        # than a synthetic fixture, because a synthetic attempt at
        # reproducing this (tremolo-enveloped tones, transients, etc.)
        # could not reliably reproduce the reported magnitude - the leak
        # is a many-block, real-programme-correlated effect. Measured
        # directly against this exact fixture/parameters before the
        # cycle-5 fix: relative_dc = 3.92e-3; after: 7.3e-8.
        if not os.path.exists(_INPUT_TEST_MP3):
            self.skipTest("input_test.mp3 reference asset not present")

        sample_rate, samples, bitrate, audio = read_audio(_INPUT_TEST_MP3,
                                                           format='mp3')
        channel = samples[:, 0].astype(np.float64)
        expanded = new_interpolation_algorithm(channel, upscale_factor=4)

        ist_changes = iterative_soft_thresholding(expanded, max_iter=50,
                                                   threshold=0.6)
        self.assertTrue(np.all(np.isfinite(ist_changes)))
        peak = np.max(np.abs(ist_changes))
        self.assertGreater(peak, 0)
        relative_dc = abs(float(np.mean(ist_changes))) / peak
        self.assertLess(relative_dc, 1e-4)

    def test_iterative_soft_thresholding_blocks_restore_multiple_bands(self):
        # Regression test for the cycle-2/cycle-3 "no measurable added
        # detail" finding: a single whole-file peak-relative threshold is
        # dominated by whichever moment/frequency is loudest across the
        # entire signal, so quieter bands never clear it. This signal has
        # one loud low-frequency segment and three much quieter
        # higher-frequency segments (realistic dynamic range, ~20 dB
        # down) spanning multiple block_size-sized blocks; with localized
        # (blocked) thresholding each segment's own local peak should let
        # its own dominant frequency survive, not just the loudest one.
        n = 40000
        t = np.arange(n) / 44100.0
        seg = n // 4
        sig = np.zeros(n, dtype=np.float64)
        freqs = [300, 1500, 5000, 9000]
        amps = [30000, 3000, 2800, 2500]
        for i, (f, a) in enumerate(zip(freqs, amps)):
            s = i * seg
            e = (i + 1) * seg if i < 3 else n
            sig[s:e] = a * np.sin(2 * np.pi * f * t[s:e])

        ist_changes = iterative_soft_thresholding(sig, max_iter=50,
                                                   threshold=0.6)
        self.assertTrue(np.all(np.isfinite(ist_changes)))
        # The blocked/overlap-add path pads by a half-block on the left and
        # a full block on the right before framing, then slices back out -
        # an off-by-a-hop error there would silently time-shift or truncate
        # ist_changes relative to the channel upscale_channels adds it to.
        # upscale_channels does `expanded + ist_changes` with no length
        # reconciliation, so this must be exact, not approximate.
        self.assertEqual(len(ist_changes), n)

        spectrum = np.abs(np.fft.rfft(ist_changes.astype(np.float64)))
        freq_axis = np.fft.rfftfreq(n, d=1.0 / 44100.0)
        peak = np.max(spectrum)
        magnitudes = []
        for f in freqs:
            idx = int(np.argmin(np.abs(freq_axis - f)))
            magnitudes.append(spectrum[idx])

        # Every one of the four bands - not just the loudest segment's -
        # must show measurable restored energy (at least 1% of the
        # overall peak). The pre-fix whole-file threshold left the three
        # quieter bands at ~0 (verified directly against the current
        # source in this same scenario: 1500/5000/9000 Hz all landed
        # below 1e-3 magnitude against a peak in the hundreds of millions).
        for f, mag in zip(freqs, magnitudes):
            self.assertGreater(mag, 0.01 * peak,
                              f"{f} Hz band shows no measurable added detail")

    def test_iterative_soft_thresholding_no_block_boundary_discontinuity(self):
        # Regression test for a cycle-4 finding: the blocked/overlap-add
        # path (cycle 3) windowed only the *analysis* side (frame =
        # padded[...] * window) then overlap-added frame_result
        # un-windowed. COLA (adjacent windows summing to a flat constant)
        # is only a valid reconstruction argument for a *linear* per-block
        # operation - windowing the input and trusting the operation to
        # preserve that taper on the way out. _ist_chain is a nonlinear
        # hard-threshold projection in the FFT domain: verified directly
        # (a single stationary tone framed through one block), an
        # analysis-windowed frame's edges taper to ~1e-4 of its own peak,
        # but perform_ist_iteration's output has edges back up at ~8-11%
        # of that frame's own peak - thresholding+ifft does not preserve
        # the input's time-domain taper. Summing that un-tapered edge
        # content in at full weight at every hop boundary produced a
        # sample-scale discontinuity ("click") audible as a broadband
        # impulse train locked to the block hop rate (measured by
        # audio-quality-checker as a +15.4 dB rise in the 16-22.05kHz band
        # of a quiet passage, ~76 Hz burst rate matching the hop exactly).
        #
        # A stationary, single-tone signal spanning multiple blocks has no
        # legitimate reason to contain a sample-scale discontinuity
        # anywhere in the IST output. The discrete second difference
        # (local curvature) is a sensitive, deterministic detector for
        # exactly that: on this fixture, the pre-fix code produces one
        # block-boundary sample whose |d2| is ~67x the signal's own
        # typical (median) curvature - a genuine click, not sampling
        # noise - while a WOLA-correct reconstruction stays within a
        # small multiple of its own typical local curvature throughout.
        block_size = 8192
        sample_rate = 44100
        n = block_size * 8
        t = np.arange(n) / sample_rate
        data = (2000 * np.sin(2 * np.pi * 500.0 * t)).astype(np.float64)

        ist_changes = iterative_soft_thresholding(
            data, max_iter=50, threshold=0.6, block_size=block_size
        ).astype(np.float64)

        d2 = np.diff(ist_changes, n=2)
        median_d2 = np.median(np.abs(d2))
        max_d2 = np.max(np.abs(d2))
        self.assertLess(max_d2, 10 * median_d2)

    def test_apply_nyquist_cutoff_removes_image_content(self):
        original_sample_rate = 8000
        upscale_factor = 4
        new_sample_rate = original_sample_rate * upscale_factor
        n = original_sample_rate  # 1 second at the original rate
        t = np.arange(n) / original_sample_rate
        f0 = 1000.0  # well below the original Nyquist (4000 Hz)
        tone = np.sin(2 * np.pi * f0 * t).astype(np.float32)

        # Zero-order-hold interpolation images the tone above the
        # original Nyquist (e.g. near new_sample_rate - f0).
        imaged = new_interpolation_algorithm(tone, upscale_factor)
        original_nyquist = original_sample_rate / 2.0
        freqs = np.fft.rfftfreq(len(imaged), d=1.0 / new_sample_rate)

        spectrum_before = np.abs(np.fft.rfft(imaged))
        energy_above_before = np.sum(spectrum_before[freqs > original_nyquist])
        self.assertGreater(energy_above_before, 0)

        filtered = apply_nyquist_cutoff(imaged, new_sample_rate,
                                        original_nyquist)
        spectrum_after = np.abs(np.fft.rfft(filtered))
        energy_above_after = np.sum(spectrum_after[freqs > original_nyquist])
        energy_below_after = np.sum(spectrum_after[freqs <= original_nyquist])

        # Content above the original Nyquist must be effectively removed
        # (observed reduction is ~4.35e-6 relative, i.e. ~-53.6 dB, which
        # is float32 rounding noise from the brick-wall zeroing - not
        # exactly 0, but well below any audible/measurable threshold).
        self.assertLess(energy_above_after, energy_above_before * 1e-4)
        # ...while the in-band tone content survives.
        self.assertGreater(energy_below_after, 0)

    @patch.object(feed_module, 'write_audio')
    @patch.object(feed_module, 'read_audio')
    def test_upscale_wires_apply_nyquist_cutoff(self, mock_read_audio,
                                                mock_write_audio):
        # Regression test for the cycle-1 finding that no test asserted
        # upscale() actually calls apply_nyquist_cutoff - the wiring was
        # only covered in isolation via
        # test_apply_nyquist_cutoff_removes_image_content, which never
        # calls upscale() itself. Mocks I/O so this runs end-to-end on a
        # tiny synthetic signal without touching the filesystem.
        sample_rate = 8000
        n_frames = 20
        mock_audio = MagicMock()
        mock_audio.channels = 2
        rng = np.random.default_rng(7)
        flat_samples = (rng.integers(-3000, 3000, size=n_frames * 2)
                       .astype(np.int16))
        mock_audio.get_array_of_samples.return_value = flat_samples
        mock_read_audio.return_value = (sample_rate, None, None, mock_audio)

        original_nyquist = sample_rate / 2.0

        with patch.object(feed_module, 'apply_nyquist_cutoff',
                          wraps=feed_module.apply_nyquist_cutoff) as spy:
            upscale(
                input_file_path='in.mp3',
                output_file_path='out.flac',
                source_format='mp3',
                target_format='flac',
                max_iterations=5,
                threshold_value=0.6,
                target_bitrate_kbps=1000
            )

        # apply_nyquist_cutoff must be called once per channel (stereo).
        self.assertEqual(spy.call_count, 2)
        for call in spy.call_args_list:
            args = call.args
            self.assertEqual(args[1], sample_rate * 4)  # upscale_factor=4
                                                          # (no bitrate given)
            self.assertEqual(args[2], original_nyquist)

        # And the filtered result is what actually gets written out.
        mock_write_audio.assert_called_once()
        write_args, write_kwargs = mock_write_audio.call_args
        written_data = write_args[2]
        self.assertEqual(written_data.shape[1], 2)
        self.assertTrue(np.all(np.isfinite(written_data)))

        # Asserting only that apply_nyquist_cutoff was *called* checks the
        # wiring, not the outcome - a later stage reordering (e.g. moving
        # normalization back after the cutoff) would keep the call and its
        # arguments identical while still shipping above-Nyquist content.
        # project-mission.md's hard constraint is about what lands in the
        # file, so assert that on the data actually handed to write_audio.
        new_sample_rate = write_args[1]
        self.assertEqual(new_sample_rate, sample_rate * 4)
        for ch in range(written_data.shape[1]):
            spectrum = np.abs(np.fft.rfft(written_data[:, ch]))
            freqs = np.fft.rfftfreq(written_data.shape[0],
                                    d=1.0 / new_sample_rate)
            above = freqs > original_nyquist
            energy_above = float(np.sum(spectrum[above] ** 2))
            energy_below = float(np.sum(spectrum[~above] ** 2))
            self.assertGreater(energy_below, 0)
            self.assertLess(energy_above, energy_below * 1e-6)

    @patch.object(feed_module, 'write_audio')
    def test_upscale_output_never_exceeds_full_scale(self, mock_write_audio):
        # Regression test for the cycle-5 clipping finding: apply_nyquist_
        # cutoff (a brick-wall FFT lowpass) ran *after* the final
        # normalize_signal() call, so any Gibbs-phenomenon filter
        # overshoot on an already-peak-normalized-to-1.0 signal could push
        # samples back above full scale - write_audio's PCM_24 subtype
        # then silently clips them (measured: a handful of samples pinned
        # at exactly +/-1.0 in a real output file). Uses the real
        # input_test.mp3 reference asset (only write_audio is mocked,
        # avoiding disk I/O for the output) because this overshoot did not
        # reproduce on small/short synthetic inputs - it needs enough
        # real-programme content and block variety for the filter's Gibbs
        # ripple to exceed the exact-1.0 peak, measured directly against
        # this same fixture: pre-fix peak 1.0000354 (9 samples clipped),
        # post-fix exactly 1.0 (0 samples clipped).
        if not os.path.exists(_INPUT_TEST_MP3):
            self.skipTest("input_test.mp3 reference asset not present")

        upscale(
            input_file_path=_INPUT_TEST_MP3,
            output_file_path='out.flac',
            source_format='mp3',
            target_format='flac',
            max_iterations=50,
            threshold_value=0.6,
            target_bitrate_kbps=1411
        )

        mock_write_audio.assert_called_once()
        write_args, write_kwargs = mock_write_audio.call_args
        written_data = write_args[2]
        self.assertTrue(np.all(np.isfinite(written_data)))
        # Full scale, not "close to it, unless the filter pushed it over" -
        # the final normalization must be the true last numeric step, so
        # this must hold exactly (up to float32 rounding), not merely
        # approximately.
        self.assertLessEqual(float(np.max(np.abs(written_data))),
                             1.0 + 1e-6)
        self.assertEqual(int(np.sum(np.abs(written_data) > 1.0)), 0)

    def test_normalize_signal(self):
        signal = np.array([1, 2, 3, 4], dtype=np.float32)
        expected_output = signal / 4
        output = normalize_signal(signal)
        np.testing.assert_array_equal(output, expected_output)
        # Normalization must peak at exactly full scale and stay in [-1, 1].
        self.assertAlmostEqual(float(np.max(np.abs(output))), 1.0, places=6)
        self.assertTrue(np.all(np.abs(output) <= 1.0))

if __name__ == '__main__':
    unittest.main()
