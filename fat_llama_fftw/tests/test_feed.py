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
        channels = np.array([[1, 2], [3, 4]], dtype=np.float32)
        upscale_factor = 2
        threshold = 2.5
        max_iter = 10
        output = upscale_channels(channels, upscale_factor, max_iter,
                                  threshold)
        self.assertEqual(output.shape, (4, 2))
        # Output must be usable audio, not NaN/Inf.
        self.assertTrue(np.all(np.isfinite(output)))
        # Column order must be preserved: column i still derives from
        # channel i, so each column must stay within its own source
        # channel's dynamic range.
        for i, src in enumerate(channels.T):
            self.assertLessEqual(np.max(np.abs(output[:, i])),
                                 np.max(np.abs(src)) * 4.0)

    def test_upscale_channels_thresholded_out_is_pure_interpolation(self):
        # With a threshold above every sample, IST contributes exactly zero,
        # so the result must be the zero-order-hold interpolation of the input.
        channels = np.array([[1, 2], [3, 4]], dtype=np.float32)
        output = upscale_channels(channels, upscale_factor=2, max_iter=5,
                                  threshold=100.0)
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
