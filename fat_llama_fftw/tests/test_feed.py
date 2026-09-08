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
    apply_nyquist_cutoff
)

class TestFeed(unittest.TestCase):

    @patch('fat_llama_fftw.audio_fattener.feed.AudioSegment.from_file')
    @patch('fat_llama_fftw.audio_fattener.feed.MP3')
    @patch('os.path.exists', return_value=True)
    def test_read_audio(self, mock_exists, mock_mp3, mock_from_file):
        mock_audio = MagicMock()
        mock_audio.frame_rate = 44100
        mock_audio.channels = 2
        mock_audio.get_array_of_samples.return_value = np.arange(44100 * 4, dtype=np.int16)
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
        data = np.random.rand(44100 * 10).astype(np.float32)
        write_audio('output.flac', 44100, data, 'flac')
        mock_write.assert_called_once()
        args, kwargs = mock_write.call_args
        np.testing.assert_array_equal(args[1], data)
        self.assertEqual(args[0], 'output.flac')
        self.assertEqual(args[2], 44100)
        self.assertEqual(kwargs['format'], 'FLAC')
        self.assertEqual(kwargs['subtype'], 'PCM_24')

    def test_new_interpolation_algorithm(self):
        data = np.array([1, 2, 3, 4])
        upscale_factor = 2
        expected_output = np.array([1, 1, 2, 2, 3, 3, 4, 4], dtype=np.float32)
        output = new_interpolation_algorithm(data, upscale_factor)
        np.testing.assert_array_equal(output, expected_output)
        self.assertEqual(output.dtype, np.float32)
        self.assertEqual(len(output), len(data) * upscale_factor)

    def test_initialize_ist(self):
        data = np.array([1, 2, 3, 4])
        threshold = 2.5
        expected_output = np.array([0, 0, 3, 4])
        output = initialize_ist(data, threshold)
        np.testing.assert_array_equal(output, expected_output)

    def test_upscale_channels(self):
        channels = np.array([[1, 2], [3, 4]], dtype=np.float32)
        upscale_factor = 2
        threshold = 2.5
        max_iter = 10
        output = upscale_channels(channels, upscale_factor, max_iter, threshold)
        self.assertEqual(output.shape, (4, 2))
        # Output must be usable audio, not NaN/Inf.
        self.assertTrue(np.all(np.isfinite(output)))
        # Column order must be preserved: column i still derives from channel i,
        # so each column must stay within its own source channel's dynamic range.
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
