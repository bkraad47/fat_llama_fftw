import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from fat_llama_fftw.audio_fattener.feed import (
    read_audio,
    write_audio,
    new_interpolation_algorithm,
    initialize_ist,
    upscale_channels,
    normalize_signal
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

    def test_normalize_signal(self):
        signal = np.array([1, 2, 3, 4], dtype=np.float32)
        expected_output = signal / 4
        output = normalize_signal(signal)
        np.testing.assert_array_equal(output, expected_output)

if __name__ == '__main__':
    unittest.main()
