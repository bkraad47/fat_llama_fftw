from fat_llama_fftw.audio_fattener.feed import upscale

# Example call to the method.
# As of v2.0.0, the IST peak-inflation cap (an internal stage of upscale(),
# not a parameter you set here) is frequency-selective and envelope-gated:
# it now preserves quiet/high-frequency detail IST adds instead of shrinking
# it along with the dominant band, without reopening the earlier peak-
# inflation attenuation regression. No new upscale() parameters were added -
# the call below is unchanged.
upscale(
    input_file_path='input_test.mp3',
    output_file_path='output_test.flac',
    source_format='mp3',
    target_format='flac',
    max_iterations=600,
    threshold_value=0.75,
    target_bitrate_kbps=1400
)
