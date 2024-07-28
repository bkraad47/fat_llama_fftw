import numpy as np
import pyfftw
import concurrent.futures
from pydub import AudioSegment
import soundfile as sf
import os
import logging
from mutagen.mp3 import MP3
from mutagen.flac import FLAC
from mutagen.oggvorbis import OggVorbis
from mutagen.wave import WAVE

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_audio(file_path, format):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    
    audio = AudioSegment.from_file(file_path, format=format)
    samples = np.array(audio.get_array_of_samples())
    sample_rate = audio.frame_rate
    bitrate = None
    
    if format == 'mp3':
        mp3_info = MP3(file_path)
        bitrate = mp3_info.info.bitrate
    elif format == 'flac':
        flac_info = FLAC(file_path)
        bitrate = flac_info.info.bitrate
    elif format == 'ogg':
        ogg_info = OggVorbis(file_path)
        bitrate = ogg_info.info.bitrate
    elif format == 'wav':
        wav_info = WAVE(file_path)
        bitrate = wav_info.info.bitrate
    else:
        duration_seconds = len(audio) / 1000.0
        bitrate = (len(samples) * 8) / duration_seconds
    
    if audio.channels == 2:
        samples = samples.reshape((-1, 2))
    
    return sample_rate, samples, bitrate, audio

def write_audio(file_path, sample_rate, data, format):
    if format == 'flac':
        sf.write(file_path, data.astype(np.float32), sample_rate, format='FLAC', subtype='PCM_24')
    elif format == 'wav':
        sf.write(file_path, data.astype(np.float32), sample_rate, format='WAV', subtype='PCM_24')
    else:
        raise ValueError(f"Unsupported target format: {format}")

def new_interpolation_algorithm(data, upscale_factor):
    original_length = len(data)
    expanded_length = original_length * upscale_factor
    expanded_data = np.zeros(expanded_length, dtype=np.float32)
    
    for i in range(original_length):
        center_point = data[i]
        for j in range(upscale_factor):
            index = i * upscale_factor + j
            expanded_data[index] = center_point
    
    return expanded_data

def initialize_ist(data, threshold):
    mask = np.abs(data) > threshold
    data_thres = np.where(mask, data, 0)
    return data_thres

def perform_ist_iteration(data_thres, threshold):
    data_fft = pyfftw.interfaces.numpy_fft.fft(data_thres)
    mask = np.abs(data_fft) > threshold
    data_fft_thres = np.where(mask, data_fft, 0)
    data_thres = pyfftw.interfaces.numpy_fft.ifft(data_fft_thres).real
    return data_thres

def iterative_soft_thresholding(data, max_iter, threshold):
    data_thres = initialize_ist(data, threshold)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(perform_ist_iteration, data_thres, threshold) for _ in range(max_iter)}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            data_thres = future.result()
    
    return data_thres

def upscale_channels(channels, upscale_factor, max_iter, threshold):
    processed_channels = []
    for channel in channels.T:
        logger.info("Interpolating data...")
        expanded_channel = new_interpolation_algorithm(channel, upscale_factor)

        logger.info("Performing IST...")
        ist_changes = iterative_soft_thresholding(expanded_channel, max_iter, threshold)
        expanded_channel = expanded_channel.astype(np.float32) + ist_changes

        processed_channels.append(expanded_channel)
    
    return np.column_stack(processed_channels)

def normalize_signal(signal):
    return signal / np.max(np.abs(signal))

def upscale(
        input_file_path,
        output_file_path,
        source_format,
        target_format='flac',
        max_iterations=800,
        threshold_value=0.6,
        target_bitrate_kbps=1411
    ):
    valid_bitrate_ranges = {
        'flac': (800, 1411),
        'wav': (800, 6444),
    }
    
    if target_format not in valid_bitrate_ranges:
        raise ValueError(f"Unsupported target format: {target_format}")
    
    min_bitrate, max_bitrate = valid_bitrate_ranges[target_format]
    
    if not (min_bitrate <= target_bitrate_kbps <= max_bitrate):
        raise ValueError(f"{target_format.upper()} bitrate out of range. Please provide a value between {min_bitrate} and {max_bitrate} kbps.")
    
    logger.info(f"Loading {source_format.upper()} file...")
    sample_rate, samples, bitrate, audio = read_audio(input_file_path, format=source_format)
    if bitrate:
        logger.info(f"Original {source_format.upper()} bitrate: {bitrate / 1000:.2f} kbps")
    
    samples = np.array(audio.get_array_of_samples())
    if audio.channels == 2:
        samples = samples.reshape((-1, 2))
    
    target_bitrate = target_bitrate_kbps * 1000
    upscale_factor = round(target_bitrate / bitrate) if bitrate else 4
    logger.info(f"Upscale factor set to: {upscale_factor}")

    if samples.ndim == 1:
        logger.info("Mono channel detected.")
        channels = samples[:, np.newaxis]
    else:
        logger.info("Stereo channels detected.")
        channels = samples

    logger.info("Upscaling and processing channels...")
    upscaled_channels = upscale_channels(
        channels,
        upscale_factor=upscale_factor,
        max_iter=max_iterations,
        threshold=threshold_value
    )
    
    logger.info("Auto-scaling amplitudes based on original audio...")
    scaled_upscaled_channels = []
    for i, channel in enumerate(channels.T):
        scaled_channel = normalize_signal(upscaled_channels[:, i]) * np.max(np.abs(channel))
        scaled_upscaled_channels.append(scaled_channel)
    scaled_upscaled_channels = np.column_stack(scaled_upscaled_channels)

    logger.info("Normalizing audio...")
    normalized_upscaled_channels = []
    for i in range(scaled_upscaled_channels.shape[1]):
        normalized_channel = normalize_signal(scaled_upscaled_channels[:, i])
        normalized_upscaled_channels.append(normalized_channel)
    normalized_upscaled_channels = np.column_stack(normalized_upscaled_channels)

    new_sample_rate = sample_rate * upscale_factor
    write_audio(output_file_path, new_sample_rate, normalized_upscaled_channels, target_format)
    logger.info(f"Saved processed {target_format.upper()} file at {output_file_path}")