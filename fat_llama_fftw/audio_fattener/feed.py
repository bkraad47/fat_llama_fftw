import os
import logging

import numpy as np
import pyfftw
from pydub import AudioSegment
import soundfile as sf
from mutagen.mp3 import MP3
from mutagen.flac import FLAC
from mutagen.oggvorbis import OggVorbis
from mutagen.wave import WAVE

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache pyfftw's planned transforms so repeated same-shape calls (the
# per-block IST loop below calls perform_ist_iteration/initialize_ist many
# times against a fixed block size) reuse a cached FFTW plan instead of
# re-planning from scratch on every call.
pyfftw.interfaces.cache.enable()

# README documents "Multi-Threaded processing on cpu" as a feature, but
# prior to an earlier cycle no call site actually passed FFTW a threads=
# value above its own default (which pyfftw's numpy_fft interfaces resolve
# to a single thread) - every FFT/IFFT in this module ran single-threaded
# regardless of the host's core count. Multi-threading FFTW is only a net
# win once a transform is large enough to amortize its own thread-dispatch
# overhead: measured directly, requesting more threads on a small
# (block_size-scale, 8192-sample) transform is 1.6x-9x SLOWER than a single
# thread (thread spawn/join overhead dwarfs the actual FFT work at that
# size), while on a large (~4.7M-sample, real upscaled-file-scale) rfft/
# irfft it is faster on a warm (already-planned) FFTW plan - measured on a
# 20-core host: 0.77x (i.e. slightly SLOWER) at 200k samples, 1.37x at 1M,
# 1.60x at 2M, and 4.48x at the real ~4,672,878-sample post-upscale length
# (not the ~6x this comment previously claimed - that number came from a
# less careful benchmark). The two paths' output is numerically equivalent,
# NOT bit-for-bit identical, at every length: verified directly, they match
# exactly at n=200k/1M/2M but differ at n=300k/500k and at the real
# 4,672,878-sample length, by up to ~3.8e-7 of the signal's own peak. This
# is float32 rounding from FFTW's threaded planner choosing a different (but
# equally valid) decomposition of the same transform at some lengths, not a
# change in what is computed - see
# test_apply_nyquist_cutoff_requests_multiple_threads_for_large_signal's
# tolerance-based (not exact-equality) regression coverage for this.
# Measured crossover to a reliable win sits below 1M samples but above
# 200k, with some noise near the boundary; _fft_thread_count uses 200k as
# a threshold so this engages for apply_nyquist_cutoff's whole-signal
# transforms on realistically-sized (post-upscale) audio, never for
# perform_ist_iteration's fixed-size per-block transforms in the IST loop,
# which stay single-threaded.
_MULTI_THREAD_FFT_MIN_SAMPLES = 200_000


def _fft_thread_count(n):
    """How many FFTW threads to request for a transform of length n.

    Single-threaded below _MULTI_THREAD_FFT_MIN_SAMPLES (multi-threading
    would be a net loss there - see the module-level comment above),
    otherwise all available CPU cores.
    """
    if n < _MULTI_THREAD_FFT_MIN_SAMPLES:
        return 1
    return os.cpu_count() or 1


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
        sf.write(file_path, data.astype(np.float32), sample_rate,
                 format='FLAC', subtype='PCM_24')
    elif format == 'wav':
        sf.write(file_path, data.astype(np.float32), sample_rate,
                 format='WAV', subtype='PCM_24')
    else:
        raise ValueError(f"Unsupported target format: {format}")


def new_interpolation_algorithm(data, upscale_factor):
    # Zero-order-hold: each input sample repeated upscale_factor times.
    # np.repeat is the vectorized form of exactly that - a plain Python
    # double loop over every output sample (the previous implementation)
    # does the identical thing at roughly two orders of magnitude more
    # wall-clock cost for real file lengths (measured: ~54x slower on a
    # 200k-sample/4x-upscale array), since it re-enters the Python
    # interpreter once per output sample instead of running the repeat in
    # C. Output is bit-for-bit identical to the old loop (verified
    # directly), so this is a pure performance change, not a behavior one.
    return np.repeat(np.asarray(data), upscale_factor).astype(np.float32)


def initialize_ist(data, threshold):
    # threshold is a fraction (0-1) of this array's own peak magnitude, not
    # an absolute cutoff - see perform_ist_iteration's docstring-equivalent
    # comment below for why. A sample survives only if it's within
    # (1 - threshold) of the loudest sample present.
    peak = np.max(np.abs(data))
    if peak == 0:
        return np.zeros_like(data)
    mask = np.abs(data) > threshold * peak
    data_thres = np.where(mask, data, 0)
    return data_thres


def perform_ist_iteration(data_thres, threshold):
    # threshold_value is dimensionless (fraction of peak magnitude), scaled
    # against each domain's own current peak rather than compared directly
    # to raw magnitudes. Real audio samples/FFT bins are raw int16-range
    # values (or larger, for FFT bin sums), so comparing a fixed threshold
    # like the documented default 0.6 straight against them keeps ~100% of
    # samples/bins - IST degenerates into an fft/ifft identity with no
    # sparsification, hence no "significant frequencies kept, noise
    # discarded" and no added detail (see project-mission.md's IST
    # description and README's "Why FFT and IST?"). Scaling by the current
    # peak makes the same threshold_value meaningful regardless of the
    # input's absolute numeric scale.
    data_fft = pyfftw.interfaces.numpy_fft.fft(data_thres)
    fft_peak = np.max(np.abs(data_fft))
    if fft_peak == 0:
        return np.zeros_like(data_thres, dtype=np.float64)
    mask = np.abs(data_fft) > threshold * fft_peak
    # The DC bin (index 0) carries no spectral detail - only a constant
    # offset - so it is never a legitimate IST "kept frequency" and must be
    # excluded outright, regardless of whether it individually clears the
    # peak-relative threshold. In practice it often does clear it: any
    # asymmetric transient content (e.g. a kick-drum-like spike whose
    # positive excursion outweighs its negative one) concentrates a large
    # share of the block's broadband energy at bin 0, sometimes making it
    # the single loudest bin. Keeping it would inject that as a literal DC
    # offset into ist_changes, which upscale_channels adds onto the whole
    # channel (root cause of a cycle-2 regression: +41.7 dB DC offset plus
    # a low-frequency drone measured across an entire 15s output file).
    mask[0] = False
    data_fft_thres = np.where(mask, data_fft, 0)
    data_thres = pyfftw.interfaces.numpy_fft.ifft(data_fft_thres).real
    return data_thres


def _ist_chain(data, max_iter, threshold, convergence_tol):
    # Each IST pass must refine the *previous* pass's own estimate - that
    # sequential dependency is the algorithm (see README's "Why FFT and
    # IST?" and project-mission.md). A ThreadPoolExecutor that submits all
    # max_iter iterations concurrently against the same starting
    # data_thres cannot express that chain: every worker independently
    # re-derives from the same input, so the result is whichever future
    # happens to finish last - equivalent to a single iteration, no matter
    # how large max_iter is. Chain them explicitly instead.
    #
    # Hard-threshold IST is also a fixed-point projection: fft/ifft are
    # exact inverses, so once a pass's thresholded frequency support is
    # produced, the next pass's fft reproduces that same support (up to
    # floating-point rounding) and thresholding it again is a no-op.
    # Running the full max_iter passes regardless would burn compute on
    # iterations that provably do not change the result, so stop as soon
    # as a pass changes the estimate by less than convergence_tol relative
    # to its own scale - this returns the same fixed point max_iter would
    # have reached, just without the wasted passes.
    data_thres = initialize_ist(data, threshold)
    scale = max(np.max(np.abs(data_thres)), 1e-12)

    for _ in range(max_iter):
        next_data_thres = perform_ist_iteration(data_thres, threshold)
        diff = np.max(np.abs(next_data_thres - data_thres))
        converged = diff < convergence_tol * scale
        data_thres = next_data_thres
        if converged:
            break

    return data_thres


def iterative_soft_thresholding(data, max_iter, threshold,
                                convergence_tol=1e-6, block_size=8192):
    # Cycle-2 root cause (found by audio-quality-checker in cycle 3):
    # thresholding the *whole file* against a single global FFT peak means
    # "within threshold of the loudest bin" is dominated by whichever
    # single moment/frequency is loudest across the entire signal - a real
    # music file's dynamic range spans ~60-100 dB, so essentially nothing
    # outside that one loud moment ever clears the cutoff, leaving other
    # frequency bands with no measurable added detail. Below block_size
    # samples this still runs as a single whole-signal chain (unchanged
    # behavior, and the common case for short test/synthetic inputs).
    # Above it, process the signal as 50%-overlapping, Hann-windowed
    # blocks (STFT-style) and overlap-add the results: the peak-relative
    # threshold in perform_ist_iteration/initialize_ist then reflects each
    # block's own local dynamics, so quieter passages/bands get a
    # meaningful cutoff relative to themselves instead of being judged
    # against the whole file's single loudest partial. This still stays
    # within the FFT/IST method (no new mechanism, just where each FFT's
    # window boundary sits) and each block's own hard-threshold fixed
    # point is found independently, so this does not reintroduce the
    # cycle-1 non-chaining bug or lose its convergence early-exit.
    n = len(data)
    if n <= block_size:
        return _ist_chain(data, max_iter, threshold, convergence_tol)

    hop = block_size // 2
    # WOLA (weighted overlap-add), not COLA-on-analysis-alone (cycle-4 fix
    # for a regression found in the previous cycle: a broadband impulse
    # train locked to this hop rate, ~75 Hz for the tested file). The old
    # code windowed only the *analysis* side (frame = padded[...] * window)
    # then overlap-added frame_result un-windowed. COLA (adjacent windows
    # summing to a flat constant) is only a valid reconstruction argument
    # for a *linear* per-block operation - windowing the input and trusting
    # the operation to preserve that taper on the way out. _ist_chain is a
    # nonlinear hard-threshold projection in the FFT domain: measured
    # directly (a single stationary tone framed through one block), an
    # analysis-windowed frame's edges taper to ~1e-4 of its own peak, but
    # perform_ist_iteration's output has edges back up at ~8-11% of that
    # frame's own peak - thresholding+ifft does not preserve the input's
    # time-domain taper. Summing that un-tapered edge content in at full
    # weight at every hop boundary is exactly the discontinuity that
    # produces the reported artifact.
    #
    # The WOLA fix applies a synthesis window to frame_result too, before
    # accumulation, so each block's own contribution is forced back down
    # toward ~0 at its edges regardless of what the nonlinear operation did
    # there - the discontinuity cannot enter the sum. Perfect reconstruction
    # of the *linear* overlap-add then requires the analysis*synthesis
    # window product to itself satisfy COLA at this hop: a plain Hann used
    # on both sides would not (Hann(n)**2 + Hann(n+hop)**2 is not constant,
    # only Hann(n) + Hann(n+hop) is), so use sqrt-Hann for both analysis and
    # synthesis - its square is exactly the plain Hann used before, whose
    # sum at 50% hop is the flat constant 1.0.
    window = np.sqrt(
        0.5 - 0.5 * np.cos(2 * np.pi * np.arange(block_size) / block_size))
    window_sum = np.sum(window)

    pad = hop
    padded = np.concatenate([
        np.zeros(pad, dtype=np.float64),
        np.asarray(data, dtype=np.float64),
        np.zeros(block_size, dtype=np.float64),
    ])
    padded_len = len(padded)

    output = np.zeros(padded_len, dtype=np.float64)
    weight = np.zeros(padded_len, dtype=np.float64)
    for start in range(0, padded_len - block_size + 1, hop):
        frame = padded[start:start + block_size] * window
        frame_result = _ist_chain(frame, max_iter, threshold,
                                  convergence_tol)
        # frame_result's own mean is (numerically) exactly zero -
        # perform_ist_iteration always excludes the FFT DC bin - but
        # multiplying by the synthesis window is a frequency-domain
        # convolution with the window's own spectrum, which is not a
        # perfect delta at 0 Hz (Hann/sqrt-Hann has a finite main lobe
        # plus sidelobes). That convolution leaks frame_result's own
        # near-DC (very low frequency) content into the windowed frame's
        # DC bin, even though frame_result itself has none. This needs
        # correcting, but NOT via a flat scalar subtracted across the
        # whole block (the cycle-5 fix's original form, `windowed_result -
        # np.mean(windowed_result)`): by this point the synthesis window
        # has already tapered windowed_result's own edges to ~0, and a flat
        # subtraction pushes those edges away from 0 by the raw mean
        # instead - which is negligible for a stationary single tone (per-
        # block mean stays ~1e-6 relative to peak there) but not for real,
        # non-stationary programme material, where a loud/transient
        # block's mean can be large enough that un-tapering its edges this
        # way creates a genuine boundary discontinuity at every hop
        # boundary (root-caused this cycle via direct measurement against
        # real programme material: boundary-phase-folded |second
        # difference| landed double-digit dB above the signal's own
        # typical/median curvature with the flat-scalar form). Instead,
        # subtract a copy of the correction shaped BY the synthesis window
        # itself (window * (sum(windowed_result) / sum(window))): its sum
        # equals sum(windowed_result), so it removes the exact same total
        # DC/near-DC contribution (verified directly: relative DC stays
        # many orders of magnitude below the regression bound), but the
        # correction itself tapers to ~0 at the block's edges just like
        # windowed_result already does, so it does not un-taper them and
        # does not reopen the cycle-4 edge-taper/click regression.
        windowed_result = frame_result * window
        windowed_result = windowed_result - window * (
            np.sum(windowed_result) / window_sum)
        output[start:start + block_size] += windowed_result
        weight[start:start + block_size] += window * window

    safe_weight = np.where(weight > 1e-8, weight, 1.0)
    output = output / safe_weight

    return output[pad:pad + n].astype(np.float32)


def _cap_ist_changes_to_baseline_peak(expanded_channel, ist_changes,
                                      max_rounds=20):
    # perform_ist_iteration's peak-relative FFT threshold keeps/boosts
    # whichever frequency dominates a block's own spectrum - for real
    # music that is usually low-frequency content, since natural audio
    # spectra carry more energy there. Left unchecked, that boost inflates
    # this channel's own peak above what plain zero-order-hold
    # interpolation alone already had. upscale()'s later stages - the
    # auto-scale-to-original-peak step and the final normalize_signal call
    # - are each a single per-channel scalar multiply (and, per
    # apply_nyquist_cutoff being linear, a scalar multiply commutes
    # straight through the cutoff too), so their net effect on the
    # *written* output is entirely determined by this channel's own peak
    # at this point, not by which specific frequencies contributed to it.
    # Measured directly: the auto-scale step is mathematically inert on
    # the final output (its scalar cancels exactly against the mandatory
    # final normalize, to ~1e-6 float rounding) - so an inflated peak here
    # is not "corrected" downstream, it is what determines how hard every
    # OTHER frequency IST never touched gets divided down when the whole
    # channel is renormalized to full scale. That is the confirmed root
    # cause of a net attenuation (up to ~1.3dB, worst at ~2kHz-22kHz,
    # tapering to near 0dB in the low bands IST itself boosts) relative to
    # a no-IST/plain-interpolation control, measured on real programme
    # material (input_test.mp3): IST was adding real detail in raw terms,
    # but the peak growth that addition caused got "spent" out of every
    # frequency, not just the ones IST boosted.
    #
    # Capping ist_changes so the combined (interpolation + IST) signal's
    # own peak does not exceed the pre-IST interpolated baseline's peak
    # keeps IST's contribution - it is rescaled, never zeroed outright -
    # while preventing that contribution from taxing untouched frequencies
    # at the final normalize. This does not touch perform_ist_iteration's
    # own FFT-domain threshold decision or the WOLA block reconstruction
    # at all (it only rescales the whole per-channel ist_changes array by
    # a scalar, after iterative_soft_thresholding has already returned),
    # so it carries no risk to the block-hop-artifact fixes those rely on.
    #
    # A single such correction is only approximate (rescaling can shift
    # which sample is the new peak), and repeated rounds converge closer
    # to fully matching the baseline peak - but full convergence was
    # measured (directly, via a bisection search for the exact scale) to
    # squeeze ist_changes down toward zero, i.e. it trades the net-
    # attenuation finding for reintroducing the "no measurable added
    # detail" bug fixed in prior cycles, so max_rounds must stay bounded,
    # not a convergence loop like _ist_chain's own early-exit.
    #
    # This cycle re-measured that bound/reduction tradeoff directly (on
    # both a synthetic loud-low-tone/quiet-high-tone fixture and a real
    # input_test.mp3 slice) at several round counts and found the
    # previous max_rounds=5 was not actually near the tradeoff's practical
    # limit: it left combined_peak ~12.5-15% above baseline_peak and the
    # worst-band attenuation around -1.0 to -1.24dB, while ist_changes
    # still retained 23-37% of its own uncapped peak magnitude - i.e.
    # there was real, unused headroom between "5 rounds" and "enough
    # rounds to reopen the no-added-detail bug" (empirically that only
    # starts to bite under ~1% surviving fraction, per the same bisection
    # investigation). max_rounds=20 was chosen from that same measurement:
    # it roughly triples the attenuation improvement (worst band ~-0.36 to
    # -0.40dB instead of ~-1.0 to -1.24dB, combined_peak within ~4.4-4.6%
    # of baseline instead of ~12.5-15%) while ist_changes still keeps
    # 7-13% of its own uncapped peak magnitude - comfortably above the
    # near-zero range where the earlier investigation found the
    # no-added-detail bug reopens. Two genuinely different (not just
    # "more rounds of the same scalar") corrections were also tried this
    # cycle and rejected on direct measurement: (a) a sample-wise time-
    # domain peak limiter (clip ist_changes only at samples where combined
    # exceeds baseline_peak, leaving every other sample untouched) - on
    # the synthetic fixture this touched >50% of samples (IST's boost is a
    # broadly, coherently boosted quasi-periodic component, not a sparse
    # handful of outliers) and made the high-band attenuation *worse*
    # (-6.7dB vs -1.24dB uncapped-scalar), i.e. hard-clipping a large
    # contiguous fraction of a waveform's crest is more distorting than a
    # uniform scalar shrink of that same region; (b) a frequency-domain
    # correction that boosted only the FFT bins ist_changes left untouched
    # by the scalar amount needed to offset the eventual normalize - this
    # is unstable: boosting "untouched" bins raises the time-domain peak
    # further, which increases the very shortfall it was trying to correct
    # (measured: peak inflation grew from 4.3dB to 6.25dB and the high
    # band attenuation worsened to -1.96dB). Both confirm the directive's
    # own assessment that a scalar-rescale-based fix is close to its
    # practical limit for this mechanism - max_rounds=20 is the verified
    # improvement available within that limit, not a full fix.
    baseline_peak = np.max(np.abs(expanded_channel))
    if baseline_peak == 0:
        return ist_changes
    current = ist_changes
    for _ in range(max_rounds):
        combined = expanded_channel.astype(np.float32) + current
        combined_peak = np.max(np.abs(combined))
        if combined_peak <= baseline_peak or combined_peak == 0:
            break
        current = current * (baseline_peak / combined_peak)
    return current


def upscale_channels(channels, upscale_factor, max_iter, threshold):
    processed_channels = []
    for channel in channels.T:
        logger.info("Interpolating data...")
        expanded_channel = new_interpolation_algorithm(channel,
                                                        upscale_factor)

        logger.info("Performing IST...")
        ist_changes = iterative_soft_thresholding(expanded_channel,
                                                   max_iter, threshold)
        ist_changes = _cap_ist_changes_to_baseline_peak(expanded_channel,
                                                        ist_changes)
        expanded_channel = expanded_channel.astype(np.float32) + ist_changes

        processed_channels.append(expanded_channel)

    return np.column_stack(processed_channels)


def normalize_signal(signal):
    return signal / np.max(np.abs(signal))


def apply_nyquist_cutoff(signal, sample_rate, original_nyquist):
    """Zero out all FFT bins above the original source's Nyquist frequency.

    fat_llama upscales precision/headroom within the original recording's
    real bandwidth - it must never leave (or reintroduce) audible content
    above original_sample_rate / 2 in the output, whether that comes from
    the zero-order-hold interpolation's spectral imaging, IST, or anything
    else. This runs after amplitude auto-scaling but *before* upscale()'s
    final normalization step (not strictly last overall) - a brick-wall
    FFT filter like this one can overshoot its own input's peak
    (Gibbs-phenomenon ripple), so the final normalization must come after
    it to guarantee the signal actually written never exceeds full scale.
    """
    n = len(signal)
    threads = _fft_thread_count(n)
    spectrum = pyfftw.interfaces.numpy_fft.rfft(signal, threads=threads)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    spectrum = np.where(freqs > original_nyquist, 0, spectrum)
    filtered = pyfftw.interfaces.numpy_fft.irfft(spectrum, n=n,
                                                 threads=threads)
    return filtered.astype(np.float32)


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
        raise ValueError(
            f"{target_format.upper()} bitrate out of range. Please "
            f"provide a value between {min_bitrate} and {max_bitrate} kbps."
        )

    logger.info(f"Loading {source_format.upper()} file...")
    sample_rate, samples, bitrate, audio = read_audio(input_file_path,
                                                       format=source_format)
    if bitrate:
        logger.info(
            f"Original {source_format.upper()} bitrate: "
            f"{bitrate / 1000:.2f} kbps"
        )

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
        scaled_channel = (normalize_signal(upscaled_channels[:, i])
                          * np.max(np.abs(channel)))
        scaled_upscaled_channels.append(scaled_channel)
    scaled_upscaled_channels = np.column_stack(scaled_upscaled_channels)

    new_sample_rate = sample_rate * upscale_factor

    # The Nyquist cutoff (a brick-wall FFT lowpass) runs *before* the final
    # normalization, not after. A brick-wall filter can overshoot its own
    # input's peak (Gibbs-phenomenon ripple), so filtering a
    # already-peak-normalized-to-1.0 signal can push some samples back
    # above full scale - write_audio's PCM_24 subtype then silently clips
    # them on write (measured: a handful of samples pinned at exactly
    # +/-1.0 in the output, clipping fraction ~4e-6). Running the cutoff
    # first and normalizing its output last guarantees the signal actually
    # handed to write_audio peaks at exactly 1.0 - not "close to it, unless
    # the filter pushed it over" - by construction, regardless of any
    # overshoot the filter introduces.
    logger.info("Removing content above the original Nyquist frequency...")
    original_nyquist = sample_rate / 2.0
    filtered_upscaled_channels = []
    for i in range(scaled_upscaled_channels.shape[1]):
        filtered_channel = apply_nyquist_cutoff(
            scaled_upscaled_channels[:, i], new_sample_rate,
            original_nyquist
        )
        filtered_upscaled_channels.append(filtered_channel)
    filtered_upscaled_channels = np.column_stack(filtered_upscaled_channels)

    logger.info("Normalizing audio...")
    final_channels = []
    for i in range(filtered_upscaled_channels.shape[1]):
        normalized_channel = normalize_signal(
            filtered_upscaled_channels[:, i])
        final_channels.append(normalized_channel)
    final_channels = np.column_stack(final_channels)

    write_audio(output_file_path, new_sample_rate, final_channels,
               target_format)
    logger.info(f"Saved processed {target_format.upper()} file at "
               f"{output_file_path}")
