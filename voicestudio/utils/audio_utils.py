"""Waveform analysis shared by the models and the metrics."""

import functools
import math

import numpy as np
import torch
import torchaudio


def show_waveform(
    audio_path: str | None,
    waveform: torch.Tensor | None = None,
    sr: int = 48000,
    max_points: int = 11025
):
    try:
        import matplotlib.pyplot as plt
        from IPython.display import Audio
    except ImportError:  # Not in Jupyter notebook
        return None

    if audio_path:
        waveform, sr = torchaudio.load(audio_path)
    elif waveform is not None:
        waveform = waveform.unsqueeze(0) if len(waveform.shape) == 1 else waveform
    else:
        raise ValueError("Either audio_path or waveform must be provided.")

    samples = waveform[0].detach().cpu()
    # Non-overlapping max envelope, so a long clip draws at most max_points bands
    hop = max(1, samples.shape[-1] // max_points)
    envelope = samples[: samples.shape[-1] // hop * hop].abs().reshape(-1, hop).amax(dim=1)
    times = torch.arange(envelope.shape[0], dtype=torch.float32) * hop / sr

    plt.figure(figsize=(10, 4))
    plt.fill_between(times, -envelope, envelope, step="pre", linewidth=0)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

    return Audio(samples, rate=sr)


@functools.cache
def _load_pyworld():
    """Returns the `pyworld` module when it is importable, and `None` otherwise."""
    try:
        import pyworld
    except ImportError:
        return None
    return pyworld


_WORLD_PI = 3.1415926535897932384


def _read_only(array):
    array.flags.writeable = False
    return array


@functools.lru_cache(maxsize=256)
def _nuttall_window(length):
    """Returns the read-only `length`-point Nuttall window."""
    tmp = np.arange(length, dtype=np.float64) / (length - 1.0)
    return _read_only(
        0.355768
        - 0.487396 * np.cos(2.0 * _WORLD_PI * tmp)
        + 0.144232 * np.cos(4.0 * _WORLD_PI * tmp)
        - 0.012604 * np.cos(6.0 * _WORLD_PI * tmp)
    )


@functools.lru_cache(maxsize=256)
def _band_pass_filter_spectrum(boundary_f0, fs, filter_length_half, fft_size):
    """Returns the transform of the Nuttall-windowed band-pass filter centred on `boundary_f0`."""
    band_pass_filter = np.zeros(fft_size, dtype=np.float64)
    taps = _nuttall_window(filter_length_half * 2 + 1)
    lags = np.arange(-filter_length_half, filter_length_half + 1, dtype=np.float64)
    band_pass_filter[: filter_length_half * 2 + 1] = taps * np.cos(2 * _WORLD_PI * boundary_f0 * lags / fs)
    return _read_only(np.fft.rfft(band_pass_filter))


@functools.lru_cache(maxsize=256)
def _low_pass_filter_spectrum(half_average_length, fft_size):
    """Returns the transform of DIO's Nuttall low-pass of `half_average_length * 4` taps."""
    low_pass_filter = np.zeros(fft_size, dtype=np.float64)
    low_pass_filter[: half_average_length * 4] = _nuttall_window(half_average_length * 4)
    return _read_only(np.fft.rfft(low_pass_filter))


@functools.lru_cache(maxsize=256)
def _low_cut_filter_spectrum(taps, fft_size):
    """Returns the transform of the `taps`-point Hann-derived low-cut filter."""
    filter_ = np.zeros(fft_size, dtype=np.float64)
    positions = np.arange(1, taps + 1, dtype=np.float64)
    filter_[:taps] = 0.5 - 0.5 * np.cos(positions * 2.0 * _WORLD_PI / (taps + 1))
    filter_[:taps] = -filter_[:taps] / filter_[:taps].sum()
    half = (taps - 1) // 2
    filter_[fft_size - half :] = filter_[:half]
    filter_[:taps] = filter_[half : half + taps].copy()
    filter_[0] += 1.0
    return _read_only(np.fft.rfft(filter_))


class WorldF0Estimator:
    r"""
    Constructs the WORLD f0 estimator, exposing
    [`~WorldF0Estimator.harvest`], [`~WorldF0Estimator.dio`] and
    [`~WorldF0Estimator.stonemask`]. Each delegates to `pyworld` where that package is
    importable and runs the ported implementation otherwise; the two produce the same contour.

    Args:
        sampling_rate (`int`):
            Rate of the waveforms passed to the estimators.
        f0_floor (`float`, *optional*, defaults to 71.0):
            Lowest f0 the search considers, in Hz.
        f0_ceil (`float`, *optional*, defaults to 800.0):
            Highest f0 the search considers, in Hz.
        prefer_pyworld (`bool`, *optional*, defaults to `True`):
            Whether to delegate to `pyworld` when it is installed. Pass `False` to run the ported
            implementation regardless.
    """

    _LOG2 = 0.69314718055994529
    _PI = 3.1415926535897932384
    _SAFE_GUARD_MINIMUM = 0.000000000001
    _FLOOR_F0_STONEMASK = 40.0
    _CUT_OFF = 50.0
    _MAXIMUM_VALUE = 100000.0

    _DECIMATE_COEFFICIENTS = {
        2: ((0.041156734567757189, -0.42599112459189636, 0.041037215479961225), (0.16797464681802227, 0.50392394045406674)),
        3: ((0.95039378983237421, -0.67429146741526791, 0.15412211621346475), (0.071221945171178636, 0.21366583551353591)),
        4: ((1.4499664446880227, -0.98943497080950582, 0.24578252340690215), (0.036710750339322612, 0.11013225101796784)),
        5: ((1.7610939654280557, -1.2554914843859768, 0.3237186507788215), (0.021334858522387423, 0.06400457556716227)),
        6: ((1.9715352749512141, -1.4686795689225347, 0.3893908434965701), (0.013469181309343825, 0.040407543928031475)),
        7: ((2.1225239019534703, -1.6395144861046302, 0.44469707800587366), (0.0090366882681608418, 0.027110064804482525)),
        8: ((2.2357462340187593, -1.7780899984041358, 0.49152555365968692), (0.0063522763407111993, 0.019056829022133598)),
        9: ((2.3236003491759578, -1.8921545617463598, 0.53148928133729068), (0.0046331164041389372, 0.013899349212416812)),
        10: ((2.3936475118069387, -1.9873904075111861, 0.5658879979027055), (0.0034818622251927556, 0.010445586675578267)),
        11: ((2.450743295230728, -2.06794904601978, 0.59574774438332101), (0.0026822508007163792, 0.0080467524021491377)),
        12: ((2.4981398605924205, -2.1368928194784025, 0.62187513816221485), (0.0021097275904709001, 0.0063291827714127002)),
    }

    _SMOOTHING_A = (1.7347257688092754, -0.76600660094326412)
    _SMOOTHING_B = (0.0078202080334971724, 0.015640416066994345)

    def __init__(
        self,
        sampling_rate: int,
        f0_floor: float = 71.0,
        f0_ceil: float = 800.0,
        prefer_pyworld: bool = True,
    ):
        self.sampling_rate = sampling_rate
        self.f0_floor = f0_floor
        self.f0_ceil = f0_ceil
        self.pyworld = _load_pyworld() if prefer_pyworld else None

    def harvest(self, waveform: np.ndarray, frame_period: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimates the f0 contour with Harvest.

        Args:
            waveform (`np.ndarray`):
                Mono waveform.
            frame_period (`float`, *optional*, defaults to 5.0):
                Spacing between analysis frames, in milliseconds.

        Returns:
            `tuple[np.ndarray, np.ndarray]`: the f0 contour in Hz and the frame positions in seconds.
        """
        x = np.ascontiguousarray(waveform, dtype=np.float64)
        if self.pyworld is not None:
            return self.pyworld.harvest(
                x, self.sampling_rate, f0_floor=self.f0_floor, f0_ceil=self.f0_ceil, frame_period=frame_period
            )

        fs = self.sampling_rate
        channels_in_octave = 40.0
        dimension_ratio = self._matlab_round(fs / 8000.0)
        if frame_period == 1.0:
            positions, f0 = self._harvest_general_body(x, fs, 1, channels_in_octave, dimension_ratio)
            return f0, positions

        _, basic_f0 = self._harvest_general_body(x, fs, 1, channels_in_octave, dimension_ratio)
        f0_length = self._samples_for_harvest(fs, x.shape[0], frame_period)
        positions = np.arange(f0_length, dtype=np.float64) * frame_period / 1000.0
        index = np.minimum(basic_f0.shape[0] - 1, self._matlab_round_array(positions * 1000.0))
        return basic_f0[index], positions

    def dio(
        self,
        waveform: np.ndarray,
        frame_period: float = 5.0,
        channels_in_octave: float = 2.0,
        speed: int = 1,
        allowed_range: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimates the f0 contour with DIO.

        Args:
            waveform (`np.ndarray`):
                Mono waveform.
            frame_period (`float`, *optional*, defaults to 5.0):
                Spacing between analysis frames, in milliseconds.
            channels_in_octave (`float`, *optional*, defaults to 2.0):
                Number of band-pass channels per octave.
            speed (`int`, *optional*, defaults to 1):
                Decimation ratio applied before the analysis, from 1 to 12.
            allowed_range (`float`, *optional*, defaults to 0.1):
                Relative f0 jump the postprocessing tolerates between frames.

        Returns:
            `tuple[np.ndarray, np.ndarray]`: the f0 contour in Hz and the frame positions in seconds.
        """
        x = np.ascontiguousarray(waveform, dtype=np.float64)
        if self.pyworld is not None:
            return self.pyworld.dio(
                x,
                self.sampling_rate,
                f0_floor=self.f0_floor,
                f0_ceil=self.f0_ceil,
                channels_in_octave=channels_in_octave,
                frame_period=frame_period,
                speed=speed,
                allowed_range=allowed_range,
            )

        fs = self.sampling_rate
        x_length = x.shape[0]
        number_of_bands = 1 + int(math.log(self.f0_ceil / self.f0_floor) / self._LOG2 * channels_in_octave)
        boundary_f0_list = self.f0_floor * np.power(
            2.0, np.arange(1, number_of_bands + 1, dtype=np.float64) / channels_in_octave
        )

        decimation_ratio = max(min(speed, 12), 1)
        y_length = 1 + int(x_length / decimation_ratio)
        actual_fs = fs / decimation_ratio
        fft_size = self._suitable_fft_size(
            y_length
            + self._matlab_round(actual_fs / self._CUT_OFF) * 2
            + 1
            + 4 * int(1.0 + actual_fs / boundary_f0_list[0] / 2.0)
        )

        spectrum = self._spectrum_for_estimation(x, y_length, actual_fs, fft_size, decimation_ratio)
        f0_length = self._samples_for_harvest(fs, x_length, frame_period)
        temporal_positions = np.arange(f0_length, dtype=np.float64) * frame_period / 1000.0

        candidates, scores = self._dio_candidates_and_scores(
            boundary_f0_list, actual_fs, y_length, temporal_positions, spectrum, fft_size
        )
        best = self._dio_best_f0_contour(candidates, scores)
        f0 = self._dio_fix_f0_contour(frame_period, candidates, best, allowed_range)
        return f0, temporal_positions

    def stonemask(self, waveform: np.ndarray, temporal_positions: np.ndarray, f0: np.ndarray) -> np.ndarray:
        """
        Refines an f0 contour by instantaneous frequency.

        Args:
            waveform (`np.ndarray`):
                Mono waveform.
            temporal_positions (`np.ndarray`):
                Frame positions of `f0`, in seconds.
            f0 (`np.ndarray`):
                Contour to refine, in Hz.

        Returns:
            `np.ndarray`: the refined contour, in Hz.
        """
        x = np.ascontiguousarray(waveform, dtype=np.float64)
        f0 = np.ascontiguousarray(f0, dtype=np.float64)
        positions = np.ascontiguousarray(temporal_positions, dtype=np.float64)
        if self.pyworld is not None:
            return self.pyworld.stonemask(x, f0, positions, self.sampling_rate)

        fs = self.sampling_rate
        refined = np.zeros_like(f0)
        frames = np.flatnonzero((f0 > self._FLOOR_F0_STONEMASK) & (f0 <= fs / 12.0))
        if frames.shape[0] == 0:
            return refined

        f0s = f0[frames]
        half_window_lengths = (1.5 * fs / f0s + 1.0).astype(np.int64)
        for half_window_length in np.unique(half_window_lengths):
            selected = np.flatnonzero(half_window_lengths == half_window_length)
            width = int(half_window_length) * 2 + 1
            fft_size = int(pow(2.0, 2.0 + int(math.log(width) / self._LOG2)))
            chunk = max(1, 2**25 // fft_size)
            for begin in range(0, selected.shape[0], chunk):
                rows = selected[begin : begin + chunk]
                refined[frames[rows]] = self._stonemask_batch(
                    x, fs, positions[frames[rows]], f0s[rows], int(half_window_length), fft_size
                )
        return refined

    def _matlab_round(self, x):
        return int(x + 0.5) if x > 0 else int(x - 0.5)

    def _matlab_round_array(self, x):
        return np.trunc(np.where(x > 0.0, x + 0.5, x - 0.5)).astype(np.int64)

    def _suitable_fft_size(self, sample):
        return int(pow(2.0, int(math.log(float(sample)) / self._LOG2) + 1))

    def _direct_form_2(self, x, a, b):
        """Runs a direct form II recursion whose state is `len(a)` samples wide."""
        denominator = torch.tensor((1.0, *(-coefficient for coefficient in a)), dtype=torch.float64)
        numerator = torch.tensor(b, dtype=torch.float64)
        # `clamp` defaults to True and would hold the output inside [-1, 1], which is not a range an
        # f0 contour or a decimated waveform stays in.
        filtered = torchaudio.functional.lfilter(
            torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0), denominator, numerator, clamp=False
        )
        return filtered.squeeze(0).numpy()

    def _filter_for_decimate(self, x, ratio):
        a, b = self._DECIMATE_COEFFICIENTS[ratio]
        return self._direct_form_2(x, a, (b[0], b[1], b[1], b[0]))

    def _decimate(self, x, ratio):
        n_fact = 9
        x_length = x.shape[0]
        padded = np.empty(x_length + n_fact * 2, dtype=np.float64)
        padded[:n_fact] = 2 * x[0] - x[n_fact:0:-1]
        padded[n_fact : n_fact + x_length] = x
        tail = np.arange(n_fact, dtype=np.int64)
        padded[n_fact + x_length :] = 2 * x[x_length - 1] - x[x_length - 2 - tail]

        filtered = self._filter_for_decimate(padded, ratio)
        filtered = self._filter_for_decimate(filtered[::-1].copy(), ratio)
        padded = filtered[::-1].copy()

        n_out = (x_length - 1) // ratio + 1
        n_beg = ratio - ratio * n_out + x_length
        positions = np.arange(n_beg, x_length + n_fact, ratio, dtype=np.int64)
        return padded[positions + n_fact - 1]

    def _interp1(self, x, y, xi):
        """Linear interpolation with the clamped-interval extrapolation `histc` gives WORLD."""
        index = np.searchsorted(x, xi, side="right")
        np.clip(index, 1, x.shape[0] - 1, out=index)
        lower = index - 1
        step = (xi - x[lower]) / (x[index] - x[lower])
        return y[lower] + step * (y[index] - y[lower])

    def _zero_crossing_engine(self, signal, length, fs):
        """Returns the reciprocal intervals between successive downward zero crossings."""
        head = signal[: length - 1]
        tail = signal[1:length]
        edges = np.flatnonzero((head > 0.0) & (tail <= 0.0)) + 1
        if edges.shape[0] < 2:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

        before = signal[edges - 1]
        fine_edges = edges - before / (signal[edges] - before)
        intervals = fs / np.diff(fine_edges)
        locations = (fine_edges[:-1] + fine_edges[1:]) / 2.0 / fs
        return locations, intervals

    def _four_zero_crossing_intervals(self, signal, length, fs):
        negative = self._zero_crossing_engine(signal, length, fs)
        inverted = -signal
        positive = self._zero_crossing_engine(inverted, length, fs)
        differentiated = inverted[: length - 1] - inverted[1:length]
        peak = self._zero_crossing_engine(differentiated, length - 1, fs)
        dip = self._zero_crossing_engine(-differentiated, length - 1, fs)
        return negative, positive, peak, dip

    def _filtered_signal(self, boundary_f0, fft_size, fs, spectrum, y_length):
        """Convolves the analysed spectrum with a Nuttall-windowed band-pass filter."""
        filter_length_half = self._matlab_round(fs / boundary_f0 * 2.0)
        filter_spectrum = _band_pass_filter_spectrum(float(boundary_f0), fs, filter_length_half, fft_size)
        signal = np.fft.irfft(spectrum * filter_spectrum, n=fft_size) * fft_size
        index_bias = filter_length_half + 1
        return signal[index_bias : index_bias + y_length]

    def _f0_candidate_contour(self, crossings, boundary_f0, temporal_positions):
        f0_length = temporal_positions.shape[0]
        if any(locations.shape[0] < 3 for locations, _ in crossings):
            return np.zeros(f0_length, dtype=np.float64)

        candidate = np.zeros(f0_length, dtype=np.float64)
        for locations, intervals in crossings:
            candidate += self._interp1(locations, intervals, temporal_positions)
        candidate /= 4.0

        rejected = (
            (candidate > boundary_f0 * 1.1)
            | (candidate < boundary_f0 * 0.9)
            | (candidate > self.f0_ceil)
            | (candidate < self.f0_floor)
        )
        candidate[rejected] = 0.0
        return candidate

    def _raw_f0_candidates(self, boundary_f0_list, actual_fs, y_length, temporal_positions, spectrum, fft_size):
        raw = np.empty((boundary_f0_list.shape[0], temporal_positions.shape[0]), dtype=np.float64)
        for band, boundary_f0 in enumerate(boundary_f0_list):
            signal = self._filtered_signal(boundary_f0, fft_size, actual_fs, spectrum, y_length)
            crossings = self._four_zero_crossing_intervals(signal, y_length, actual_fs)
            raw[band] = self._f0_candidate_contour(crossings, boundary_f0, temporal_positions)
        return raw

    def _detect_official_f0_candidates(self, raw, max_candidates):
        """Averages each band-contiguous voiced run of the per-channel candidates into one candidate."""
        number_of_channels, f0_length = raw.shape
        candidates = np.zeros((f0_length, max_candidates), dtype=np.float64)
        number_of_candidates = 0

        voiced = raw > 0
        voiced[0] = False
        voiced[number_of_channels - 1] = False
        transitions = voiced[1:].astype(np.int8) - voiced[:-1].astype(np.int8)

        for frame in range(f0_length):
            column = transitions[:, frame]
            starts = np.flatnonzero(column == 1) + 1
            ends = np.flatnonzero(column == -1) + 1
            sections = min(starts.shape[0], ends.shape[0])
            count = 0
            for section in range(sections):
                start, end = starts[section], ends[section]
                if end - start < 10:
                    continue
                candidates[frame, count] = raw[start:end, frame].mean()
                count += 1
            number_of_candidates = max(number_of_candidates, count)

        return candidates, number_of_candidates

    def _overlap_f0_candidates(self, candidates, number_of_candidates):
        """Spreads each frame's candidates onto the three frames either side of it."""
        f0_length = candidates.shape[0]
        n = 3
        for shift in range(1, n + 1):
            block = slice(number_of_candidates * shift, number_of_candidates * (shift + 1))
            candidates[shift:, block] = candidates[: f0_length - shift, :number_of_candidates]
            block = slice(number_of_candidates * (shift + n), number_of_candidates * (shift + n + 1))
            candidates[: f0_length - shift, block] = candidates[shift:, :number_of_candidates]
        return candidates

    def _refine_f0_batch(self, x, fs, positions, f0s, half_window_length, fft_size):
        """Refines one batch of candidates sharing a window length, by instantaneous frequency."""
        width = half_window_length * 2 + 1
        window_length_in_time = width / fs

        # One zero column either side lets the interior difference formula also produce the two
        # endpoint values the upstream loop special-cases.
        main_window = np.zeros((f0s.shape[0], width + 2), dtype=np.float64)
        offsets = np.arange(width, dtype=np.int64)

        basic_index = self._matlab_round_array((positions - half_window_length / fs) * fs + 0.001)
        base_index = basic_index[:, None] + offsets[None, :]
        elapsed = (base_index - 1.0) / fs - positions[:, None]
        window = main_window[:, 1 : width + 1]
        np.cos(2.0 * self._PI * elapsed / window_length_in_time, out=window)
        window *= 0.5
        window += 0.42
        window += 0.08 * np.cos(4.0 * self._PI * elapsed / window_length_in_time)
        diff_window = -(main_window[:, 2:] - main_window[:, :width]) / 2.0

        samples = x[np.clip(base_index - 1, 0, x.shape[0] - 1)]

        padded = np.zeros((f0s.shape[0], fft_size), dtype=np.float64)
        np.multiply(samples, window, out=padded[:, :width])
        main_spectrum = np.fft.rfft(padded, axis=-1)
        np.multiply(samples, diff_window, out=padded[:, :width])
        diff_spectrum = np.fft.rfft(padded, axis=-1)

        harmonics = np.arange(1, 7, dtype=np.float64)
        counts = np.minimum((fs / 2.0 / f0s).astype(np.int64), 6)
        used = harmonics[None, :] <= counts[:, None]
        index = self._matlab_round_array(f0s[:, None] * fft_size / fs * harmonics[None, :])
        np.clip(index, 0, fft_size // 2, out=index)

        rows = np.arange(f0s.shape[0], dtype=np.int64)[:, None]
        # Only the bins the harmonics land on are read, so the gather comes before the arithmetic.
        main = main_spectrum[rows, index]
        diff = diff_spectrum[rows, index]
        numerator_i = main.real * diff.imag - main.imag * diff.real
        power = main.real**2 + main.imag**2
        silent = power == 0.0
        instantaneous = np.where(
            silent,
            0.0,
            index * fs / fft_size + numerator_i / np.where(silent, 1.0, power) * fs / 2.0 / self._PI,
        )
        amplitude = np.sqrt(power)

        numerator = np.sum(np.where(used, amplitude * instantaneous, 0.0), axis=-1)
        denominator = np.sum(np.where(used, amplitude * harmonics[None, :], 0.0), axis=-1)
        deviation = np.sum(
            np.where(used, np.abs((instantaneous / harmonics[None, :] - f0s[:, None]) / f0s[:, None]), 0.0), axis=-1
        )

        refined = numerator / (denominator + self._SAFE_GUARD_MINIMUM)
        score = 1.0 / (deviation / counts + self._SAFE_GUARD_MINIMUM)
        rejected = (refined < self.f0_floor) | (refined > self.f0_ceil) | (score < 2.5)
        refined[rejected] = 0.0
        score[rejected] = 0.0
        return refined, score

    def _refine_f0_candidates(self, x, fs, temporal_positions, candidates):
        scores = np.zeros_like(candidates)
        frame, column = np.nonzero(candidates > 0.0)
        if frame.shape[0] == 0:
            return candidates, scores

        f0s = candidates[frame, column]
        positions = temporal_positions[frame]
        half_window_lengths = (1.5 * fs / f0s + 1.0).astype(np.int64)

        for half_window_length in np.unique(half_window_lengths):
            selected = np.flatnonzero(half_window_lengths == half_window_length)
            width = int(half_window_length) * 2 + 1
            fft_size = int(pow(2.0, 2.0 + int(math.log(width) / self._LOG2)))
            chunk = max(1, 2**25 // fft_size)
            for start in range(0, selected.shape[0], chunk):
                rows = selected[start : start + chunk]
                refined, score = self._refine_f0_batch(
                    x, fs, positions[rows], f0s[rows], int(half_window_length), fft_size
                )
                candidates[frame[rows], column[rows]] = refined
                scores[frame[rows], column[rows]] = score

        return candidates, scores

    def _remove_unreliable_candidates(self, candidates, scores):
        """Zeroes a candidate that no neighbouring frame corroborates within five percent."""
        f0_length = candidates.shape[0]
        reference = candidates.copy()
        threshold = 0.05

        for start in range(1, f0_length - 1, 2048):
            stop = min(start + 2048, f0_length - 1)
            centre = reference[start:stop, :, None]
            with np.errstate(divide="ignore", invalid="ignore"):
                forward = np.abs(centre - reference[start + 1 : stop + 1, None, :]) / centre
                backward = np.abs(centre - reference[start - 1 : stop - 1, None, :]) / centre
            error = np.minimum(
                np.minimum(forward.min(axis=-1), 1.0), np.minimum(backward.min(axis=-1), 1.0)
            )
            unreliable = (reference[start:stop] != 0.0) & (error > threshold)
            candidates[start:stop][unreliable] = 0.0
            scores[start:stop][unreliable] = 0.0
        return candidates, scores

    def _search_f0_base(self, candidates, scores):
        best = np.argmax(scores, axis=-1)
        rows = np.arange(candidates.shape[0], dtype=np.int64)
        contour = candidates[rows, best]
        contour[scores[rows, best] <= 0.0] = 0.0
        return contour

    def _fix_step_1(self, f0_base, allowed_range):
        f0_step1 = np.zeros_like(f0_base)
        body = f0_base[2:]
        with np.errstate(divide="ignore", invalid="ignore"):
            reference = f0_base[1:-1] * 2 - f0_base[:-2]
            jumped = np.abs((body - reference) / reference) > allowed_range
            stepped = np.abs(body - f0_base[1:-1]) / f0_base[1:-1] > allowed_range
        f0_step1[2:] = np.where(jumped & stepped, 0.0, body)
        return f0_step1

    def _boundary_list(self, f0):
        voiced = f0 > 0
        voiced[0] = False
        voiced[-1] = False
        edges = np.flatnonzero(voiced[1:] != voiced[:-1]) + 1
        return edges - np.arange(edges.shape[0], dtype=np.int64) % 2

    def _fix_step_2(self, f0_step1, voice_range_minimum):
        f0_step2 = f0_step1.copy()
        boundaries = self._boundary_list(f0_step1)
        for start, end in zip(boundaries[::2], boundaries[1::2]):
            if end - start < voice_range_minimum:
                f0_step2[start : end + 1] = 0.0
        return f0_step2

    def _multi_channel_f0(self, f0, boundaries):
        sections = boundaries.shape[0] // 2
        channels = np.zeros((sections, f0.shape[0]), dtype=np.float64)
        for section in range(sections):
            start, end = boundaries[section * 2], boundaries[section * 2 + 1]
            channels[section, start : end + 1] = f0[start : end + 1]
        return channels

    def _select_best_f0(self, reference_f0, candidates, allowed_range):
        best_f0 = 0.0
        best_error = allowed_range
        for candidate in candidates:
            error = abs(reference_f0 - candidate) / reference_f0
            if error > best_error:
                continue
            best_f0 = candidate
            best_error = error
        return best_f0, best_error

    def _extend_f0(self, origin, last_point, shift, candidates, allowed_range, extended_f0):
        threshold = 4
        tmp_f0 = extended_f0[origin]
        shifted_origin = origin
        count = 0
        for step in range(abs(last_point - origin) + 1):
            target = origin + shift * step + shift
            extended_f0[target], _ = self._select_best_f0(tmp_f0, candidates[target], allowed_range)
            if extended_f0[target] == 0.0:
                count += 1
            else:
                tmp_f0 = extended_f0[target]
                count = 0
                shifted_origin = target
            if count == threshold:
                break
        return shifted_origin

    def _extend(self, channels, f0_length, boundaries, candidates, allowed_range):
        """Grows each voiced section outwards along the candidates that continue its contour."""
        threshold = 100
        sections = channels.shape[0]
        for section in range(sections):
            boundaries[section * 2 + 1] = self._extend_f0(
                boundaries[section * 2 + 1],
                min(f0_length - 2, boundaries[section * 2 + 1] + threshold),
                1,
                candidates,
                allowed_range,
                channels[section],
            )
            boundaries[section * 2] = self._extend_f0(
                boundaries[section * 2],
                max(1, boundaries[section * 2] - threshold),
                -1,
                candidates,
                allowed_range,
                channels[section],
            )

        # The running mean is not reset between sections, matching the upstream accumulation.
        running_mean = 0.0
        count = 0
        for section in range(sections):
            start, end = boundaries[section * 2], boundaries[section * 2 + 1]
            running_mean += channels[section, start:end].sum()
            running_mean /= end - start
            if 2200.0 / running_mean < end - start:
                channels[[count, section]] = channels[[section, count]]
                boundaries[[count * 2, section * 2]] = boundaries[[section * 2, count * 2]]
                boundaries[[count * 2 + 1, section * 2 + 1]] = boundaries[[section * 2 + 1, count * 2 + 1]]
                count += 1
        return count

    def _make_sorted_order(self, boundaries, sections):
        """Reproduces the upstream ordering pass, which compares against the displaced element."""
        order = list(range(sections))
        for i in range(1, sections):
            for j in range(i - 1, -1, -1):
                if boundaries[order[j] * 2] > boundaries[order[i] * 2]:
                    order[i], order[j] = order[j], order[i]
                else:
                    break
        return order

    def _search_score(self, f0, candidates, scores):
        matched = candidates == f0
        if not matched.any():
            return 0.0
        return max(0.0, float(scores[matched].max()))

    def _merge_f0_sub(self, merged, start1, end1, channel, start2, end2, candidates, scores):
        if start1 <= start2 and end1 >= end2:
            return end1

        span = slice(start2, end1 + 1)
        score1 = sum(
            self._search_score(merged[i], candidates[i], scores[i]) for i in range(start2, end1 + 1)
        )
        score2 = sum(
            self._search_score(channel[i], candidates[i], scores[i]) for i in range(start2, end1 + 1)
        )
        del span
        if score1 > score2:
            merged[end1 : end2 + 1] = channel[end1 : end2 + 1]
        else:
            merged[start2 : end2 + 1] = channel[start2 : end2 + 1]
        return end2

    def _merge_f0(self, channels, boundaries, sections, candidates, scores):
        order = self._make_sorted_order(boundaries, sections)
        merged = channels[0].copy()

        for i in range(1, sections):
            current = order[i]
            if boundaries[current * 2] - boundaries[1] > 0:
                start, end = boundaries[current * 2], boundaries[current * 2 + 1]
                merged[start : end + 1] = channels[current, start : end + 1]
                boundaries[0] = start
                boundaries[1] = end
            else:
                boundaries[1] = self._merge_f0_sub(
                    merged,
                    boundaries[0],
                    boundaries[1],
                    channels[current],
                    boundaries[current * 2],
                    boundaries[current * 2 + 1],
                    candidates,
                    scores,
                )
        return merged

    def _fix_step_3(self, f0_step2, candidates, scores, allowed_range):
        f0_step3 = f0_step2.copy()
        f0_length = f0_step2.shape[0]
        boundaries = self._boundary_list(f0_step2)
        if boundaries.shape[0] == 0:
            return f0_step3

        channels = self._multi_channel_f0(f0_step2, boundaries)
        sections = self._extend(channels, f0_length, boundaries, candidates, allowed_range)
        if sections != 0:
            f0_step3 = self._merge_f0(channels, boundaries, sections, candidates, scores)
        return f0_step3

    def _fix_step_4(self, f0_step3, threshold):
        f0_step4 = f0_step3.copy()
        boundaries = self._boundary_list(f0_step3)
        for section in range(boundaries.shape[0] // 2 - 1):
            end = boundaries[section * 2 + 1]
            start = boundaries[(section + 1) * 2]
            distance = start - end - 1
            if distance >= threshold:
                continue
            head = f0_step3[end] + 1
            tail = f0_step3[start] - 1
            coefficient = (tail - head) / (distance + 1.0)
            steps = np.arange(1, start - end, dtype=np.float64)
            f0_step4[end + 1 : start] = head + coefficient * steps
        return f0_step4

    def _fix_f0_contour(self, candidates, scores):
        contour = self._search_f0_base(candidates, scores)
        contour = self._fix_step_1(contour, 0.008)
        contour = self._fix_step_2(contour, 6)
        contour = self._fix_step_3(contour, candidates, scores, 0.18)
        return self._fix_step_4(contour, 9)

    def _filtering_f0(self, x, start, end):
        x = x.copy()
        x[:start] = x[start]
        x[end + 1 :] = x[end]
        a = self._SMOOTHING_A
        b = (self._SMOOTHING_B[0], self._SMOOTHING_B[1], self._SMOOTHING_B[0])
        forward = self._direct_form_2(x, a, b)[::-1].copy()
        return self._direct_form_2(forward, a, b)[::-1].copy()

    def _smooth_f0_contour(self, f0):
        lag = 300
        f0_length = f0.shape[0]
        contour = np.zeros(f0_length + lag * 2, dtype=np.float64)
        contour[lag : lag + f0_length] = f0

        boundaries = self._boundary_list(contour)
        channels = self._multi_channel_f0(contour, boundaries)

        smoothed = np.zeros(f0_length, dtype=np.float64)
        for section in range(boundaries.shape[0] // 2):
            start, end = boundaries[section * 2], boundaries[section * 2 + 1]
            filtered = self._filtering_f0(channels[section], start, end)
            smoothed[start - lag : end + 1 - lag] = filtered[start : end + 1]
        return smoothed

    def _samples_for_harvest(self, fs, x_length, frame_period):
        return int(1000.0 * x_length / fs / frame_period) + 1

    def _waveform_and_spectrum(self, x, y_length, fft_size, decimation_ratio):
        y = np.zeros(fft_size, dtype=np.float64)
        if decimation_ratio == 1:
            y[: x.shape[0]] = x
        else:
            # The decimated waveform is noisy at both ends, so the input is extended first.
            lag = int(math.ceil(140.0 / decimation_ratio) * decimation_ratio)
            extended = np.empty(x.shape[0] + lag * 2, dtype=np.float64)
            extended[:lag] = x[0]
            extended[lag : lag + x.shape[0]] = x
            extended[lag + x.shape[0] :] = x[-1]
            decimated = self._decimate(extended, decimation_ratio)
            y[:y_length] = decimated[lag // decimation_ratio : lag // decimation_ratio + y_length]

        y[:y_length] -= y[:y_length].mean()
        y[y_length:] = 0.0
        return y, np.fft.rfft(y)

    def _harvest_general_body(self, x, fs, frame_period, channels_in_octave, speed):
        adjusted_f0_floor = self.f0_floor * 0.9
        adjusted_f0_ceil = self.f0_ceil * 1.1
        number_of_channels = 1 + int(math.log(adjusted_f0_ceil / adjusted_f0_floor) / self._LOG2 * channels_in_octave)
        boundary_f0_list = adjusted_f0_floor * np.power(
            2.0, np.arange(1, number_of_channels + 1, dtype=np.float64) / channels_in_octave
        )

        x_length = x.shape[0]
        decimation_ratio = max(min(speed, 12), 1)
        y_length = int(math.ceil(x_length / decimation_ratio))
        actual_fs = fs / decimation_ratio
        fft_size = self._suitable_fft_size(y_length + 5 + 2 * int(2.0 * actual_fs / boundary_f0_list[0]))

        y, y_spectrum = self._waveform_and_spectrum(x, y_length, fft_size, decimation_ratio)

        f0_length = self._samples_for_harvest(fs, x_length, frame_period)
        temporal_positions = np.arange(f0_length, dtype=np.float64) * frame_period / 1000.0

        overlap_parameter = 7
        max_candidates = self._matlab_round(number_of_channels / 10.0) * overlap_parameter

        raw = self._raw_f0_candidates(
            boundary_f0_list, actual_fs, y_length, temporal_positions, y_spectrum, fft_size
        )
        candidates, detected = self._detect_official_f0_candidates(raw, max_candidates)
        candidates = self._overlap_f0_candidates(candidates, detected)
        number_of_candidates = detected * overlap_parameter

        candidates = candidates[:, :number_of_candidates]
        candidates, scores = self._refine_f0_candidates(
            y[:y_length], actual_fs, temporal_positions, candidates
        )
        candidates, scores = self._remove_unreliable_candidates(candidates, scores)

        contour = self._fix_f0_contour(candidates, scores)
        return temporal_positions, self._smooth_f0_contour(contour)

    def _fix_f0_stonemask(self, main_spectrum, diff_spectrum, fft_size, fs, f0s, number_of_harmonics):
        # Only the bins the harmonics land on are read, so the gather comes before the arithmetic.
        harmonics = np.arange(1, number_of_harmonics + 1, dtype=np.float64)
        index = self._matlab_round_array(f0s[:, None] * fft_size / fs * harmonics[None, :])
        np.clip(index, 0, fft_size // 2, out=index)

        rows = np.arange(f0s.shape[0], dtype=np.int64)[:, None]
        main = main_spectrum[rows, index]
        diff = diff_spectrum[rows, index]
        numerator_i = main.real * diff.imag - main.imag * diff.real
        power = main.real**2 + main.imag**2
        safe_power = np.where(power == 0.0, 1.0, power)
        instantaneous = np.where(
            power == 0.0,
            0.0,
            index * fs / fft_size + numerator_i / safe_power * fs / 2.0 / self._PI,
        )
        amplitude = np.sqrt(power)
        numerator = np.sum(amplitude * instantaneous, axis=-1)
        denominator = np.sum(amplitude * harmonics[None, :], axis=-1)
        return numerator / (denominator + self._SAFE_GUARD_MINIMUM)

    def _stonemask_batch(self, x, fs, positions, f0s, half_window_length, fft_size):
        width = half_window_length * 2 + 1
        window_length_in_time = width / fs

        offsets = np.arange(width, dtype=np.int64)
        base_time = (offsets[None, :] - half_window_length) / fs
        index_raw = self._matlab_round_array((positions[:, None] + base_time) * fs)

        main_window = np.zeros((f0s.shape[0], width + 2), dtype=np.float64)
        elapsed = (index_raw - 1.0) / fs - positions[:, None]
        window = main_window[:, 1 : width + 1]
        np.cos(2.0 * self._PI * elapsed / window_length_in_time, out=window)
        window *= 0.5
        window += 0.42
        window += 0.08 * np.cos(4.0 * self._PI * elapsed / window_length_in_time)
        diff_window = -(main_window[:, 2:] - main_window[:, :width]) / 2.0

        samples = x[np.clip(index_raw - 1, 0, x.shape[0] - 1)]
        padded = np.zeros((f0s.shape[0], fft_size), dtype=np.float64)
        np.multiply(samples, window, out=padded[:, :width])
        main_spectrum = np.fft.rfft(padded, axis=-1)
        np.multiply(samples, diff_window, out=padded[:, :width])
        diff_spectrum = np.fft.rfft(padded, axis=-1)

        tentative = self._fix_f0_stonemask(main_spectrum, diff_spectrum, fft_size, fs, f0s, 2)
        accepted = (tentative > 0.0) & (tentative <= f0s * 2)
        refined = np.zeros_like(f0s)
        if accepted.any():
            rows = np.flatnonzero(accepted)
            refined[rows] = self._fix_f0_stonemask(
                main_spectrum[rows], diff_spectrum[rows], fft_size, fs, tentative[rows], 6
            )

        # A correction beyond twenty percent is rejected in favour of the initial estimate.
        return np.where(np.abs(refined - f0s) > f0s * 0.2, f0s, refined)

    def _spectrum_for_estimation(self, x, y_length, actual_fs, fft_size, decimation_ratio):
        y = np.zeros(fft_size, dtype=np.float64)
        if decimation_ratio != 1:
            decimated = self._decimate(x, decimation_ratio)
            y[: decimated.shape[0]] = decimated
        else:
            y[: x.shape[0]] = x

        y[:y_length] -= y[:y_length].mean()
        y[y_length:] = 0.0
        spectrum = np.fft.rfft(y)

        cutoff_in_sample = self._matlab_round(actual_fs / self._CUT_OFF)
        return spectrum * _low_cut_filter_spectrum(cutoff_in_sample * 2 + 1, fft_size)

    def _dio_filtered_signal(self, half_average_length, fft_size, spectrum, y_length):
        """Convolves the spectrum with a Nuttall low-pass whose cutoff follows its own length."""
        filter_spectrum = _low_pass_filter_spectrum(half_average_length, fft_size)
        signal = np.fft.irfft(spectrum * filter_spectrum, n=fft_size) * fft_size
        index_bias = half_average_length * 2
        return signal[index_bias : index_bias + y_length]

    def _dio_f0_candidate_contour(self, crossings, boundary_f0, temporal_positions):
        f0_length = temporal_positions.shape[0]
        if any(locations.shape[0] < 3 for locations, _ in crossings):
            return np.zeros(f0_length, dtype=np.float64), np.full(f0_length, self._MAXIMUM_VALUE)

        interpolated = np.stack(
            [self._interp1(locations, intervals, temporal_positions) for locations, intervals in crossings]
        )
        candidate = interpolated.mean(axis=0)
        score = np.sqrt(((interpolated - candidate) ** 2).sum(axis=0) / 3.0)

        rejected = (
            (candidate > boundary_f0)
            | (candidate < boundary_f0 / 2.0)
            | (candidate > self.f0_ceil)
            | (candidate < self.f0_floor)
        )
        candidate[rejected] = 0.0
        score[rejected] = self._MAXIMUM_VALUE
        return candidate, score

    def _dio_candidates_and_scores(
        self,
        boundary_f0_list, actual_fs, y_length, temporal_positions, spectrum, fft_size
    ):
        bands = boundary_f0_list.shape[0]
        candidates = np.empty((bands, temporal_positions.shape[0]), dtype=np.float64)
        scores = np.empty_like(candidates)
        for band, boundary_f0 in enumerate(boundary_f0_list):
            signal = self._dio_filtered_signal(
                self._matlab_round(actual_fs / boundary_f0 / 2.0), fft_size, spectrum, y_length
            )
            crossings = self._four_zero_crossing_intervals(signal, y_length, actual_fs)
            candidate, score = self._dio_f0_candidate_contour(
                crossings, boundary_f0, temporal_positions
            )
            candidates[band] = candidate
            scores[band] = score / (candidate + self._SAFE_GUARD_MINIMUM)
        return candidates, scores

    def _dio_best_f0_contour(self, candidates, scores):
        return candidates[np.argmin(scores, axis=0), np.arange(candidates.shape[1])]

    def _dio_fix_step_1(self, best_f0_contour, voice_range_minimum, allowed_range):
        f0_length = best_f0_contour.shape[0]
        f0_base = np.zeros(f0_length, dtype=np.float64)
        f0_base[voice_range_minimum : f0_length - voice_range_minimum] = best_f0_contour[
            voice_range_minimum : f0_length - voice_range_minimum
        ]

        f0_step1 = np.zeros(f0_length, dtype=np.float64)
        body = f0_base[voice_range_minimum:]
        previous = f0_base[voice_range_minimum - 1 : -1]
        within = np.abs((body - previous) / (self._SAFE_GUARD_MINIMUM + body)) < allowed_range
        f0_step1[voice_range_minimum:] = np.where(within, body, 0.0)
        return f0_step1

    def _dio_fix_step_2(self, f0_step1, voice_range_minimum):
        f0_step2 = f0_step1.copy()
        centre = (voice_range_minimum - 1) // 2
        f0_length = f0_step1.shape[0]
        if f0_length - centre <= centre:
            return f0_step2
        windows = np.lib.stride_tricks.sliding_window_view(f0_step1, voice_range_minimum)
        silent = (windows == 0.0).any(axis=-1)
        f0_step2[centre : f0_length - centre][silent] = 0.0
        return f0_step2

    def _voiced_section_edges(self, f0):
        onset = (f0[1:] != 0.0) & (f0[:-1] == 0.0)
        offset = (f0[1:] == 0.0) & (f0[:-1] != 0.0)
        return np.flatnonzero(onset) + 1, np.flatnonzero(offset)

    def _dio_select_best_f0(self, current_f0, past_f0, candidates, target_index, allowed_range):
        reference_f0 = (current_f0 * 3.0 - past_f0) / 2.0
        column = candidates[:, target_index]
        best_f0 = float(column[np.argmin(np.abs(reference_f0 - column))])
        if abs(1.0 - best_f0 / reference_f0) > allowed_range:
            return 0.0
        return best_f0

    def _dio_fix_step_3(self, f0_step2, candidates, allowed_range, offsets):
        f0_length = f0_step2.shape[0]
        f0_step3 = f0_step2.copy()
        for index, start in enumerate(offsets):
            limit = f0_length - 1 if index == offsets.shape[0] - 1 else offsets[index + 1]
            for j in range(start, limit):
                f0_step3[j + 1] = self._dio_select_best_f0(
                    f0_step3[j], f0_step3[j - 1], candidates, j + 1, allowed_range
                )
                if f0_step3[j + 1] == 0.0:
                    break
        return f0_step3

    def _dio_fix_step_4(self, f0_step3, candidates, allowed_range, onsets):
        f0_length = f0_step3.shape[0]
        # One trailing zero stands in for the read one past the contour the upstream loop makes.
        f0_step4 = np.zeros(f0_length + 1, dtype=np.float64)
        f0_step4[:f0_length] = f0_step3
        for index in range(onsets.shape[0] - 1, -1, -1):
            limit = 1 if index == 0 else onsets[index - 1]
            for j in range(onsets[index], limit, -1):
                f0_step4[j - 1] = self._dio_select_best_f0(
                    f0_step4[j], f0_step4[j + 1], candidates, j - 1, allowed_range
                )
                if f0_step4[j - 1] == 0.0:
                    break
        return f0_step4[:f0_length]

    def _dio_fix_f0_contour(self, frame_period, candidates, best_f0_contour, allowed_range):
        f0_length = best_f0_contour.shape[0]
        voice_range_minimum = int(0.5 + 1000.0 / frame_period / self.f0_floor) * 2 + 1
        if f0_length <= voice_range_minimum:
            return np.zeros(f0_length, dtype=np.float64)

        contour = self._dio_fix_step_1(best_f0_contour, voice_range_minimum, allowed_range)
        contour = self._dio_fix_step_2(contour, voice_range_minimum)
        onsets, offsets = self._voiced_section_edges(contour)
        contour = self._dio_fix_step_3(contour, candidates, allowed_range, offsets)
        return self._dio_fix_step_4(contour, candidates, allowed_range, onsets)


class CheapTrickEnvelope:
    r"""
    Constructs the CheapTrick spectral envelope estimator, exposing [`~CheapTrickEnvelope.envelope`].
    The analysis window follows the f0 it is given, so harmonic structure does not reach the envelope
    and the result depends on the vocal tract rather than the pitch it was excited at. Delegates to
    `pyworld` where that package is importable and runs the ported implementation otherwise; the two
    agree to within the infinitesimal noise CheapTrick adds by design.

    Args:
        sampling_rate (`int`):
            Rate of the waveforms passed to the estimator.
        fft_size (`int`, *optional*, defaults to 512):
            Transform length the envelope is reported over, giving `fft_size // 2 + 1` bins.
        q1 (`float`, *optional*, defaults to -0.15):
            Coefficient of the compensation lifter that restores the envelope after smoothing.
        prefer_pyworld (`bool`, *optional*, defaults to `True`):
            Whether to delegate to `pyworld` when it is installed. Pass `False` to run the ported
            implementation regardless.
    """

    _DEFAULT_F0 = 500.0
    _EPS = 2.220446049250313e-16

    def __init__(
        self,
        sampling_rate: int,
        fft_size: int = 512,
        q1: float = -0.15,
        prefer_pyworld: bool = True,
    ):
        self.sampling_rate = sampling_rate
        self.fft_size = fft_size
        self.q1 = q1
        self.pyworld = _load_pyworld() if prefer_pyworld else None

    @property
    def f0_floor(self) -> float:
        """Lowest f0 the window can accommodate at this transform length."""
        return 3.0 * self.sampling_rate / (self.fft_size - 3.0)

    def envelope(
        self, waveform: torch.Tensor, f0: torch.Tensor, temporal_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimates the spectral envelope at each frame.

        Args:
            waveform (`torch.Tensor`):
                Mono waveform, shaped `(samples,)`.
            f0 (`torch.Tensor`):
                The f0 contour, in Hz, zero where the frame is unvoiced.
            temporal_positions (`torch.Tensor`):
                Centre of each frame, in seconds.

        Returns:
            `torch.Tensor`: The envelope, shaped `(frames, fft_size // 2 + 1)`.
        """
        if self.pyworld is not None:
            spectrogram = self.pyworld.cheaptrick(
                waveform.double().cpu().numpy(),
                f0.double().cpu().numpy(),
                temporal_positions.double().cpu().numpy(),
                self.sampling_rate,
                fft_size=self.fft_size,
            )
            return torch.from_numpy(spectrogram).to(waveform.device)

        # An unvoiced frame has no period to adapt the window to, so it is analysed at a fixed rate.
        current = torch.where(f0 <= self.f0_floor, torch.full_like(f0, self._DEFAULT_F0), f0)
        windowed = self._windowed_waveform(waveform, current, temporal_positions)
        power = self._power_spectrum(windowed, current)
        power = self._linear_smoothing(power, current * 2.0 / 3.0)
        # A frame of digital silence carries no power at all, and the cepstral stage takes a
        # logarithm. WORLD guards this with a noise floor of the same magnitude; a fixed floor keeps
        # the result reproducible and leaves the frame flat, which is what silence should look like.
        return self._smoothing_with_recovery(power.clamp_min(self._EPS), current)

    @staticmethod
    def _matlab_round(values: torch.Tensor) -> torch.Tensor:
        return torch.where(values >= 0, torch.floor(values + 0.5), torch.ceil(values - 0.5))

    @staticmethod
    def _interp1q(origin, shift: float, table: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """Linear interpolation of a uniformly sampled table, as WORLD's `interp1Q`."""
        scaled = (query - origin) / shift
        # A frame whose own band is shorter than the batch's widest runs past the table; those
        # columns are masked off by the caller, so clamping keeps the gather in range without
        # changing a kept value.
        base = scaled.to(torch.int64).clamp(0, table.shape[1] - 1)
        fraction = scaled - base
        delta = torch.diff(table, dim=1, append=table.new_zeros(table.shape[0], 1))
        delta[:, -1] = 0.0
        return table.gather(1, base) + delta.gather(1, base) * fraction

    def _windowed_waveform(
        self, waveform: torch.Tensor, f0: torch.Tensor, temporal_positions: torch.Tensor
    ) -> torch.Tensor:
        """Windows the waveform once per frame, over three periods of that frame's own f0."""
        half = self._matlab_round(1.5 * self.sampling_rate / f0).to(torch.int64)
        index = torch.arange(self.fft_size, device=f0.device)

        base = index[None, :] - half[:, None]
        inside = index[None, :] <= 2 * half[:, None]
        origin = self._matlab_round(temporal_positions * self.sampling_rate + 0.001).to(torch.int64)
        safe = (origin[:, None] + base).clamp(0, waveform.shape[0] - 1)

        position = base.to(f0.dtype) / 1.5 / self.sampling_rate
        window = torch.where(inside, 0.5 * torch.cos(math.pi * position * f0[:, None]) + 0.5, torch.zeros(()))
        window = window / window.pow(2).sum(1, keepdim=True).sqrt()

        windowed = waveform[safe] * window
        weight = windowed.sum(1, keepdim=True) / window.sum(1, keepdim=True)
        return torch.where(inside, windowed - window * weight, torch.zeros(()))

    def _power_spectrum(self, windowed: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        """Power spectrum of each windowed frame, with the band below f0 folded back onto itself."""
        spectrum = torch.fft.rfft(windowed, n=self.fft_size)
        power = spectrum.real.pow(2) + spectrum.imag.pow(2)

        limits = 2 + (f0 * self.fft_size / self.sampling_rate).to(torch.int64)
        width = int(limits.max())
        axis = torch.arange(width, device=f0.device, dtype=f0.dtype) * self.sampling_rate / self.fft_size
        replica = self._interp1q(
            f0[:, None],
            -float(self.sampling_rate) / self.fft_size,
            power,
            axis[None, :].expand(f0.shape[0], width),
        )
        keep = torch.arange(width, device=f0.device)[None, :] < (limits - 1)[:, None]
        corrected = power.clone()
        corrected[:, :width] = power[:, :width] + torch.where(keep, replica, torch.zeros(()))
        return corrected

    def _linear_smoothing(self, power: torch.Tensor, width: torch.Tensor) -> torch.Tensor:
        """Averages the power over a rectangular band of `width` Hz on the linear frequency axis."""
        half = self.fft_size // 2
        boundary = int(float(width.max()) * self.fft_size / self.sampling_rate) + 1

        # Reflected about bin zero on the left and about the Nyquist bin on the right, so the band
        # stays defined where it runs off either end.
        left = power[:, 1 : boundary + 1].flip(1)
        right = power[:, half - boundary : half + 1].flip(1)
        mirrored = torch.cat([left, power[:, :half], right], dim=1)

        segment = torch.cumsum(mirrored * self.sampling_rate / self.fft_size, dim=1)
        origin = -(boundary - 0.5) * self.sampling_rate / self.fft_size
        step = float(self.sampling_rate) / self.fft_size

        axis = torch.arange(half + 1, device=power.device, dtype=power.dtype) / self.fft_size * self.sampling_rate
        low = self._interp1q(origin, step, segment, axis[None, :] - width[:, None] / 2.0)
        high = self._interp1q(origin, step, segment, axis[None, :] + width[:, None] / 2.0)
        return (high - low) / width[:, None]

    def _smoothing_with_recovery(self, power: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        """Smooths the log spectrum in the cepstral domain and undoes the smoothing's own bias."""
        half = self.fft_size // 2
        quefrency = torch.arange(half + 1, device=f0.device, dtype=f0.dtype) / self.sampling_rate

        scaled = math.pi * f0[:, None] * quefrency[None, :]
        smoothing = torch.where(scaled == 0, torch.ones(()), torch.sin(scaled) / scaled)
        compensation = (1.0 - 2.0 * self.q1) + 2.0 * self.q1 * torch.cos(
            2.0 * math.pi * quefrency[None, :] * f0[:, None]
        )
        compensation[:, 0] = (1.0 - 2.0 * self.q1) + 2.0 * self.q1

        # The log spectrum is transformed as a real, even sequence of the full transform length, so it
        # is reflected rather than zero padded out to it.
        logarithm = power.log()
        mirrored = torch.cat([logarithm, logarithm[:, 1:half].flip(1)], dim=1)
        cepstrum = torch.fft.rfft(mirrored, n=self.fft_size).real
        return torch.fft.irfft(cepstrum * smoothing * compensation, n=self.fft_size)[:, : half + 1].exp()
