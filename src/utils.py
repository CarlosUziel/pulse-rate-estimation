from copy import deepcopy
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt


def get_troika_files(data_dir: Path) -> Tuple[Iterable[Path], Iterable[Path]]:
    """
    Retrieve the .mat filenames for the troika dataset. Paths are returned in a orderly
        fasshion, so that the reference file at index `i` correspondons to the reference
        data at index `i`.

    Args:
        data_dir: Location of the troika dataset.

    Returns:
        Paths of the .mat files that contain signal data
        Paths of the .mat files that contain reference data
    """
    return (
        sorted(data_dir.glob("**/DATA_*.mat")),
        sorted(data_dir.glob("**/REF_*.mat")),
    )


def load_troika_file(file_path: Path) -> np.array:
    """
    Loads and extracts signals from a troika data file.

    Args:
        data_fl: (str) filepath to a troika .mat file.

    Returns:
        Numpy arrays for ppg, accx, accy, accz signals.
    """
    return sp.io.loadmat(file_path)["sig"][2:]


def bandpass_filter(
    signal: np.array, passband: Tuple[float, float] = (2 / 3, 4), fs: int = 125
) -> np.array:
    """Bandpass filter the signal bfor a given passband

    Args:
        signal: Signal data to filter.
        passband: Tuple of lower and upper bound frequencies (in Hz).
        fs: Sampling rate (in Hz).

    Returns:
        Filtered signal whose frequencies are within the passband.
    """
    b, a = sp.signal.butter(3, passband, btype="bandpass", fs=fs)
    return sp.signal.filtfilt(b, a, signal)


def get_optimal_ppg_peaks(
    acc_mag_specs: np.array,
    acc_mag_freqs: np.array,
    ppg_specs: np.array,
    ppg_freqs: np.array,
    bpm_min_diff: float = 5.0,
) -> Tuple[np.array, np.array]:
    """Compute optimal PPG peaks from PPG and Accelerometer signals by avoiding
        spurious correlations.

    Args:
        acc_mag_specs: Accelerometer magnitude signal spectrogram matrix.
        acc_mag_freqs: Accelerometer magnitude frequencies.
        ppg_specs: PPG signal spectrogram matrix.
        ppg_freqs: PPG frequencies.
        bpm_min_diff: Minimum acceptable difference between PPG and Accelerometer peak
            frequencies, in beats per minute (bpm).

    Returns:
        Optimal PPG peaks.
    """
    # 1. Get acc mag peaks
    acc_mag_max_freq = acc_mag_freqs[np.argmax(acc_mag_specs, axis=0)]

    # 2. Get ppg freqs for each window, sorted
    ppg_specs_freqs = ppg_freqs[ppg_specs.argsort(axis=0)[::-1]]

    # 3. Get absolute difference between frequencies
    ppg_acc_diffs = np.abs(ppg_specs_freqs - acc_mag_max_freq)

    # 4. Find best PPG peak alternative to acc mag peak
    peak_inds = np.argmax(ppg_acc_diffs > bpm_min_diff / 60, axis=0)
    optimal_ppg_peaks = ppg_specs_freqs[peak_inds, np.arange(len(peak_inds))]

    return optimal_ppg_peaks, ppg_specs_freqs


def compute_bpm_confidence(
    optimal_ppg_peaks: np.array,
    ppg_specs: np.array,
    ppg_specs_freqs: np.array,
    conf_window: float = 10,
) -> np.array:
    """
    Compute BPM estimates confidence scores by summing up the frequency spectrum within
        a window around the pulse rate estimate and dividing it by the sum of the entire
        spectrum.

    Args:
        optimal_ppg_peaks: PPG peaks, which are the BPM estimates in seconds.
        ppg_specs: PPG signal spectrogram matrix.
        ppg_specs_freqs: PPG signal frequencies for each window, sorted.
        conf_window: Size of the window in one direction used to compute peak power.

    Returns:
        Confidence scores.
    """
    # 1. Ger confidence  windows
    lower_bound = optimal_ppg_peaks - conf_window / 60
    upper_bound = optimal_ppg_peaks + conf_window / 60
    confidence_windows = (lower_bound < ppg_specs_freqs) & (
        ppg_specs_freqs < upper_bound
    )

    # 2. Compute energy around and outside of the PPG peak
    ppg_specs_windows = deepcopy(ppg_specs)
    ppg_specs_windows[confidence_windows] = 0

    # 3. Compute and return confidence scores
    return np.sum(ppg_specs_windows, axis=0) / np.sum(ppg_specs, axis=0)


def run_pulse_rate_estimation(
    signal_file: Path,
    ref_file: Path,
    fs: int = 125,
    passband: Tuple[float, float] = (40, 240),
    bpm_min_diff: float = 5,
    conf_window: float = 10,
):
    """Estimate pulse rate from PPG and Accelerometer data.

    Args:
        signal_file: File containing all sensor singals information.
        ref_file: File containing ground truth pulse rate.
        fs: Signals sampling rate, in Hz.
        passband: Passband, in Hz, used for filtering the signals.
        bpm_min_diff: Minimum acceptable difference between PPG and Accelerometer peak
            frequencies, in beats per minute (bpm).
    """
    # 0. Setup
    ppg, acc_x, acc_y, acc_z = load_troika_file(signal_file)
    acc_mag = np.sqrt(np.sum(np.square(np.vstack((acc_x, acc_y, acc_z))), axis=0))
    bpm_true = sp.io.loadmat(ref_file)["BPM0"][..., 0]
    passband = np.array(passband) / 60

    # 1. Pre-processing
    ppg_filt, acc_mag_filt = [
        bandpass_filter(sig_data, passband, fs) for sig_data in (ppg, acc_mag)
    ]

    # 2. Calculate spectrograms
    ppg_specs, ppg_freqs, _, _ = plt.specgram(
        ppg_filt, Fs=fs, NFFT=8 * fs, noverlap=6 * fs
    )
    plt.clf()
    acc_mag_specs, acc_mag_freqs, _, _ = plt.specgram(
        acc_mag_filt, Fs=fs, NFFT=8 * fs, noverlap=6 * fs
    )
    plt.clf()

    # 3. Get optimal PPG peaks
    optimal_ppg_peaks, ppg_specs_freqs = get_optimal_ppg_peaks(
        acc_mag_specs,
        acc_mag_freqs,
        ppg_specs,
        ppg_freqs,
        bpm_min_diff,
    )

    # 4. Compute estimated bpm
    bpm_est = optimal_ppg_peaks * 60

    # 5. Compute error
    bpm_error = np.abs(bpm_est - bpm_true)

    # 6. Get confidence scores
    confidence_scores = compute_bpm_confidence(
        optimal_ppg_peaks,
        ppg_specs,
        ppg_specs_freqs,
        conf_window,
    )

    return bpm_error, confidence_scores


def RunPulseRateAlgorithm(data_fl, ref_fl):
    return run_pulse_rate_estimation(data_fl, ref_fl)


def aggregate_error_metrics(pr_errors: np.array, confidence_est: np.array) -> float:
    """
    Computes an aggregate error metric based on confidence estimates.

    Computes the MAE at 90% availability.

    Args:
        pr_errors: a numpy array of errors between pulse rate estimates and corresponding
            reference heart rates.
        confidence_est: a numpy array of confidence estimates for each pulse rate
            error.

    Returns:
        The MAE at 90% availability
    """
    # Higher confidence means a better estimate. The best 90% of the estimates
    #    are above the 10th percentile confidence.
    percentile90_confidence = np.percentile(confidence_est, 10)

    # Find the errors of the best pulse rate estimates
    best_estimates = pr_errors[confidence_est >= percentile90_confidence]

    # Return the mean absolute error
    return np.mean(np.abs(best_estimates))


def evaluate():
    """
    Top-level function evaluation function.

    Runs the pulse rate algorithm on the Troika dataset and returns an aggregate error
        metric.

    Returns:
        Pulse rate error on the Troika dataset. See AggregateErrorMetric.
    """
    # 1. Retrieve dataset files
    signal_files, ref_files = get_troika_files(
        Path(__file__).resolve().parents[1].joinpath("data")
    )

    # 2. Compute error and confidence values for each sample
    errs, confs = [], []
    for signal_file, ref_file in zip(signal_files, ref_files):
        errors, confidence = run_pulse_rate_estimation(signal_file, ref_file)
        errs.append(errors)
        confs.append(confidence)

    # 3. Aggregate and return final error metric
    errs = np.hstack(errs)
    confs = np.hstack(confs)
    return aggregate_error_metrics(errs, confs)
