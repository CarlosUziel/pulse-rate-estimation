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
    ppg: np.array,
    acc_x: np.array,
    acc_y: np.array,
    acc_z: np.array,
    fs: int = 125,
    top_k: int = 7,
) -> np.array:
    """Compute optimal PPG peaks from PPG and Accelerometer signals by avoiding
        spurious correlations.

    Args:
        ppg: PPG signal.
        acc_x: Accelerometer signal (X axis).
        acc_y: Accelerometer signal (Y axis).
        acc_z: Accelerometer signal (Z axis).
        fs: Signals sampling rate, in Hz.
        top_k: Select top K frequency candidates.

    Returns:
        Optimal PPG peaks.
    """
    # 1. Compute spectrograms for each signal
    (
        (ppg_spec, ppg_freqs, _, _),
        (acc_x_spec, acc_x_freqs, _, _),
        (acc_y_spec, acc_y_freqs, _, _),
        (acc_z_spec, acc_z_freqs, _, _),
    ) = [
        plt.specgram(signal, Fs=fs, NFFT=8 * fs, noverlap=6 * fs)
        for signal in (ppg, acc_x, acc_y, acc_z)
    ]
    plt.close("all")

    # 2. Compute best frequency estimate for each window
    ppg_peaks = []
    for ppg_win, acc_x_win, acc_y_win, acc_z_win in zip(
        ppg_spec.T, acc_x_spec.T, acc_y_spec.T, acc_z_spec.T
    ):
        # 2.1. Compute max frequency component of each accelerometer signal
        acc_x_max_freq = acc_x_freqs[np.argmax(acc_x_win)]
        acc_y_max_freq = acc_y_freqs[np.argmax(acc_y_win)]
        acc_z_max_freq = acc_z_freqs[np.argmax(acc_z_win)]

        # 2.2. Get top K PPG frequencies according to their magnitude
        ppg_freq_sorted = ppg_freqs[ppg_win.argsort()[::-1]][:top_k]

        # 2.3. Discard those frequencies present in the accelerometer
        ppg_freq_filt = ppg_freq_sorted[
            ~(
                (ppg_freq_sorted == acc_x_max_freq)
                | (ppg_freq_sorted == acc_y_max_freq)
                | (ppg_freq_sorted == acc_z_max_freq)
            )
        ]

        # 2.4. Select optimal PPG peak
        if len(ppg_peaks) == 0:
            ppg_peaks.append(ppg_freq_filt[0])
        else:
            # select closest to previous estimate
            ppg_peaks.append(
                ppg_freq_filt[np.abs(ppg_freq_filt - ppg_peaks[-1]).argmin()]
            )

    return np.array(ppg_peaks)


def compute_bpm_confidence(
    bps_estimates: np.array,
    ppg: np.array,
    fs: int = 125,
) -> np.array:
    """
    Compute BPM estimates confidence scores by summing up the frequency spectrum within
        a window around the pulse rate estimate and dividing it by the sum of the entire
        spectrum. Use first harmonic as window size.

    Args:
        bps_estimates: PPG peaks, which are the BPM estimates in seconds.
        ppg: PPG signal.
        fs: Signals sampling rate, in Hz.

    Returns:
        Confidence scores.
    """
    # 0. PPG Spectrogram
    ppg_spec, ppg_freqs, _, _ = plt.specgram(ppg, Fs=fs, NFFT=8 * fs, noverlap=6 * fs)
    plt.close("all")

    # 1. Compute confidence per estimated peak
    confs = []
    for bps_est, ppg_win in zip(bps_estimates, ppg_spec.T):
        # 1.1. Get frequencies around estimate (within window)
        lower_bound = bps_est - (bps_est * 2)
        upper_bound = bps_est + (bps_est * 2)
        conf_win = (lower_bound < ppg_freqs) & (ppg_freqs < upper_bound)

        # 1.2. Compute confidence
        confs.append(np.sum(ppg_win[conf_win]) / np.sum(ppg_win))

    return np.array(confs)


def run_pulse_rate_estimation(
    signal_file: Path,
    ref_file: Path,
    fs: int = 125,
    passband: Tuple[float, float] = (40, 240),
):
    """Estimate pulse rate from PPG and Accelerometer data.

    Args:
        signal_file: File containing all sensor singals information.
        ref_file: File containing ground truth pulse rate.
        fs: Signals sampling rate, in Hz.
        passband: Passband, in Hz, used for filtering the signals.
    """
    # 0. Setup
    ppg, acc_x, acc_y, acc_z = load_troika_file(signal_file)
    bpm_true = sp.io.loadmat(ref_file)["BPM0"][..., 0]
    passband = np.array(passband) / 60

    # 1. Pre-processing
    ppg_filt, acc_x_filt, acc_y_filt, acc_z_filt = [
        bandpass_filter(sig_data, passband, fs)
        for sig_data in (ppg, acc_x, acc_y, acc_z)
    ]

    # 2. Get optimal PPG peaks
    bps_est = get_optimal_ppg_peaks(ppg_filt, acc_x_filt, acc_y_filt, acc_z_filt)

    # 3. Compute estimated bpm
    bpm_est = bps_est * 60

    # 4. Compute error
    bpm_error = np.abs(bpm_est - bpm_true)

    # 5. Get confidence scores
    confidence_scores = compute_bpm_confidence(bps_est, ppg_filt, fs)

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
