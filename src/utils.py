from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import scipy as sp


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
    signal: np.array, passband: Tuple[float, float] = (5, 15), fs: int = 125
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


def AggregateErrorMetric(pr_errors, confidence_est):
    """
    Computes an aggregate error metric based on confidence estimates.

    Computes the MAE at 90% availability.

    Args:
        pr_errors: a numpy array of errors between pulse rate estimates and corresponding
            reference heart rates.
        confidence_est: a numpy array of confidence estimates for each pulse rate
            error.

    Returns:
        the MAE at 90% availability
    """
    # Higher confidence means a better estimate. The best 90% of the estimates
    #    are above the 10th percentile confidence.
    percentile90_confidence = np.percentile(confidence_est, 10)

    # Find the errors of the best pulse rate estimates
    best_estimates = pr_errors[confidence_est >= percentile90_confidence]

    # Return the mean absolute error
    return np.mean(np.abs(best_estimates))


def Evaluate():
    """
    Top-level function evaluation function.

    Runs the pulse rate algorithm on the Troika dataset and returns an aggregate error metric.

    Returns:
        Pulse rate error on the Troika dataset. See AggregateErrorMetric.
    """
    # Retrieve dataset files
    data_fls, ref_fls = LoadTroikaDataset()
    errs, confs = [], []
    for data_fl, ref_fl in zip(data_fls, ref_fls):
        # Run the pulse rate algorithm on each trial in the dataset
        errors, confidence = RunPulseRateAlgorithm(data_fl, ref_fl)
        errs.append(errors)
        confs.append(confidence)
        # Compute aggregate error metric
    errs = np.hstack(errs)
    confs = np.hstack(confs)
    return AggregateErrorMetric(errs, confs)


def RunPulseRateAlgorithm(data_fl, ref_fl):
    # Load data using LoadTroikaDataFile
    ppg, accx, accy, accz = LoadTroikaDataFile(data_fl)

    # Compute pulse rate estimates and estimation confidence.

    # Return per-estimate mean absolute error and confidence as a 2-tuple of numpy arrays.
    errors, confidence = np.ones(100), np.ones(100)  # Dummy placeholders. Remove
    return errors, confidence
