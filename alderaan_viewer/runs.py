"""
Run discovery, target paths, and target state

The only module that knows the ALDERAAN output layout:

    <data_dir>/<run_id>/alderaan_batch_<run_id>.csv
    <data_dir>/<run_id>/results/<koi>/<koi>_detrended.fits
    <data_dir>/<run_id>/results/<koi>/<koi>_results.fits
    <data_dir>/<run_id>/logs/alderaan_<koi>_<run_id>.log
"""

import os
from dataclasses import dataclass

import pandas as pd

COMPLETE = "complete"
INCOMPLETE = "incomplete"
NOT_PROCESSED = "not processed"


def run_dir(data_dir, run_id):
    return os.path.join(data_dir, run_id)


def catalog_path(data_dir, run_id):
    return os.path.join(run_dir(data_dir, run_id), f"alderaan_batch_{run_id}.csv")


def list_runs(data_dir):
    """Run IDs in data_dir (subdirectories with a batch catalog), newest first"""
    runs = [
        d for d in os.listdir(data_dir) if os.path.isfile(catalog_path(data_dir, d))
    ]
    return sorted(runs, reverse=True)


def read_catalog(data_dir, run_id):
    """Batch catalog copy: one row per planet"""
    return pd.read_csv(catalog_path(data_dir, run_id), index_col=0)


def target_ids(catalog):
    """Unique koi_id values, in catalog order"""
    return list(dict.fromkeys(catalog["koi_id"]))


@dataclass(frozen=True)
class TargetPaths:
    results: str
    detrended: str
    log: str


def target_paths(data_dir, run_id, koi):
    rdir = run_dir(data_dir, run_id)
    results_dir = os.path.join(rdir, "results", koi)
    return TargetPaths(
        results=os.path.join(results_dir, f"{koi}_results.fits"),
        detrended=os.path.join(results_dir, f"{koi}_detrended.fits"),
        log=os.path.join(rdir, "logs", f"alderaan_{koi}_{run_id}.log"),
    )


def target_state(paths):
    """
    COMPLETE (both FITS files), INCOMPLETE (detrended only), or NOT_PROCESSED

    Raises ValueError for a results file without its detrended file.
    """
    has_results = os.path.isfile(paths.results)
    has_detrended = os.path.isfile(paths.detrended)
    if has_results and has_detrended:
        return COMPLETE
    if has_results:
        raise ValueError(f"{paths.results} exists without {paths.detrended}")
    if has_detrended:
        return INCOMPLETE
    return NOT_PROCESSED


def log_status(log_path):
    """The `status:` line of the run summary in a target log, or why there is none"""
    if not os.path.isfile(log_path):
        return f"no log file ({os.path.basename(log_path)})"
    status = None
    with open(log_path) as f:
        for line in f:
            if line.strip().startswith("status:"):
                status = line.split("status:", 1)[1].strip()
    if status is None:
        return "no run summary in log (still running or killed)"
    return status
