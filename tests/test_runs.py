import os

import pandas as pd
import pytest

from alderaan_viewer import runs

RUN = "20260101T000000"


def _touch(path, text=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def test_list_runs_requires_batch_catalog(tmp_path):
    _touch(runs.catalog_path(tmp_path, RUN))
    _touch(runs.catalog_path(tmp_path, "20250101T000000"))
    os.makedirs(tmp_path / "not_a_run")
    assert runs.list_runs(tmp_path) == [RUN, "20250101T000000"]


def test_target_ids_unique_in_catalog_order():
    catalog = pd.DataFrame({"koi_id": ["K00002", "K00001", "K00002"]})
    assert runs.target_ids(catalog) == ["K00002", "K00001"]


@pytest.mark.parametrize(
    "files, state",
    [
        (("results", "detrended"), runs.COMPLETE),
        (("detrended",), runs.INCOMPLETE),
        ((), runs.NOT_PROCESSED),
    ],
)
def test_target_state(tmp_path, files, state):
    paths = runs.target_paths(tmp_path, RUN, "K00001")
    for f in files:
        _touch(getattr(paths, f))
    assert runs.target_state(paths) == state


def test_results_without_detrended_fails_loudly(tmp_path):
    paths = runs.target_paths(tmp_path, RUN, "K00001")
    _touch(paths.results)
    with pytest.raises(ValueError, match="without"):
        runs.target_state(paths)


def test_log_status(tmp_path):
    paths = runs.target_paths(tmp_path, RUN, "K00001")
    assert runs.log_status(paths.log).startswith("no log file")

    _touch(paths.log, "INFO running\n")
    assert (
        runs.log_status(paths.log) == "no run summary in log (still running or killed)"
    )

    _touch(
        paths.log, "===== run summary =====\n  status:   unsupported (known limit): x\n"
    )
    assert runs.log_status(paths.log) == "unsupported (known limit): x"
