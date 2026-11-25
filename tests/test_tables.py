import re

import pytest
from alderaan.data_models.results import Results

from alderaan_viewer import runs, tables
from tests.conftest import SAMPLE_RUN


def test_period_ratios_sorted_by_period():
    rows = [
        {"name": "b", "period": 7.0},
        {"name": "x", "not_modeled": True},
        {"name": "a", "period": 3.5},
        {"name": "c", "period": 14.0},
    ]
    out = tables.period_ratios(rows)
    assert [r["planets"] for r in out] == ["b / a", "c / b"]
    assert [r["ratio"] for r in out] == [2.0, 2.0]


def test_single_planet_has_no_ratios():
    assert tables.period_ratios([{"name": "a", "period": 3.5}]) == []


@pytest.mark.slow
def test_properties_match_pipeline_run_summary(sample_dir):
    """Planet table equals the 'posterior medians' block of the K00137 log"""
    paths = runs.target_paths(sample_dir, SAMPLE_RUN, "K00137")
    results = Results.from_fits(paths.results, paths.detrended)
    rows = tables.planet_properties(results, ["K00137.01", "K00137.02", "K00137.03"])

    with open(paths.log) as f:
        log = f.read()
    block = log.split("posterior medians:")[1].split("results:")[0]
    expected = re.findall(
        r"(K\d+\.\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", block
    )
    assert len(expected) == len(rows) == 3
    for row, (name, p, ror, b, t14, _model, good) in zip(rows, expected, strict=True):
        assert row["name"] == name
        assert f"{row['period']:.6f}" == p
        assert f"{row['ror']:.4f}" == ror
        assert f"{row['impact']:.3f}" == b
        assert f"{row['dur14_hr']:.3f}" == t14
        assert f"{row['n_good']}/{row['n_transits']}" == good
