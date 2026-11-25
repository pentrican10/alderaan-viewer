"""
Shared test fixtures

Fast tests build small synthetic run directories in tmp_path. Tests that read the
real sample output (sample_output/, gitignored) are @pytest.mark.slow and skip
when it is absent.
"""

import os
import shutil

import pytest

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "sample_output")
SAMPLE_RUN = "20261123T214548"


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: reads the real sample output; skipped by test-fast"
    )


@pytest.fixture
def sample_dir():
    if not os.path.isfile(
        os.path.join(SAMPLE_DIR, SAMPLE_RUN, f"alderaan_batch_{SAMPLE_RUN}.csv")
    ):
        pytest.skip(f"sample output for run {SAMPLE_RUN} not in {SAMPLE_DIR}")
    return SAMPLE_DIR


@pytest.fixture
def client(sample_dir, tmp_path):
    """Logged-in test client on a copy of the sample output (the app writes reviews)"""
    from alderaan_viewer.app import create_app

    shutil.copytree(sample_dir, tmp_path / "output")
    app = create_app(str(tmp_path / "output"))
    c = app.test_client()
    c.post("/login", data={"username": "tester"})
    return c
