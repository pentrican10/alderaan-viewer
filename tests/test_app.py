import json

import pytest

from alderaan_viewer.app import create_app
from tests.conftest import SAMPLE_RUN

API = f"/api/{SAMPLE_RUN}"
FIGURES = (
    "lightcurve",
    "folded",
    "transits/0",
    "omc",
    "corner/0?params=C0,C1,ROR,IMPACT,DUR14",
)


def test_login_required(tmp_path):
    c = create_app(str(tmp_path)).test_client()
    assert c.get("/home").status_code == 302
    assert c.get(f"{API}/K00001/status").status_code == 401


def test_missing_directory_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        create_app(str(tmp_path / "missing"))


@pytest.mark.slow
def test_home_lists_all_targets(client):
    html = client.get(f"/home?run={SAMPLE_RUN}").get_data(as_text=True)
    for koi, state in [
        ("K00001", "complete"),
        ("K00137", "complete"),
        ("K00099", "not processed"),
        ("K00971", "incomplete"),
    ]:
        assert f'data-koi-id="{koi}" data-state="{state}"' in html


@pytest.mark.slow
@pytest.mark.parametrize("koi", ["K00001", "K00137"])
def test_complete_target_routes(client, koi):
    for path in FIGURES:
        r = client.get(f"{API}/{koi}/{path}")
        assert r.status_code == 200, path
        assert b"bdata" in r.data, path  # numpy arrays sent as binary
    assert client.get(f"{API}/{koi}/tables").status_code == 200
    assert len(client.get(f"{API}/{koi}/lightcurve").data) < 2e6


@pytest.mark.slow
def test_planet_names_from_results_header(client):
    planets = client.get(f"{API}/K00137/status").get_json()["planets"]
    assert [p["name"] for p in planets] == ["K00137.03", "K00137.01", "K00137.02"]


@pytest.mark.slow
def test_single_transit_paging(client):
    meta = json.loads(client.get(f"{API}/K00137/transits/0?start=9").data)["layout"][
        "meta"
    ]
    assert meta["start"] == 9 and meta["count"] == 9 and meta["n_transits"] > 18
    assert client.get(f"{API}/K00137/transits/0?start=100000").status_code == 400


@pytest.mark.slow
@pytest.mark.parametrize(
    "koi, message",
    [
        ("K00099", "1 transits within litecurve span"),
        ("K00971", "no run summary in log"),
    ],
)
def test_incomplete_targets_explain_themselves(client, koi, message):
    assert message in client.get(f"{API}/{koi}/status").get_json()["message"]
    for path in FIGURES[1:]:
        r = client.get(f"{API}/{koi}/{path}")
        assert r.status_code == 409 and message in r.get_json()["error"], path


@pytest.mark.slow
def test_detrended_only_target_shows_lightcurve(client):
    assert client.get(f"{API}/K00971/lightcurve").status_code == 200


@pytest.mark.slow
def test_corner_ticks_and_default_ld_inset(client):
    """Default params add the LD_Q1 / LD_Q2 inset; three distinct ticks per axis"""
    default = f"{API}/K00137/corner/0?params=C0,C1,ROR,IMPACT,DUR14"
    other = f"{API}/K00137/corner/0?params=ROR,IMPACT,DUR14"
    for url, inset in ((default, True), (other, False)):
        layout = client.get(url).get_json()["layout"]
        labels = {a["text"] for a in layout["annotations"]}
        assert ("LD_Q1" in labels) == inset
        for key, axis in layout.items():
            if key.startswith(("xaxis", "yaxis")) and "ticktext" in axis:
                assert len(set(axis["ticktext"])) == 3, key


@pytest.mark.slow
def test_kepler_lightcurve_shows_every_quarter(client):
    """K00001 is missing quarters; every quarter still has a labeled band"""
    layout = client.get(f"{API}/K00001/lightcurve").get_json()["layout"]
    labels = [a["text"] for a in layout["annotations"]]
    assert labels == [str(q) for q in range(18)]
    assert len(layout["shapes"]) == 18


@pytest.mark.slow
def test_unknown_run_target_and_params(client):
    assert client.get("/api/nope/K00001/status").status_code == 404
    assert client.get(f"{API}/K99999/status").status_code == 404
    assert client.get(f"{API}/K00001/corner/0?params=BAD,ROR").status_code == 400


@pytest.mark.slow
def test_review_and_comments_roundtrip(client):
    r = client.post(f"{API}/K00001/review", json={"status": "Looks good"})
    assert r.status_code == 200
    assert client.post(f"{API}/K00001/review", json={"status": "ok"}).status_code == 400
    html = client.get(f"/home?run={SAMPLE_RUN}").get_data(as_text=True)
    assert 'class="koi-row looks-good"' in html

    client.post(f"{API}/K00001/comments", json={"comment": "<b>x</b>"})
    text = client.get(f"{API}/K00001/comments").get_data(as_text=True)
    assert "User: tester\nComment: <b>x</b>" in text
    client.put(f"{API}/K00001/comments", json={"content": "edited"})
    assert client.get(f"{API}/K00001/comments").get_data(as_text=True) == "edited"
