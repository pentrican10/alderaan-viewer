import numpy as np
from alderaan.data_models.transit_times import TransitTimes

from alderaan_viewer import arrays


def test_bin_cadences_groups_within_visits():
    cadno = np.arange(10)
    visit = np.array([0] * 5 + [1] * 5)
    time = cadno.astype(float)
    flux = np.ones(10)
    t, f = arrays.bin_cadences(time, flux, visit, cadno, 4)
    # visit 0: cadences 0-3, 4; visit 1: 5-7, 8-9
    np.testing.assert_allclose(t, [1.5, 4.0, 6.0, 8.5])
    np.testing.assert_allclose(f, 1.0)


def test_transits_with_data():
    time = np.arange(0.0, 10.0, 0.01)
    ttimes = np.array([-5.0, 2.0, 5.0, 20.0])
    np.testing.assert_array_equal(arrays.transits_with_data(time, ttimes, 0.1), [1, 2])


def _tt(model_name="poly_1"):
    index = np.arange(6)
    ttv = np.array([0, 1, -1, 0, 1, 30]) / 1440
    return TransitTimes(
        "Kepler",
        "K00001",
        index,
        10.0 + 3.0 * index + ttv,
        ttime_err=np.full(6, 1 / 1440),
        ttime_mod=10.0 + 3.0 * index,
        kept_ttv=np.array([True] * 5 + [False]),
        model_name=model_name,
        model_dof=2,
        ttv_scatter=1 / 1440,
    )


def test_omc_about_linear_ephemeris():
    tt = _tt()
    d = arrays.omc(tt)
    np.testing.assert_allclose(
        d["omc_obs"], (tt.ttime_obs - tt.linear_ephemeris()) * 1440
    )
    np.testing.assert_array_equal(d["good"], [True] * 5 + [False])
    np.testing.assert_array_equal(d["outlier"], [False] * 5 + [True])
    assert d["chi2_red"] is not None


def test_omc_linear_model_puts_transits_on_model():
    d = arrays.omc(_tt("linear"))
    np.testing.assert_allclose(d["omc_obs"], d["omc_mod"])
    assert d["chi2_red"] is None
