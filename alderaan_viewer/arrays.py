"""
Display arrays derived from ALDERAAN results (no plotting, no file I/O)

Folding and O-C mirror the pipeline's matplotlib plots (Results.plot_folded_transit,
Results.plot_transit, TransitTimes.plot_omc) so the viewer shows the same numbers.
Science quantities (medians, models, transit times) come from alderaan, never
re-derived here.
"""

import numpy as np
from alderaan.constants import EXPTIME, KEPLER_SCIT
from alderaan.transit_model import exposure_offsets, transit_flux
from alderaan.utils.stats import bin_data


def bin_cadences(time, flux, visit, cadno, nbin):
    """
    Mean time and flux in groups of nbin consecutive cadence numbers, per visit

    Vectorised (one pass); used to reduce short cadence photometry for display.
    """
    key = visit.astype(np.int64) * (int(cadno.max()) + 1) + cadno // nbin
    _, inv = np.unique(key, return_inverse=True)
    count = np.bincount(inv)
    return np.bincount(inv, time) / count, np.bincount(inv, flux) / count


def planet_inputs(results):
    """
    Per-planet median params, median transit times, and median model flux

    Each is computed once per planet (every median_* call re-runs summary()).
    """
    npl = results.posteriors.npl
    params = [results.median_transit_params(n) for n in range(npl)]
    ttimes = [results.median_transit_times(n) for n in range(npl)]
    models = [results._planet_model(n)[0] for n in range(npl)]
    return params, ttimes, models


def folded_planet(results, n, params, ttimes, models):
    """
    Folded photometry of planet n with sibling models divided out

    As Results.plot_folded_transit; times in days from mid-transit, flux relative.
    """
    T14 = params[n][4]
    npl = len(models)
    sibling = 1 + np.sum([models[m] - 1 for m in range(npl) if m != n], axis=0)

    lc = results.litecurve
    flux = lc.flux / sibling

    # cadences near each transit (lc.time is sorted): a superset of `use`
    lo = np.searchsorted(lc.time, ttimes[n] - 3 * T14)
    hi = np.searchsorted(lc.time, ttimes[n] + 3 * T14)

    t, f, m, om = [], [], [], []
    n_transit = 0
    for tc, i, j in zip(ttimes[n], lo, hi, strict=True):
        use = np.abs(lc.time[i:j] - tc) / T14 < 1.5
        n_transit += int(np.any(use))
        t.append(lc.time[i:j][use] - tc)
        f.append(flux[i:j][use])
        m.append(models[n][i:j][use])
        om.append(lc.obsmode[i:j][use])
    t, f, m, om = (np.hstack(x) for x in (t, f, m, om))

    supersample = results.posteriors.supersample
    t_mod = np.arange(t.min(), t.max(), KEPLER_SCIT)
    f_mod = transit_flux(t_mod, *params[n])
    f_mod_integrated = [
        transit_flux(
            t_mod,
            *params[n],
            exptime_offsets=exposure_offsets(EXPTIME[o], supersample[o]),
        )
        for o in np.unique(om)
        if supersample[o] > 1
    ]

    resid = f - m
    t_bin, f_bin, f_bin_err = bin_data(t, f, T14 / 11)
    _, r_bin, r_bin_err = bin_data(t, resid, T14 / 11)
    empty = np.isnan(f_bin)

    return {
        "time": t,
        "flux": f,
        "resid": resid,
        "t_bin": t_bin,
        "f_bin": f_bin,
        "f_bin_err": f_bin_err,
        "r_bin": r_bin,
        "r_bin_err": r_bin_err,
        "empty": empty,
        "f_empty": transit_flux(t_bin[empty], *params[n]),
        "t_mod": t_mod,
        "f_mod": f_mod,
        "f_mod_integrated": f_mod_integrated,
        "n_transit": n_transit,
        "period": params[n][1],
    }


def transits_with_data(time, ttimes, T14):
    """Positions in ttimes with photometry within 1.5 T14 (as Results.plot_transit)"""
    lo = np.searchsorted(time, ttimes - 1.5 * T14)
    hi = np.searchsorted(time, ttimes + 1.5 * T14)
    return np.flatnonzero(hi > lo)


def single_transit(results, n, i, params, ttimes):
    """
    Photometry (ppt) and median model (ppt) around transit position i of planet n

    As Results.plot_transit: model on a KEPLER_SCIT grid over the 1.5 T14 window,
    integrated over the exposure of the transit's obsmode as sampled.
    """
    lc = results.litecurve
    tc = ttimes[i]
    T14 = params[4]
    use = np.abs(lc.time - tc) / T14 < 1.5
    t_obs = lc.time[use]
    f_obs = (lc.flux[use] - 1.0) * 1000
    t_mod = np.arange(tc - 1.5 * T14, tc + 1.5 * T14, KEPLER_SCIT)
    om = lc.obsmode[use][0]
    offsets = exposure_offsets(EXPTIME[om], results.posteriors.supersample[om])
    f_mod = (transit_flux(t_mod - tc, *params, exptime_offsets=offsets) - 1.0) * 1000
    return {"tc": tc, "t_obs": t_obs, "f_obs": f_obs, "t_mod": t_mod, "f_mod": f_mod}


def omc(tt):
    """
    Observed and model O-C (minutes) about linear_ephemeris(), as TransitTimes.plot_omc

    'linear' models (not measured) put every transit on the model at error
    ttv_scatter.
    """
    ttime_obs, ttime_err = tt.ttime_obs, tt.ttime_err
    if tt.model_name == "linear":
        ttime_obs = tt.ttime_mod
        ttime_err = np.full(len(tt.index), tt.ttv_scatter)

    omc_obs = (ttime_obs - tt.linear_ephemeris()) * 24 * 60
    omc_mod = (tt.ttime_mod - tt.linear_ephemeris()) * 24 * 60
    obs = np.isfinite(ttime_obs)
    flag = ~tt.kept_ttv
    good = obs & ~flag

    yrange = max(
        5,
        1.1 * np.max(np.abs(omc_obs[good]), initial=0),
        1.1 * np.max(np.abs(omc_mod)),
    )
    if tt.model_name == "linear":
        yrange = max(yrange, 1.5 * tt.ttv_scatter * 24 * 60)

    resid = omc_obs[good] - omc_mod[good]
    chi2_red = None
    if tt.model_name != "linear" and tt.model_dof is not None:
        if np.sum(good) > tt.model_dof:
            chi2 = np.sum((resid / (ttime_err[good] * 24 * 60)) ** 2)
            chi2_red = chi2 / (np.sum(good) - tt.model_dof)

    return {
        "ttime_obs": ttime_obs,
        "err_min": ttime_err * 24 * 60,
        "omc_obs": omc_obs,
        "omc_mod": omc_mod,
        "good": good,
        "outlier": obs & flag,
        "yrange": yrange,
        "resid": resid,
        "chi2_red": chi2_red,
    }
