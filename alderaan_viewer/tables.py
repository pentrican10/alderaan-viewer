"""Planet properties and period ratios, as lists of row dicts"""

import numpy as np
from alderaan.utils.stats import weighted_percentile


def planet_names(results):
    """PLNAME{n} header cards, in pipeline planet order"""
    return [results.meta[f"PLNAME{n}"] for n in range(results.posteriors.npl)]


def planet_properties(results, catalog_names):
    """
    One row per modeled planet (pipeline order), then catalog planets not modeled

    Posterior medians, as in the pipeline run summary. n_good counts transits
    with good photometry; chi2_red is the CHI2RED header card.
    """
    median = results.summary()["median"]
    w = results.posteriors.weights()
    names = planet_names(results)
    rows = []
    for n, name in enumerate(names):
        period = results.planet_samples(n)["PERIOD"].to_numpy()
        p50 = weighted_percentile(period, 50, w=w)
        tt = results.transit_times[n]
        rows.append(
            {
                "name": name,
                "period": p50,
                "ror": median[f"ROR_{n}"],
                "impact": median[f"IMPACT_{n}"],
                "dur14_hr": median[f"DUR14_{n}"] * 24,
                "n_good": int(np.sum(tt.phot_coverage & tt.phot_noise)),
                "n_transits": len(tt.ttime_mod),
                "chi2_red": results.meta.get(f"CHI2RED{n}"),
            }
        )
    for name in catalog_names:
        if name not in names:
            rows.append({"name": name, "not_modeled": True})
    return rows


def period_ratios(rows):
    """Ratios of adjacent modeled planets, sorted by median period"""
    planets = sorted(
        (r for r in rows if not r.get("not_modeled")), key=lambda r: r["period"]
    )
    return [
        {
            "planets": f"{b['name']} / {a['name']}",
            "periods": f"{b['period']:.4f} / {a['period']:.4f}",
            "ratio": b["period"] / a["period"],
        }
        for a, b in zip(planets[:-1], planets[1:], strict=True)
    ]
