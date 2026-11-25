"""
Plotly figure builders (no file I/O)

Dense traces use Scattergl with numpy arrays, so fig.to_json() sends them as
binary (bdata). Each figure is one graph div, so one WebGL context.
"""

import numpy as np
import plotly.graph_objects as go
from alderaan.constants import KEPLER_LCIT, KEPLER_SCIT, SEED
from alderaan.utils.stats import weighted_percentile
from astropy.stats import mad_std
from plotly.subplots import make_subplots

from alderaan_viewer import arrays

COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2")
MAX_FOLDED_PTS = 20000
TRANSITS_PER_PAGE = 9
N_BINS_CORNER = 40
SAMPLED = ("C0", "C1", "ROR", "IMPACT", "DUR14")  # per planet, column <name>_<n>
CORNER_PARAMS = (*SAMPLED, "LD_Q1", "LD_Q2", "PERIOD", "EPOCH", "LD_U1", "LD_U2")


def _color(n):
    return COLORS[n % len(COLORS)]


def _layout(fig, height, title=None, shapes=(), notes=()):
    """
    Common layout; shapes and notes (plain dicts) are added in one update

    Plotly's add_hline / add_vline / add_vrect / add_annotation with row and col
    take about 10 ms per call, so figures build the dicts directly.
    """
    fig.update_layout(
        template="plotly_white",
        height=height,
        title=title,
        showlegend=False,
        margin={"l": 60, "r": 20, "t": 50, "b": 40},
    )
    if shapes:
        fig.update_layout(shapes=list(shapes))
    if notes:
        fig.update_layout(annotations=[*fig.layout.annotations, *notes])
    return fig


def _axis(row, col=1, cols=1):
    """Axis id suffix of subplot (row, col) in a make_subplots grid: '', '2', ..."""
    i = (row - 1) * cols + col
    return "" if i == 1 else str(i)


def _hline(y, line, ax=""):
    return {
        "type": "line",
        "line": line,
        "x0": 0,
        "x1": 1,
        "xref": f"x{ax} domain",
        "y0": y,
        "y1": y,
        "yref": f"y{ax}",
    }


def _vline(x, line, ax=""):
    return {
        "type": "line",
        "line": line,
        "x0": x,
        "x1": x,
        "xref": f"x{ax}",
        "y0": 0,
        "y1": 1,
        "yref": f"y{ax} domain",
    }


def _corner_note(text, row, corner, color="black"):
    x, xanchor = (0.01, "left") if "left" in corner else (0.99, "right")
    y, yanchor = (0.98, "top") if "upper" in corner else (0.02, "bottom")
    ax = _axis(row)
    return {
        "text": text,
        "x": x,
        "y": y,
        "xref": f"x{ax} domain",
        "yref": f"y{ax} domain",
        "xanchor": xanchor,
        "yanchor": yanchor,
        "showarrow": False,
        "bordercolor": color,
        "borderwidth": 1 if color != "black" else 0,
    }


def lightcurve(lc, ttimes=None):
    """
    Detrended flux vs time; ttimes (one array per planet) marked below the data

    Long cadence is drawn unbinned; short cadence is binned to the long cadence
    exposure within each visit.
    """
    fig = go.Figure()
    nbin = int(round(KEPLER_LCIT / KEPLER_SCIT))
    for om in np.unique(lc.obsmode):
        q = lc.obsmode == om
        time, flux = lc.time[q], lc.flux[q]
        name = om
        if om == "short cadence":
            time, flux = arrays.bin_cadences(time, flux, lc.visit[q], lc.cadno[q], nbin)
            name = f"short cadence, binned x{nbin}"
        fig.add_trace(
            go.Scattergl(
                x=time,
                y=flux.astype(np.float32),
                mode="markers",
                marker={"size": 2, "color": "black"},
                name=name,
            )
        )

    yrange = 1.3 * np.max(np.abs(1 - np.percentile(lc.flux, [0.1, 99.9])))
    ymin = 1 - yrange
    if ttimes is not None:
        ymin = 1 - yrange * (1.3 + 0.1 * len(ttimes))
        for n, tc in enumerate(ttimes):
            y = 1 - (0.9 - 0.1 * n) * (1 - ymin)
            fig.add_trace(
                go.Scatter(
                    x=tc,
                    y=np.full(len(tc), y),
                    mode="markers",
                    marker={"symbol": "triangle-up", "size": 7, "color": _color(n)},
                    name=f"planet {n}",
                )
            )

    shapes, notes = [], []
    for i, v in enumerate(np.unique(lc.visit)):
        t = lc.time[lc.visit == v]
        shapes.append(
            {
                "type": "rect",
                "fillcolor": "lightgrey" if i % 2 else "white",
                "opacity": 0.3,
                "line": {"width": 0},
                "layer": "below",
                "x0": t.min(),
                "x1": t.max(),
                "xref": "x",
                "y0": 0,
                "y1": 1,
                "yref": "y domain",
            }
        )
        notes.append(
            {
                "text": str(v),
                "x": t.min(),
                "xref": "x",
                "xanchor": "left",
                "y": 1,
                "yref": "y domain",
                "yanchor": "top",
                "showarrow": False,
            }
        )

    fig.update_yaxes(range=[ymin, 1 + yrange], title="Flux")
    fig.update_xaxes(range=[lc.time.min(), lc.time.max()], title="Time [BKJD]")
    return _layout(fig, 400, shapes=shapes, notes=notes)


def folded(results, names):
    """Folded transit and residuals (ppm) for every planet, two rows per planet"""
    npl = results.posteriors.npl
    params, ttimes, models = arrays.planet_inputs(results)
    fig = make_subplots(
        rows=2 * npl,
        cols=1,
        row_heights=[2, 1] * npl,
        vertical_spacing=0.25 / (2 * npl),
        subplot_titles=[t for n in range(npl) for t in (names[n], "")],
    )
    shapes, notes = [], []
    rng = np.random.default_rng(SEED)
    for n in range(npl):
        d = arrays.folded_planet(results, n, params, ttimes, models)
        top, bottom = 2 * n + 1, 2 * n + 2
        hrs = d["time"] * 24
        inds = rng.choice(len(hrs), size=min(MAX_FOLDED_PTS, len(hrs)), replace=False)
        inds.sort()

        raw = {"mode": "markers", "marker": {"size": 3, "color": "lightgrey"}}
        binned = {"mode": "markers", "marker": {"size": 6, "color": "black"}}
        empty = {
            "mode": "markers",
            "marker": {
                "size": 6,
                "color": "white",
                "line": {"color": "black", "width": 1},
            },
        }
        fig.add_trace(
            go.Scattergl(x=hrs[inds], y=((d["flux"][inds] - 1) * 1e6), **raw),
            row=top,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=d["t_bin"] * 24,
                y=(d["f_bin"] - 1) * 1e6,
                error_y={"array": d["f_bin_err"] * 1e6},
                **binned,
            ),
            row=top,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=d["t_bin"][d["empty"]] * 24, y=(d["f_empty"] - 1) * 1e6, **empty
            ),
            row=top,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=d["t_mod"] * 24,
                y=(d["f_mod"] - 1) * 1e6,
                mode="lines",
                line={"color": _color(n), "width": 3},
            ),
            row=top,
            col=1,
        )
        for f in d["f_mod_integrated"]:
            fig.add_trace(
                go.Scatter(
                    x=d["t_mod"] * 24,
                    y=(f - 1) * 1e6,
                    mode="lines",
                    line={"color": "black", "width": 2, "dash": "dash"},
                ),
                row=top,
                col=1,
            )

        fig.add_trace(
            go.Scattergl(x=hrs[inds], y=d["resid"][inds] * 1e6, **raw),
            row=bottom,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=d["t_bin"] * 24,
                y=d["r_bin"] * 1e6,
                error_y={"array": d["r_bin_err"] * 1e6},
                **binned,
            ),
            row=bottom,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=d["t_bin"][d["empty"]] * 24,
                y=np.zeros(int(d["empty"].sum())),
                **empty,
            ),
            row=bottom,
            col=1,
        )
        shapes.append(_hline(0, {"color": "black", "dash": "dash"}, _axis(bottom)))

        chi2_red = results.meta.get(f"CHI2RED{n}")
        if chi2_red is None:
            chi2_text, chi2_color = "reduced χ² = n/a (not in results file)", "black"
        else:
            chi2_text = f"reduced χ² = {chi2_red:.2f}"
            chi2_color = "red" if chi2_red > 2 else "black"
        rms = int(np.std(d["resid"]) * 1e6)
        notes += [
            _corner_note(f"N_transit = {d['n_transit']}", top, "upper left"),
            _corner_note(f"P = {d['period']:.2f} d", top, "upper right"),
            _corner_note(f"residual RMS = {rms} ppm", top, "lower right"),
            _corner_note(chi2_text, top, "lower left", chi2_color),
        ]

        xrange = [hrs.min(), hrs.max()]
        fig.update_xaxes(range=xrange, row=top, col=1)
        fig.update_xaxes(range=xrange, title="Time from mid-transit [hrs]", row=bottom)
        fig.update_yaxes(title="Flux [ppm]", row=top, col=1)
        fig.update_yaxes(title="Residuals", row=bottom, col=1)

    return _layout(fig, 450 * npl, shapes=shapes, notes=notes)


def single_transits(results, n, start, name):
    """
    Up to TRANSITS_PER_PAGE transits of planet n with photometry, from position start

    layout.meta = {start, count, n_transits}; positions count transits with data.
    """
    params = results.median_transit_params(n)
    ttimes = results.median_transit_times(n)
    tt = results.transit_times[n]
    with_data = arrays.transits_with_data(results.litecurve.time, ttimes, params[4])
    if not 0 <= start < len(with_data):
        raise ValueError(f"start {start} outside 0..{len(with_data) - 1}")
    page = with_data[start : start + TRANSITS_PER_PAGE]

    titles = []
    for i in page:
        flags = [
            label
            for label, ok in (
                ("poor coverage", tt.phot_coverage[i]),
                ("noisy", tt.phot_noise[i]),
                ("TTV outlier", tt.kept_ttv[i]),
            )
            if not ok
        ]
        titles.append(
            f"transit {tt.index[i]}" + (f" ({', '.join(flags)})" if flags else "")
        )

    fig = make_subplots(rows=3, cols=3, subplot_titles=titles)
    for k, i in enumerate(page):
        d = arrays.single_transit(results, n, i, params, ttimes)
        row, col = k // 3 + 1, k % 3 + 1
        fig.add_trace(
            go.Scatter(
                x=d["t_obs"],
                y=d["f_obs"],
                mode="markers",
                marker={"size": 4, "color": "lightgrey"},
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=d["t_mod"],
                y=d["f_mod"],
                mode="lines",
                line={"color": _color(n), "width": 2},
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(
            range=[d["tc"] - 1.55 * params[4], d["tc"] + 1.55 * params[4]],
            row=row,
            col=col,
        )
    fig.update_xaxes(title="Time [BKJD]", row=3)
    fig.update_yaxes(title="Flux [ppt]", col=1)
    _layout(fig, 750, f"{name}: P = {params[1]:.2f} d")
    fig.update_layout(
        meta={"start": start, "count": len(page), "n_transits": len(with_data)}
    )
    return fig


def omc(results, names):
    """O-C (minutes) and residuals for every planet, two rows per planet"""
    npl = results.posteriors.npl
    fig = make_subplots(
        rows=2 * npl,
        cols=1,
        row_heights=[2, 1] * npl,
        vertical_spacing=0.25 / (2 * npl),
        subplot_titles=[t for n in range(npl) for t in (names[n], "")],
    )
    shapes, notes = [], []
    for n, tt in enumerate(results.transit_times):
        d = arrays.omc(tt)
        top, bottom = 2 * n + 1, 2 * n + 2
        good, yr = d["good"], d["yrange"]
        t_good = d["ttime_obs"][good]

        fig.add_trace(
            go.Scatter(
                x=t_good,
                y=d["omc_obs"][good],
                error_y={"array": d["err_min"][good], "color": "lightgrey"},
                mode="markers",
                marker={"size": 6, "color": "grey"},
            ),
            row=top,
            col=1,
        )
        on = d["outlier"] & (np.abs(d["omc_obs"]) <= yr)
        fig.add_trace(
            go.Scatter(
                x=d["ttime_obs"][on],
                y=d["omc_obs"][on],
                mode="markers",
                marker={"symbol": "x", "size": 7, "color": "red"},
            ),
            row=top,
            col=1,
        )
        for sign, symbol in ((1, "triangle-up-open"), (-1, "triangle-down-open")):
            off = d["outlier"] & (sign * d["omc_obs"] > yr)
            fig.add_trace(
                go.Scatter(
                    x=d["ttime_obs"][off],
                    y=np.full(int(off.sum()), sign * yr),
                    mode="markers",
                    marker={"symbol": symbol, "size": 7, "color": "red"},
                ),
                row=top,
                col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=tt.ttime_mod,
                y=d["omc_mod"],
                mode="lines",
                line={"color": _color(n), "width": 2},
            ),
            row=top,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t_good,
                y=d["resid"],
                mode="markers",
                marker={"size": 5, "color": "grey"},
            ),
            row=bottom,
            col=1,
        )
        shapes.append(_hline(0, {"color": "black", "dash": "dash"}, _axis(bottom)))

        notes.append(_corner_note(f"P = {tt.period:.2f} d", top, "upper right"))
        rms = mad_std(d["resid"]) if len(d["resid"]) else np.nan
        notes.append(_corner_note(f"residual RMS = {rms:.1f} min", top, "lower right"))
        if tt.model_name is not None:
            notes.append(_corner_note(f"model = {tt.model_name}", top, "upper left"))
        if d["chi2_red"] is not None:
            chi2_text = f"reduced χ² = {d['chi2_red']:.2f}"
            notes.append(_corner_note(chi2_text, top, "lower left"))

        fig.update_yaxes(range=[-yr, yr], title="O-C [min]", row=top, col=1)
        fig.update_yaxes(title="Residuals [min]", row=bottom, col=1)
        fig.update_xaxes(title="Time [BKJD]", row=bottom, col=1)

    return _layout(fig, 400 * npl, shapes=shapes, notes=notes)


def _corner_samples(results, n, params):
    post = results.posteriors
    physical = post.planet_samples(n)
    out = {}
    for p in params:
        if p in SAMPLED:
            x = post.samples[f"{p}_{n}"].to_numpy()
        elif p in ("LD_Q1", "LD_Q2"):
            x = post.samples[p].to_numpy()
        else:
            x = physical[p].to_numpy()
        out[p] = x * 24 if p == "DUR14" else x
    return out


def _mass_levels(h, fractions):
    """Histogram values enclosing each fraction of the total mass"""
    flat = np.sort(h.ravel())[::-1]
    cum = np.cumsum(flat) / flat.sum()
    return [flat[min(np.searchsorted(cum, f), len(flat) - 1)] for f in fractions]


def corner(results, n, params, name):
    """
    Weighted corner plot of planet n: 1D histograms and 39% / 86% mass contours

    Posterior weights are applied once. DUR14 is in hours.
    """
    unknown = [p for p in params if p not in CORNER_PARAMS]
    if unknown or len(params) < 2:
        raise ValueError(f"need >= 2 params from {CORNER_PARAMS}, got {params}")

    w = results.posteriors.weights()
    x = _corner_samples(results, n, params)
    # one call per param (one sort): axis range, then 15.9 / 50 / 84.1 lines
    pct = {p: weighted_percentile(x[p], [0.5, 99.5, 15.9, 50, 84.1], w) for p in params}
    ranges = {p: pct[p][:2] for p in params}
    labels = {p: "DUR14 [hr]" if p == "DUR14" else p for p in params}

    k = len(params)
    fig = make_subplots(rows=k, cols=k, horizontal_spacing=0.02, vertical_spacing=0.02)
    # axis properties and shapes are collected and applied in one layout update
    axes, shapes = {}, []

    def axis(xy, row, col, **props):
        axes.setdefault(f"{xy}axis{_axis(row, col, k)}", {}).update(props)

    for i, pi in enumerate(params):
        for j, pj in enumerate(params):
            row, col = i + 1, j + 1
            ax = _axis(row, col, k)
            if j > i:
                axis("x", row, col, visible=False)
                axis("y", row, col, visible=False)
                continue
            if i == j:
                h, edges = np.histogram(
                    x[pi], bins=N_BINS_CORNER, range=ranges[pi], weights=w
                )
                fig.add_trace(
                    go.Scatter(
                        x=0.5 * (edges[1:] + edges[:-1]),
                        y=h,
                        mode="lines",
                        line={"shape": "hvh", "color": "black"},
                    ),
                    row=row,
                    col=col,
                )
                for q in pct[pi][2:]:
                    shapes.append(_vline(q, {"color": _color(n), "dash": "dot"}, ax))
                axis("y", row, col, showticklabels=False)
            else:
                h, xe, ye = np.histogram2d(
                    x[pj],
                    x[pi],
                    bins=N_BINS_CORNER,
                    range=[ranges[pj], ranges[pi]],
                    weights=w,
                )
                lo, hi = _mass_levels(h, [0.86, 0.39])
                if hi > lo:
                    fig.add_trace(
                        go.Contour(
                            x=0.5 * (xe[1:] + xe[:-1]),
                            y=0.5 * (ye[1:] + ye[:-1]),
                            z=h.T,
                            contours={
                                "coloring": "lines",
                                "start": lo,
                                "end": hi,
                                "size": hi - lo,
                            },
                            line={"color": _color(n)},
                            showscale=False,
                        ),
                        row=row,
                        col=col,
                    )
                if pj == "IMPACT":
                    shapes.append(_vline(1, {"color": "grey", "dash": "dash"}, ax))
                if pi == "IMPACT":
                    shapes.append(_hline(1, {"color": "grey", "dash": "dash"}, ax))
                axis("y", row, col, range=ranges[pi])
            axis("x", row, col, range=ranges[pj])
            if i == k - 1:
                axis("x", row, col, title={"text": labels[pj]})
            else:
                axis("x", row, col, showticklabels=False)
            if j == 0 and i > 0:
                axis("y", row, col, title={"text": labels[pi]})
            elif j > 0:
                axis("y", row, col, showticklabels=False)

    fig.update_layout(axes)
    return _layout(fig, 150 * k + 100, name, shapes=shapes)
