"""
Plotly figure builders (no file I/O)

Dense traces use Scattergl with numpy arrays, so fig.to_json() sends them as
binary (bdata). Each figure is one graph div, so one WebGL context.

Style follows the pipeline's matplotlib quicklook plots: data in light grey,
binned data in black, models in the planet color (matplotlib C{n}), outliers
in red. Plotly draws WebGL traces above SVG traces whatever the trace order, so
a figure that overlays a model on Scattergl data draws the model as Scattergl.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
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

# Kepler long cadence quarter spans (BKJD), from the llc TSTART / TSTOP headers
KEPLER_QUARTERS = {
    0: (120.5281, 130.2562),
    1: (131.5014, 164.9947),
    2: (169.5092, 258.4784),
    3: (260.2138, 349.5067),
    4: (352.3659, 442.2139),
    5: (443.4793, 538.1736),
    6: (539.4390, 629.3074),
    7: (630.1642, 719.5592),
    8: (735.3526, 802.3557),
    9: (808.5050, 905.9374),
    10: (906.8350, 1000.2795),
    11: (1001.1975, 1098.3366),
    12: (1099.3977, 1182.0326),
    13: (1182.7260, 1273.0677),
    14: (1274.1288, 1371.3332),
    15: (1373.4772, 1471.1475),
    16: (1472.0861, 1557.9700),
    17: (1559.2151, 1591.0124),
}

DATA_COLOR = "lightgrey"
RESIDUAL_TITLE = {"font": {"size": 16}}  # smaller than the other axis titles
ZERO_LINE = {"color": "black", "dash": "dash", "width": 2}
BINNED = {
    "mode": "markers",
    "marker": {"size": 8, "color": "black"},
    "error_y": {"color": "black", "thickness": 1.5, "width": 0},
}
EMPTY_BIN = {
    "mode": "markers",
    "marker": {"size": 8, "color": "white", "line": {"color": "black", "width": 1.5}},
}

# matplotlib look: framed axes, outside ticks, no grid
_axis_style = {
    "mirror": True,
    "ticks": "outside",
    "showgrid": False,
    "zeroline": False,
    "linecolor": "black",
    "tickcolor": "black",
    "tickfont": {"size": 14},
    "title": {"font": {"size": 20}},
    "automargin": True,
}
TEMPLATE = go.layout.Template(pio.templates["simple_white"])
TEMPLATE.layout.update(
    font={"size": 14, "color": "black"},
    title={"font": {"size": 22}, "x": 0.5, "xanchor": "center"},
    xaxis=_axis_style,
    yaxis=_axis_style,
    annotationdefaults={"bgcolor": "white", "font": {"size": 16}},
    hoverlabel={"font": {"size": 13}},
    shapedefaults={"opacity": 1},  # simple_white sets 0.3
)


def _shared_x(fig, top, bottom, xrange, title, last):
    """Data and residual rows share x (as sharex); x title on the last row only"""
    fig.update_xaxes(range=xrange, showticklabels=False, row=top, col=1)
    fig.update_xaxes(range=xrange, matches=f"x{_axis(top)}", row=bottom, col=1)
    if last:
        fig.update_xaxes(title=title, row=bottom, col=1)


def _color(n):
    return COLORS[n % len(COLORS)]


def _with_errors(style, err):
    """Copy of a marker style with y error bars err"""
    return {**style, "error_y": {**style["error_y"], "array": err}}


def _layout(fig, height, title=None, shapes=(), notes=()):
    """
    Common layout; shapes and notes (plain dicts) are added in one update

    Plotly's add_hline / add_vline / add_vrect / add_annotation with row and col
    take about 10 ms per call, so figures build the dicts directly.
    """
    fig.update_layout(
        template=TEMPLATE,
        height=height,
        title=title,
        showlegend=False,
        margin={"l": 80, "r": 30, "t": 60, "b": 60},
    )
    # subplot titles (the only annotations so far; make_subplots sets 16)
    fig.update_annotations(font={"size": 18})
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
    x, xanchor = (0.02, "left") if "left" in corner else (0.98, "right")
    y, yanchor = (0.96, "top") if "upper" in corner else (0.04, "bottom")
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
                marker={"size": 3, "color": "black"},
                name=name,
            )
        )

    # scaled as LiteCurve.plot, with half its headroom above flux = 1
    yrange = 1.3 * np.max(np.abs(1 - np.percentile(lc.flux, [0.1, 99.9])))
    ymin = 1 - yrange
    ymax = 1 + 0.5 * yrange
    if ttimes is not None:
        ymin = 1 - yrange * (1.3 + 0.1 * len(ttimes))
        for n, tc in enumerate(ttimes):
            y = 1 - (0.9 - 0.1 * n) * (1 - ymin)
            fig.add_trace(
                go.Scattergl(
                    x=tc,
                    y=np.full(len(tc), y),
                    mode="markers",
                    marker={"symbol": "triangle-up", "size": 10, "color": _color(n)},
                    name=f"planet {n}",
                )
            )

    # Kepler: every quarter, observed or not; other missions: the visits observed
    if lc.mission == "Kepler":
        spans = KEPLER_QUARTERS
    else:
        spans = {
            v: (lc.time[lc.visit == v].min(), lc.time[lc.visit == v].max())
            for v in np.unique(lc.visit)
        }

    shapes, notes = [], []
    for v, (t0, t1) in spans.items():
        shapes.append(
            {
                "type": "rect",
                "fillcolor": "lightgrey" if v % 2 else "white",
                "opacity": 0.3,
                "line": {"width": 0},
                "layer": "below",
                "x0": t0,
                "x1": t1,
                "xref": "x",
                "y0": 0,
                "y1": 1,
                "yref": "y domain",
            }
        )
        notes.append(
            {
                "text": str(v),
                "font": {"size": 12},
                "bgcolor": "rgba(0, 0, 0, 0)",
                "x": 0.5 * (t0 + t1),
                "xref": "x",
                "xanchor": "center",
                "y": 1,
                "yref": "y domain",
                "yanchor": "top",
                "showarrow": False,
            }
        )

    t_lo = min(lc.time.min(), min(t0 for t0, _ in spans.values()))
    t_hi = max(lc.time.max(), max(t1 for _, t1 in spans.values()))
    fig.update_yaxes(range=[ymin, ymax], title="Flux", tickformat=".3f")
    fig.update_xaxes(range=[t_lo, t_hi], title="Time [BKJD]")
    _layout(fig, 450, shapes=shapes, notes=notes)
    fig.update_layout(margin={"t": 30})
    return fig


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

        # every trace is Scattergl so trace order sets draw order: data, then
        # binned, then models on top
        raw = {"mode": "markers", "marker": {"size": 5, "color": DATA_COLOR}}
        flux_ppm = (d["flux"] - 1) * 1e6
        resid_ppm = d["resid"] * 1e6
        f_bin_ppm = (d["f_bin"] - 1) * 1e6
        r_bin_ppm = d["r_bin"] * 1e6
        fig.add_trace(
            go.Scattergl(x=hrs[inds], y=flux_ppm[inds], **raw), row=top, col=1
        )
        fig.add_trace(
            go.Scattergl(
                x=d["t_bin"] * 24,
                y=f_bin_ppm,
                **_with_errors(BINNED, d["f_bin_err"] * 1e6),
            ),
            row=top,
            col=1,
        )
        fig.add_trace(
            go.Scattergl(
                x=d["t_bin"][d["empty"]] * 24, y=(d["f_empty"] - 1) * 1e6, **EMPTY_BIN
            ),
            row=top,
            col=1,
        )
        fig.add_trace(
            go.Scattergl(
                x=d["t_mod"] * 24,
                y=(d["f_mod"] - 1) * 1e6,
                mode="lines",
                line={"color": _color(n), "width": 4},
            ),
            row=top,
            col=1,
        )
        for f in d["f_mod_integrated"]:
            fig.add_trace(
                go.Scattergl(
                    x=d["t_mod"] * 24,
                    y=(f - 1) * 1e6,
                    mode="lines",
                    line={"color": "black", "width": 2.5, "dash": "dash"},
                ),
                row=top,
                col=1,
            )

        fig.add_trace(
            go.Scattergl(x=hrs[inds], y=resid_ppm[inds], **raw), row=bottom, col=1
        )
        fig.add_trace(
            go.Scattergl(
                x=d["t_bin"] * 24,
                y=r_bin_ppm,
                **_with_errors(BINNED, d["r_bin_err"] * 1e6),
            ),
            row=bottom,
            col=1,
        )
        fig.add_trace(
            go.Scattergl(
                x=d["t_bin"][d["empty"]] * 24,
                y=np.zeros(int(d["empty"].sum())),
                **EMPTY_BIN,
            ),
            row=bottom,
            col=1,
        )
        shapes.append(_hline(0, ZERO_LINE, _axis(bottom)))

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

        # y scale from the bulk of the data, the binned points, and the model;
        # not from outliers. Headroom for the corner notes; residuals symmetric.
        lo = min(
            np.percentile(flux_ppm, 0.5),
            np.nanmin(f_bin_ppm),
            (d["f_mod"].min() - 1) * 1e6,
        )
        hi = max(np.percentile(flux_ppm, 99.5), np.nanmax(f_bin_ppm))
        span = hi - lo
        r = 1.2 * max(
            np.max(np.abs(np.percentile(resid_ppm, [0.5, 99.5]))),
            np.nanmax(np.abs(r_bin_ppm)),
        )

        xrange = [hrs.min(), hrs.max()]
        last = n == npl - 1
        _shared_x(fig, top, bottom, xrange, "Time from mid-transit [hrs]", last)
        fig.update_yaxes(
            range=[lo - 0.2 * span, hi + 0.3 * span], title="Flux [ppm]", row=top, col=1
        )
        fig.update_yaxes(
            range=[-r, r],
            title={"text": "Residuals", **RESIDUAL_TITLE},
            row=bottom,
            col=1,
        )

    return _layout(fig, 450 * npl, shapes=shapes, notes=notes)


def single_transits(results, n, start, name):
    """
    Up to TRANSITS_PER_PAGE transits of planet n with photometry, from position start

    layout.meta = {start, count, n_transits, first, last}; positions count transits
    with data; first and last are the transit indices (as in the panel titles).
    """
    params = results.median_transit_params(n)
    ttimes = results.median_transit_times(n)
    tt = results.transit_times[n]
    with_data = arrays.transits_with_data(results.litecurve.time, ttimes, params[4])
    if not 0 <= start < len(with_data):
        raise ValueError(f"start {start} outside 0..{len(with_data) - 1}")
    page = with_data[start : start + TRANSITS_PER_PAGE]

    titles, boxes = [], []
    for k, i in enumerate(page):
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
        # pale fill, in the model color, on transits with bad photometry
        if not (tt.phot_coverage[i] and tt.phot_noise[i]):
            ax = _axis(k // 3 + 1, k % 3 + 1, cols=3)
            boxes.append(
                {
                    "type": "rect",
                    "xref": f"x{ax} domain",
                    "yref": f"y{ax} domain",
                    "x0": 0,
                    "x1": 1,
                    "y0": 0,
                    "y1": 1,
                    "fillcolor": _color(n),
                    "opacity": 0.15,
                    "line": {"width": 0},
                    "layer": "below",
                }
            )

    fig = make_subplots(rows=3, cols=3, subplot_titles=titles, vertical_spacing=0.12)
    f_all, m_min = [], 0.0
    for k, i in enumerate(page):
        d = arrays.single_transit(results, n, i, params, ttimes)
        row, col = k // 3 + 1, k % 3 + 1
        fig.add_trace(
            go.Scatter(
                x=d["t_obs"],
                y=d["f_obs"],
                mode="markers",
                marker={"size": 11, "color": "dimgrey"},
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=d["t_mod"],
                y=d["f_mod"],
                mode="lines",
                line={"color": _color(n), "width": 3},
            ),
            row=row,
            col=col,
        )
        # ticks at the window edges and mid-transit; decimals enough to differ
        tc, half = d["tc"], 1.5 * params[4]
        ticks = [tc - half, tc, tc + half]
        dec = 2
        while len({f"{t:.{dec}f}" for t in ticks}) < 3:
            dec += 1
        fig.update_xaxes(
            range=[tc - 1.55 * params[4], tc + 1.55 * params[4]],
            tickmode="array",
            tickvals=ticks,
            ticktext=[f"{t:.{dec}f}" for t in ticks],
            row=row,
            col=col,
        )
        f_all.append(d["f_obs"])
        m_min = min(m_min, d["f_mod"].min())

    # one y scale for the page (transits compare directly), from the bulk of
    # the photometry and the model depth, not from outliers
    lo, hi = np.percentile(np.hstack(f_all), [0.5, 99.5])
    lo = min(lo, m_min)
    pad = 0.1 * (hi - lo)
    fig.update_yaxes(range=[lo - pad, hi + pad])
    fig.update_xaxes(title="Time [BKJD]", row=3)
    fig.update_yaxes(title="Flux [ppt]", col=1)
    _layout(fig, 800, f"{name}: P = {params[1]:.2f} d", shapes=boxes)
    fig.update_layout(
        meta={
            "start": start,
            "count": len(page),
            "n_transits": len(with_data),
            "first": int(tt.index[page[0]]),
            "last": int(tt.index[page[-1]]),
        }
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

        # as TransitTimes.plot_omc: kept times in light grey, outliers in red
        # (off-scale ones as open triangles on the axis edge), model on top, and
        # the typical (median) uncertainty drawn once right of the data
        fig.add_trace(
            go.Scatter(
                x=t_good,
                y=d["omc_obs"][good],
                customdata=d["err_min"][good],
                hovertemplate="%{x:.3f}, %{y:.1f} ± %{customdata:.1f} min"
                "<extra></extra>",
                mode="markers",
                marker={"size": 8, "color": DATA_COLOR},
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
                marker={
                    "symbol": "x-thin",
                    "size": 9,
                    "line": {"color": "red", "width": 2},
                },
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
                    marker={"symbol": symbol, "size": 9, "color": "red"},
                    cliponaxis=False,
                ),
                row=top,
                col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=tt.ttime_mod,
                y=d["omc_mod"],
                mode="lines",
                line={"color": _color(n), "width": 3},
            ),
            row=top,
            col=1,
        )
        t0, t1 = tt.ttime_mod[0], tt.ttime_mod[-1]
        span = t1 - t0
        err = np.median(d["err_min"][good]) if np.any(good) else np.nan
        if np.isfinite(err):
            fig.add_trace(
                go.Scatter(
                    x=[t1 + 0.05 * span],
                    y=[0],
                    error_y={"array": [err], "color": "grey", "thickness": 2},
                    mode="markers",
                    marker={"size": 8, "color": "grey"},
                    hovertemplate=f"typical error ± {err:.1f} min<extra></extra>",
                ),
                row=top,
                col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=t_good,
                y=d["resid"],
                mode="markers",
                marker={"size": 8, "color": DATA_COLOR},
            ),
            row=bottom,
            col=1,
        )
        shapes.append(_hline(0, ZERO_LINE, _axis(bottom)))

        notes.append(_corner_note(f"P = {tt.period:.2f} d", top, "upper right"))
        rms = mad_std(d["resid"]) if len(d["resid"]) else np.nan
        notes.append(_corner_note(f"residual RMS = {rms:.1f} min", top, "lower right"))
        if tt.model_name is not None:
            notes.append(_corner_note(f"model = {tt.model_name}", top, "upper left"))
        if d["chi2_red"] is not None:
            chi2_text = f"reduced χ² = {d['chi2_red']:.2f}"
            notes.append(_corner_note(chi2_text, top, "lower left"))

        # residuals on the O-C scale, so the two panels compare directly
        xrange = [t0 - 0.02 * span, t1 + 0.08 * span]
        fig.update_yaxes(range=[-yr, yr], title="O-C [min]", row=top, col=1)
        fig.update_yaxes(
            range=[-yr, yr],
            title={"text": "Residuals", **RESIDUAL_TITLE},
            row=bottom,
            col=1,
        )
        _shared_x(fig, top, bottom, xrange, "Time [BKJD]", n == npl - 1)

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


def _three_ticks(lo, hi):
    """
    Three ticks in (lo, hi), labeled with 2 decimal places, or the fewest more
    that make the labels distinct; tick values equal their rounded labels
    """
    for d in range(2, 12):
        v = np.round(lo + np.array([0.2, 0.5, 0.8]) * (hi - lo), d)
        if len(np.unique(v)) == 3 and lo <= v[0] and v[-1] <= hi:
            break
    return {
        "tickmode": "array",
        "tickvals": v.tolist(),
        "ticktext": [f"{t:.{d}f}" for t in v],
    }


def corner(results, n, params, name):
    """
    Weighted corner plot of planet n: 1D histograms and 39% / 86% mass contours

    Posterior weights are applied once. DUR14 is in hours. Every panel is
    square. For the default parameters (SAMPLED), LD_Q1 and LD_Q2 are drawn as
    their own 2-parameter corner plot in the empty upper right.
    """
    unknown = [p for p in params if p not in CORNER_PARAMS]
    if unknown or len(params) < 2:
        raise ValueError(f"need >= 2 params from {CORNER_PARAMS}, got {params}")

    ld_inset = list(params) == list(SAMPLED)
    every = [*params, "LD_Q1", "LD_Q2"] if ld_inset else list(params)

    w = results.posteriors.weights()
    x = _corner_samples(results, n, every)
    # one call per param (one sort): axis range, then 15.9 / 50 / 84.1 lines
    pct = {p: weighted_percentile(x[p], [0.5, 99.5, 15.9, 50, 84.1], w) for p in every}
    ranges = {p: pct[p][:2] for p in every}
    labels = {p: "DUR14 [hr]" if p == "DUR14" else p for p in every}
    # C0 and C1 (small offsets about 0) keep plotly's automatic ticks
    ticks = {p: {} if p in ("C0", "C1") else _three_ticks(*ranges[p]) for p in every}

    k = len(params)
    fig = make_subplots(rows=k, cols=k, horizontal_spacing=0.02, vertical_spacing=0.02)
    # axis properties and shapes are collected and applied in one layout update
    axes, shapes, notes = {}, [], []
    c = _color(n)
    fill = f"rgba({int(c[1:3], 16)}, {int(c[3:5], 16)}, {int(c[5:7], 16)}, 0.5)"
    quantile_line = {"color": c, "dash": "dash", "width": 2}
    impact_line = {"color": "grey", "dash": "dash", "width": 1.5}

    # many small panels: fewer, smaller tick labels than the other figures
    def axis(xy, row, col, **props):
        default = {"nticks": 5, "tickfont": {"size": 12}}
        axes.setdefault(f"{xy}axis{_axis(row, col, k)}", default).update(props)

    def cell(row, col, px, py, show_x, show_y, xside="bottom", yside="left"):
        """Histogram of px (py None) or contours of py vs px, in a square panel"""
        ax = _axis(row, col, k)
        if py is None:
            h, edges = np.histogram(
                x[px], bins=N_BINS_CORNER, range=ranges[px], weights=w
            )
            fig.add_trace(
                go.Scatter(
                    x=0.5 * (edges[1:] + edges[:-1]),
                    y=h,
                    mode="lines",
                    line={"shape": "hvh", "color": c, "width": 1.5},
                    fill="tozeroy",
                    fillcolor=fill,
                ),
                row=row,
                col=col,
            )
            for q in pct[px][2:]:
                shapes.append(_vline(q, quantile_line, ax))
            yrange = [0, 1.05 * h.max()]
            axis("y", row, col, range=yrange)
        else:
            h, xe, ye = np.histogram2d(
                x[px],
                x[py],
                bins=N_BINS_CORNER,
                range=[ranges[px], ranges[py]],
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
                        # "lines" coloring takes the colorscale, not line.color
                        colorscale=[[0, c], [1, c]],
                        line={"width": 2},
                        showscale=False,
                    ),
                    row=row,
                    col=col,
                )
            if px == "IMPACT":
                shapes.append(_vline(1, impact_line, ax))
            if py == "IMPACT":
                shapes.append(_hline(1, impact_line, ax))
            yrange = ranges[py]
            axis("y", row, col, range=yrange, **ticks[py])
        axis("x", row, col, range=ranges[px], **ticks[px])

        # square at any figure width: y pixels per unit fixed to x pixels per
        # unit so that both spans have the same length
        xspan = ranges[px][1] - ranges[px][0]
        axis("x", row, col, visible=True, constrain="domain")
        axis(
            "y",
            row,
            col,
            visible=True,
            constrain="domain",
            scaleanchor=f"x{ax}",
            scaleratio=xspan / (yrange[1] - yrange[0]),
        )
        axis("x", row, col, showticklabels=show_x, side=xside)
        if show_x:
            axis("x", row, col, tickangle=-45)
        axis("y", row, col, showticklabels=show_y, side=yside)

    for i, pi in enumerate(params):
        for j, pj in enumerate(params):
            row, col = i + 1, j + 1
            if j > i:
                axis("x", row, col, visible=False)
                axis("y", row, col, visible=False)
                continue
            py = None if i == j else pi
            cell(row, col, pj, py, show_x=i == k - 1, show_y=j == 0 and i != j)

    # parameter labels at a fixed offset from the outer panels (bottom row,
    # first column), not as axis titles (those sit beyond the tick labels,
    # whose widths differ)
    for i, p in enumerate(params):
        notes.append(_param_label(labels[p], _axis(k, i + 1, k), "bottom"))
        if i > 0:
            notes.append(_param_label(labels[p], _axis(i + 1, 1, k), "left"))

    margin = {"l": 100, "b": 100}
    if ld_inset:
        # LD_Q1 / LD_Q2 corner flush with the empty upper right: the main
        # corner mirrored, 2D panel top right, ticks and labels on the outside
        q1, q2 = "LD_Q1", "LD_Q2"
        cell(1, k - 1, q1, None, show_x=True, show_y=False, xside="top")
        cell(1, k, q2, q1, show_x=True, show_y=True, xside="top", yside="right")
        cell(2, k, q2, None, show_x=False, show_y=False)
        notes.append(_param_label(labels[q1], _axis(1, k - 1, k), "top"))
        notes.append(_param_label(labels[q2], _axis(1, k, k), "top"))
        notes.append(_param_label(labels[q1], _axis(1, k, k), "right"))
        margin.update(t=160, r=100)

    fig.update_layout(axes)
    _layout(fig, 150 * k + 100, name, shapes=shapes, notes=notes)
    # title at the top of the figure, above the inset labels in the top margin
    fig.update_layout(
        margin=margin, title={"y": 0.99, "yref": "container", "yanchor": "top"}
    )
    return fig


def _param_label(text, ax, side):
    """Parameter name on one side (bottom, left, top, right) of panel ax, 70 px out"""
    horizontal = side in ("bottom", "top")
    x = {"left": 0, "right": 1}.get(side, 0.5)
    y = {"bottom": 0, "top": 1}.get(side, 0.5)
    return {
        "text": text,
        "font": {"size": 20},
        "bgcolor": "rgba(0, 0, 0, 0)",
        "x": x,
        "y": y,
        "xref": f"x{ax} domain",
        "yref": f"y{ax} domain",
        "xanchor": {"left": "right", "right": "left"}.get(side, "center"),
        "yanchor": {"bottom": "top", "top": "bottom"}.get(side, "middle"),
        "xshift": {"left": -70, "right": 70}.get(side, 0),
        "yshift": {"bottom": -70, "top": 70}.get(side, 0),
        "textangle": 0 if horizontal else -90,
        "showarrow": False,
    }
