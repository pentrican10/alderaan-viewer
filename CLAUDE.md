# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

ALDERAAN-viewer: a Flask + plotly web app for inspecting outputs of the ALDERAAN transit
photometry pipeline (companion repo: `~/projects/alderaan`). Users pick a pipeline run,
pick a KOI, view light curves, folded transits, O-C plots, and posteriors, then record a
review status and comments.

The viewer is being updated to read the current ALDERAAN output format. Compatibility
with old output formats is **not** required. The pipeline's design principles apply here
too: no hidden state, no implicit assumptions, fail loudly, prefer clarity over cleverness,
simplest possible implementation. The viewer displays pipeline results; it must not
silently recompute or reinterpret science quantities the pipeline already provides.

## Working Context

- Sessions normally start in `~/projects/alderaan`. The alderaan `CLAUDE.md` governs
  shared conventions.
- This is a separate git repo. Run git here with `git -C ~/projects/alderaan_viewer`.
- Work on branch `alderaan-v0.3.0`.

## Layout

- `alderaan-viewer.py`: Flask app; all routes and plot construction.
- `data_load.py`: file reading and derived quantities.
- `templates/`: `login.html`, `index.html` (wraps `left.html`, `right_top.html`,
  `right_bottom.html`); JS in the templates calls the Flask routes.
- `static/styles.css`.

## Pipeline Output Format (current, ALDERAAN 0.3.0)

One directory per run, `<output_dir>/<run_id>/`:

```
alderaan_batch_<run_id>.csv        # catalog copy, one row per planet (koi_id, planet_name, period, ...)
results/<koi>/<koi>_detrended.fits # LITECURVE table: TIME, FLUX, ERROR, CADNO, VISIT, OBSMODE
results/<koi>/<koi>_results.fits   # SAMPLES, EPHEMERIS, BATCHES, TTIMES_NN (one per planet)
plots/<koi>/*.png                  # pipeline diagnostic plots
logs/alderaan_<koi>_<run_id>.log   # per-target log; logs/alderaan_batch_<run_id>.log
```

- `_results.fits` primary header: `NPL`, `PLNAME0..`, `RUN_ID`, `VERSION`, `GIT_HASH`, ...
- `SAMPLES` columns: `C0_n`, `C1_n`, `ROR_n`, `IMPACT_n`, `DUR14_n`, `LD_Q1`, `LD_Q2`,
  `LN_LIKE`, `LN_VOL`, `N_LIVE`, `N_CALL`, `SAMP_ID`, `SAMP_BATCH`.
- `EPHEMERIS` columns: `PLANET`, `REF_PERIOD`, `REF_EPOCH`.
- `TTIMES_NN` columns: `INDEX`, `TTIME_OBS`, `TTIME_MOD`, `TTIME_ERR`, `PHOT_COVERAGE`,
  `PHOT_NOISE`, `KEPT_TTV`; header `OMCMODEL`, `OMCDOF`, `TTVSCAT`.
- A target can have no results (unsupported; reason in its log) or only
  `_detrended.fits` (incomplete run). The viewer must show these states explicitly.
- Authoritative readers: `alderaan.data_models.results.Results.from_fits` and
  `alderaan.data_models.litecurve.LiteCurve.from_fits`. Check the alderaan source when
  the format is in question, not this summary.

## Sample Data

`sample_output/20261123T214548/` (gitignored) is a 4-target subset of a real batch run
(`~/data/ALDERAAN_outputs/20261123T214548/`, read-only):

- `K00001`: one planet, complete.
- `K00137`: three planets, TTVs, complete.
- `K00099`: not processed (unsupported); empty results directory.
- `K00971`: incomplete; `_detrended.fits` only.

## Current State (pre-migration)

The code still reads the legacy format (`*_lc_filtered.fits`, `*_sc_filtered.fits`,
`*-results.fits`, `*_NN_quick.ttvs`, `../alderaan/Results/<run_id>/<run_id>.csv`) and does
not run against the sample data.

## Environment

The viewer uses the `alderaan` conda env. Its dependencies (`flask`, `plotly`) are the
`viewer` optional group in alderaan's `pyproject.toml` and are listed in alderaan's
`environment.yml`. Add viewer dependencies there, not in this repo.

## Running

```bash
conda run -n alderaan python alderaan-viewer.py   # Flask dev server at http://127.0.0.1:5000
```

Never expose the dev server beyond localhost: it runs with `debug=True` and has no real
authentication.
