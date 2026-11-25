"""
Flask app: pages, plot and table API routes

Configuration (data_dir) is set once in app.config. The run and
target always come from the URL; the session holds only the username. Results
are read from disk on every request (no cache).
"""

import os
import secrets

import plotly
from alderaan.data_models.litecurve import LiteCurve
from alderaan.data_models.results import Results
from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    make_response,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)

from alderaan_viewer import figures, reviews, runs, tables

CATALOG_COLUMNS = ("kep_mag", "Rstar", "logrho", "Teff", "logg")
PLOTLY_JS = os.path.join(
    os.path.dirname(plotly.__file__), "package_data", "plotly.min.js"
)


def _fail(code, message):
    abort(make_response(jsonify({"error": message}), code))


def _figure(fig):
    return Response(fig.to_json(), mimetype="application/json")


def create_app(data_dir):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"directory not found: {data_dir}")

    app = Flask(__name__)
    app.secret_key = secrets.token_hex(32)
    app.config["DATA_DIR"] = data_dir

    def target(run_id, koi):
        """(paths, state, catalog rows); 404 if the run or koi is unknown"""
        if run_id not in runs.list_runs(data_dir):
            _fail(404, f"unknown run {run_id!r}")
        catalog = runs.read_catalog(data_dir, run_id)
        rows = catalog[catalog["koi_id"] == koi]
        if rows.empty:
            _fail(404, f"{koi!r} not in the catalog of run {run_id}")
        paths = runs.target_paths(data_dir, run_id, koi)
        return paths, runs.target_state(paths), rows

    def load_results(run_id, koi):
        """(Results with litecurve, catalog rows); 409 if the target is not complete"""
        paths, state, rows = target(run_id, koi)
        if state != runs.COMPLETE:
            _fail(409, f"{koi}: {state}: {runs.log_status(paths.log)}")
        return Results.from_fits(paths.results, paths.detrended), rows

    @app.before_request
    def require_login():
        if request.endpoint in ("login", "static") or "username" in session:
            return None
        if request.path.startswith("/api/"):
            _fail(401, "not logged in")
        return redirect(url_for("login"))

    @app.route("/")
    def index():
        return redirect(url_for("home"))

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            session["username"] = request.form["username"].strip()
            return redirect(url_for("home"))
        return render_template("login.html")

    @app.route("/logout", methods=["POST"])
    def logout():
        session.pop("username", None)
        return redirect(url_for("login"))

    @app.route("/plotly.min.js")
    def plotly_js():
        return send_file(PLOTLY_JS, mimetype="text/javascript")

    @app.route("/home")
    def home():
        run_ids = runs.list_runs(data_dir)
        if not run_ids:
            abort(404, f"no runs in {data_dir}")
        run_id = request.args.get("run")
        if run_id is None:
            return redirect(url_for("home", run=run_ids[0]))
        if run_id not in run_ids:
            abort(404, f"unknown run {run_id!r}")

        catalog = runs.read_catalog(data_dir, run_id)
        statuses = reviews.read_statuses(runs.run_dir(data_dir, run_id))
        first = catalog.drop_duplicates("koi_id").set_index("koi_id")
        targets = []
        for koi in runs.target_ids(catalog):
            row = {c: first.at[koi, c] for c in CATALOG_COLUMNS}
            row["koi_id"] = koi
            row["state"] = runs.target_state(runs.target_paths(data_dir, run_id, koi))
            row["review"] = statuses.get(koi, "None")
            targets.append(row)

        return render_template(
            "index.html",
            run_id=run_id,
            run_ids=run_ids,
            targets=targets,
            statuses=reviews.STATUSES,
            corner_params=figures.CORNER_PARAMS,
        )

    @app.route("/api/<run_id>/<koi>/status")
    def status(run_id, koi):
        paths, state, rows = target(run_id, koi)
        out = {"state": state, "message": "", "planets": []}
        if state == runs.COMPLETE:
            results = Results.from_fits(paths.results)
            names = tables.planet_names(results)
            out["planets"] = [{"n": n, "name": name} for n, name in enumerate(names)]
        else:
            out["message"] = runs.log_status(paths.log)
        return jsonify(out)

    @app.route("/api/<run_id>/<koi>/lightcurve")
    def lightcurve(run_id, koi):
        paths, state, _ = target(run_id, koi)
        if state == runs.NOT_PROCESSED:
            _fail(409, f"{koi}: {state}: {runs.log_status(paths.log)}")
        if state == runs.INCOMPLETE:
            return _figure(figures.lightcurve(LiteCurve.from_fits(paths.detrended)))
        results = Results.from_fits(paths.results, paths.detrended)
        ttimes = [tt.ttime_mod for tt in results.transit_times]
        return _figure(figures.lightcurve(results.litecurve, ttimes))

    @app.route("/api/<run_id>/<koi>/folded")
    def folded(run_id, koi):
        results, _ = load_results(run_id, koi)
        return _figure(figures.folded(results, tables.planet_names(results)))

    @app.route("/api/<run_id>/<koi>/transits/<int:n>")
    def transits(run_id, koi, n):
        results, _ = load_results(run_id, koi)
        names = tables.planet_names(results)
        if not 0 <= n < len(names):
            _fail(404, f"{koi} has no planet {n}")
        start = request.args.get("start", 0, type=int)
        try:
            fig = figures.single_transits(results, n, start, names[n])
        except ValueError as e:
            _fail(400, str(e))
        return _figure(fig)

    @app.route("/api/<run_id>/<koi>/omc")
    def omc(run_id, koi):
        results, _ = load_results(run_id, koi)
        return _figure(figures.omc(results, tables.planet_names(results)))

    @app.route("/api/<run_id>/<koi>/corner/<int:n>")
    def corner(run_id, koi, n):
        results, _ = load_results(run_id, koi)
        names = tables.planet_names(results)
        if not 0 <= n < len(names):
            _fail(404, f"{koi} has no planet {n}")
        params = [p for p in request.args.get("params", "").split(",") if p]
        try:
            fig = figures.corner(results, n, params, names[n])
        except ValueError as e:
            _fail(400, str(e))
        return _figure(fig)

    @app.route("/api/<run_id>/<koi>/tables")
    def planet_tables(run_id, koi):
        results, rows = load_results(run_id, koi)
        properties = tables.planet_properties(results, list(rows["planet_name"]))
        return jsonify(
            {
                "properties": properties,
                "period_ratios": tables.period_ratios(properties),
            }
        )

    @app.route("/api/<run_id>/<koi>/review", methods=["POST"])
    def review(run_id, koi):
        target(run_id, koi)
        status_value = request.get_json()["status"]
        try:
            reviews.append_status(
                runs.run_dir(data_dir, run_id), koi, status_value, session["username"]
            )
        except ValueError as e:
            _fail(400, str(e))
        return jsonify({"status": status_value})

    @app.route("/api/<run_id>/<koi>/comments", methods=["GET", "POST", "PUT"])
    def comments(run_id, koi):
        target(run_id, koi)
        run_dir = runs.run_dir(data_dir, run_id)
        if request.method == "POST":
            text = request.get_json()["comment"]
            reviews.append_comment(run_dir, koi, text, session["username"])
        elif request.method == "PUT":
            reviews.replace_comments(run_dir, koi, request.get_json()["content"])
        return Response(reviews.read_comments(run_dir, koi), mimetype="text/plain")

    return app
