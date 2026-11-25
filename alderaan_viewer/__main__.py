"""Entry point: python -m alderaan_viewer --data_dir DIR"""

import argparse
import os
import threading
import webbrowser

from alderaan_viewer.app import create_app


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m alderaan_viewer")
    parser.add_argument(
        "--data_dir",
        required=True,
        help="ALDERAAN output directory; each run is a subdirectory <run_id>/ "
        "(review status and comments are written there)",
    )
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args(argv)

    app = create_app(args.data_dir)

    # the debug reloader re-runs this module in a child process with
    # WERKZEUG_RUN_MAIN set; open the browser once, from the parent
    if "WERKZEUG_RUN_MAIN" not in os.environ:
        url = f"http://127.0.0.1:{args.port}/"
        threading.Timer(1.0, webbrowser.open, args=[url]).start()

    # Flask dev server: localhost only (debug mode, no real authentication)
    app.run(host="127.0.0.1", port=args.port, debug=True)


if __name__ == "__main__":
    main()
