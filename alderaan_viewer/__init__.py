"""ALDERAAN-viewer: a Flask + plotly app for inspecting ALDERAAN pipeline outputs"""

# alderaan imports matplotlib.pyplot at module level; select the non-GUI backend
# before any alderaan import so request threads never touch a GUI backend.
import matplotlib

matplotlib.use("Agg")
