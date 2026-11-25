# alderaan-viewer
Web app for viewing ALDERAAN pipeline outputs. The core ALDERAAN pipeline can be found at https://github.com/gjgilbert/alderaan.

## Installation Instructions:
```
$ git clone https://github.com/pentrican10/alderaan-viewer <LOCAL_DIR>/alderaan-viewer
```

The viewer runs in the `alderaan` conda environment. Create it from the ALDERAAN repo
(https://github.com/gjgilbert/alderaan) with `conda env create -f environment.yml`, or
add the viewer dependencies to an existing install with `pip install -e ".[viewer]"`.

## Quickstart guide:

ALDERAAN-viewer reads the output directory of an ALDERAAN batch run (ALDERAAN 0.3.0):

    <data_dir>/
        <run_id>/
            alderaan_batch_<run_id>.csv
            results/<koi_id>/<koi_id>_detrended.fits
            results/<koi_id>/<koi_id>_results.fits
            logs/alderaan_<koi_id>_<run_id>.log

Each subdirectory of `<data_dir>` that contains its `alderaan_batch_<run_id>.csv` is a run.
The viewer writes only its review status and comments into each run directory, beside
`alderaan_batch_<run_id>.csv`.

For California Planet Search members, ALDERAAN data products may be retrieved from the Caltech cadence server. Eventually these data products will be hosted publicly.

To run the ALDERAAN viewer, from the root of this repository:

    $ conda activate alderaan
    $ python -m alderaan_viewer --data_dir <data_dir>

then open http://127.0.0.1:5000. The server listens on localhost only.

To run the tests: `python -m pytest` (tests that read `sample_output/` skip when it is absent).

-----
## Use:
- Login with username
- Select a run from the dropdown on the left side
- Select a KOI ID to view
    - Shows the detrended light curve, folded transits, O-C plots, corner plot, planet properties and period ratios
    - Select the Single Transit box to view single transits; use the arrows to page
    - Change planet via the planet select dropdowns for single transit and corner plots
    - Mark review status and leave any relevant comments
- Targets that the pipeline did not process, or did not finish, show the reason from the pipeline log

## Notes:
- Review status: `<data_dir>/<run_id>/review_status.csv` (one row per change; the latest row wins)
- Comments: `<data_dir>/<run_id>/comments/<koi_id>.txt`

## Example Plots:
### For K00624, a three-planet system
![image](https://github.com/user-attachments/assets/47e2ee27-fee6-4197-8024-f56888bc2297)
![image](https://github.com/user-attachments/assets/e2721b9d-5a2b-4bb6-90a7-40b6efff6b0a)
![image](https://github.com/user-attachments/assets/4d04b5b1-79ea-4a6b-bd1a-a532168255d6)
![image](https://github.com/user-attachments/assets/f2c08f6b-ba51-4bcd-88b3-63ccbaaefb6c)
![image](https://github.com/user-attachments/assets/9d0eef31-b6db-4eff-91d0-70f004118089)
