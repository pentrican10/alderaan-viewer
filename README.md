# alderaan-viewer
Web app for viewing ALDERAAN pipeline outputs. The core ALDERAAN pipeline can be found at https://github.com/gjgilbert/alderaan.

## Installation Instructions:
```
$ git clone https://github.com/pentrican10/alderaan-viewer <LOCAL_DIR>/alderaan-viewer
$ conda env create -n <ENV_NAME> -f <LOCAL_DIR>/alderaan-viewer/environment.yml

if <ENV_NAME> is not specified, the conda environment will be named "alderaan-viewer"
```

## Quickstart guide:

ALDERAAN-viewer expects the following files and directory structure:

   - <LOCAL_DIR>
        - alderaan-viewer/
            - alderaan-viewer.py
            - data_load.py
            - static/
            - templates/
        - alderaan/
            - Results/
                - <run_ID>
                    - <koi_id>
                        - <koi_id>_(n)_quick.ttvs
                        - <koi_id>_lc_filtered.fits
                        - <koi_id>_sc_filtered.fits
                        - <koi_id>-results.fits

The "alderaan-viewer" folder should contain a clone of this repository, while the "alderaan" folder will be used to store outputs from the ALDERAAN pipeline. For example the outputs for the three-planet system KOI-137 during ALDERAAN run with the identifier "ttv-active" would be placed in <LOCAL_DIR>/alderaan/Results/ttv-active/K00137/ and would expect the following files:

- K00137_00_quick.ttvs
- K00137_01_quick.ttvs
- K00137_02_quick.ttvs
- K00137_lc_filtered.fits
- K00137_sc_filtered.fits
- K00137-results.fits

These data products are generated automatically by the ALDERAAN pipeline with the expected file structure. The K00137_sc_filtered.fits file will only exist for targets which have been observed in short cadence mode.

For California Planet Search members, ALDERAAN data products may be retrieved from the Caltech cadence server. Eventually these data products will be hosted publicly.


To run the ALDERAAN viewer:

1. Ensure that all files are placed in the correct location 
  
2. Manually copy the csv input table to <LOCAL_DIR>/alderaan/Results/<run_id>/ folder
    - Change table name from 'kepler_dr25_gaia_dr2_crossmatch.csv' to '<run_id>.csv'
    - Future versions of ALDERAAN will updated to generate these tables automatically

3. Activate the conda environment

    ```$ conda activate alderaan-viewer```

4. From the terminal, navigate into <LOCAL_DIR>/alderaan-viewer/ and run the app

    ```$ python alderaan-viewer.py```

-----
## Troubleshooting:
- if TTV says 'error loading plot' when first loading, uncheck and recheck the TTV box at the top of the screen

## Use:
- Login with username
- Select table from dropdown on left side
- Select KOI ID to be viewed
    - Automatically populates detrended light curve, folded light curves, OMC plots, posterior plots, and planet properties table
    - Select Single Transit box to view single transits
    - Change planet via planet select dropdown for single transit and posterior plots
    - Mark review status and leave any relevant comments
 
## Notes:
- When you leave a comment, a comment file will be created at '<LOCAL_DIR>/alderaan/Results/<run_id>/<koi_id>/<koi_id>_comments.txt'

## Example Plots:
- For K00624
![image](https://github.com/user-attachments/assets/47e2ee27-fee6-4197-8024-f56888bc2297)

