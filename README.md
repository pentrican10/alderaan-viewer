# alderaan-viewer
Web app for viewing ALDERAAN pipeline outputs. The core ALDERAAN pipeline can be found at https://github.com/gjgilbert/alderaan.

## Installation Instructions:
```
$ git clone https://github.com/pentrican10/alderaan-viewer <LOCAL_DIR>
$ conda env create -n <ENV_NAME> -f <LOCAL_DIR>/environment.yml

if <ENV_NAME> is not specified, the conda environment will be named "alderaan-viewer"
```

## Quickstart guide:

To run the ALDERAAN viewer:

1. Set the data directory 
    - Manually change data_directory at top of alderaan_viewer.py and data_load.py
    - ex: 'c:\\Users\\Paige\\Projects\\data\\alderaan_results' (where you store data)

2. For California Planet Search members, ALDERAAN data products may be retrieved from the `cadence` server. Eventually these data products will be hosted publicly.
  
3. Ensure necessary files are in the data directory (structured after the data produced from ALDERAAN)
    - <data_directory>
        - <data_run>
            - <koi_id>
                - <koi_id>_(n)_quick.ttvs
                - <koi_id>_lc_filtered.fits
                - <koi_id>-results.fits
    - Example: <data_directory>
        - ecc-all-LC
            - K00001
                - K00001_00_quick.ttvs
                - K00001_lc_filtered.fits
                - K00001-results.fits
                
4. Manually add table to <data_run> folder (path for table should be '<data_directory>\\<data_run>\\<data_run.csv>')
    - Make sure to change table name from 'kepler_dr25_gaia_dr2_crossmatch.csv' to '<data_run>.csv'
        - copy paste table 'kepler_dr25_gaia_dr2_crossmatch.csv' and change name

5. From terminal, run alderaan_viewer.py


-----
## Troubleshooting:
- Make sure to mannually reset the corner plot and single transit to '00' option before changing targets
    - If this is not done, reload the page and continue
- if TTV says 'error loading plot' when first loading, uncheck and recheck the TTV box at the top of the screen

## Use:
- Login with username
- Select table from dropdown on left side
- Select KOI ID to be viewed
    - Automatically populates detrended light curve, folded light curves, OMC plots, posterior plots, and planet properties table
    - Select Single Transit box to view single transits
    - Change planet via planet select dropdown for single transit and posterior plots
        - Make sure to select '00' before changing targets
    - Mark review status and leave any relevant comments
 
## Notes:
- When you leave a comment, a comment file will be created at '<data_directory>\\<data_run>\\<koi_id>\\<koi_id_comments.txt>'

