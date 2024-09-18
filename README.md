# alderaan-viewer
Web app for viewing ALDERAAN pipeline outputs.

## Installation Instructions:
```
$ git clone https://github.com/pentrican10/alderaan-viewer <LOCAL_DIR>
$ conda env create -n <ENV_NAME> -f <LOCAL_DIR>/environment.yml

if <ENV_NAME> is not specified, the conda environment will be named "alderaan-viewer"
```

## Quickstart guide:

To run the ALDERAAN viewer

1. Set the data directory
    - Manually change data_directory at top of alderaan_viewer.py and data_load.py
    - ex: 'c:\\Users\\Paige\\Projects\\data\\alderaan_results'


-----

File Structure:
- data_directory: path to data for alderaan-results 
    - Manually change data_directory at top of alderaan_viewer.py and data_load.py
    - ex: 'c:\\Users\\Paige\\Projects\\data\\alderaan_results'
- table: default table when app opens:
    - ex: 'ecc-all-LC.csv'

Data downloaded from Cadence: from '/data/user/gjgilbert/projects/alderaan/Results/<data_run>' to '<data_directory>'
- substitute 'c:\\' for '/mnt/c/' if using wsl or mac
    - ex: 'rsync -avh paige@cadence.caltech.edu:/data/user/gjgilbert/projects/alderaan/Results/ecc-all-LC /mnt/c/Users/Paige/Projects/data/alderaan_results'
        - note here that 'ecc-all-LC' is the data_run
- manually add table to this folder (path for table should be '<data_directory>\\<data_run>\\<data_run.csv>')
    - Make sure to change table name from 'kepler_dr25_gaia_dr2_crossmatch.csv' to '<data_run>.csv'
        - copy paste table 'kepler_dr25_gaia_dr2_crossmatch.csv' and change name



Notes:
- When you leave a comment, a comment file will be created at '<data_directory>\\<data_run>\\<koi_id>\\<koi_id_comments.txt>'
- bug work-arounds (until fixed):
    - Make sure to mannually reset the corner plot and single transit to '00' option before changing targets
    - if OMC says 'error loading plot' when first loading, uncheck and recheck the OMC box at the top of the screen

Use:
- Login with username
- Select table from dropdown on left side
- Select KOI ID to be viewed
    - Automatically populates detrended light curve, folded light curves, OMC plots, posterior plots, and planet properties table
    - Select Single Transit box to view single transits
    - Change planet via planet select dropdown for single transit and posterior plots
        - Make sure to select '00' before changing targets
    - Mark review status and leave any relevant comments
