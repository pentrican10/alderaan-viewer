# alderaan-viewer
Web app for viewing ALDERAAN pipeline outputs.

File Structure:
- data_directory: path to data for alderaan-results 
    - ex: 'c:\\Users\\Paige\\Projects\\data\\alderaan_results'
- table: default table when app opens:
    - ex: 'ecc-all-LC.csv'

Data downloaded from Cadence: from '/data/user/gjgilbert/projects/alderaan/Results/<data_run>' to '<data_directory>'
- substitute 'c:\\' for '/mnt/c/' if using wsl or mac
    - ex: 'rsync -avh paige@cadence.caltech.edu:/data/user/gjgilbert/projects/alderaan/Results/ecc-all-LC /mnt/c/Users/Paige/Projects/data/alderaan_results'
- manually add table to this folder (path for table should be '<data_directory>\\<data_run>\\<data_run.csv>')
    - Make sure to change table name from 'kepler_dr25_gaia_dr2_crossmatch.csv' to '<data_run>.csv'
        - copy paste table 'kepler_dr25_gaia_dr2_crossmatch.csv' and change name



Notes:
- When you leave a comment, a comment file will be created at '<data_directory>\\<data_run>\\<koi_id>\\<koi_id_comments.txt>'
- bug work-arounds (until fixed):
    - Make sure to mannually reset the corner plot and single transit to '00' option before changing targets
    - if OMC says 'error loading plot' when first loading, uncheck and recheck the OMC box at the top of the screen
