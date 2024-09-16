# alderaan-viewer
Web app for viewing ALDERAAN pipeline outputs.

File Structure:
- data_directory: path to data for alderaan-results 
    - ex: 'c:\\Users\\Paige\\Projects\\data\\alderaan_results'
- table: default table when app opens:
    - ex: 'ecc-all-LC.csv'

Data downloaded from Cadence: '/data/user/gjgilbert/projects/alderaan/Results/<data_run>' to '<data_directory>'
- substitute 'c:\\' for '/mnt/c/' if using wsl or mac
- manually add table to this folder (path for table should be '<data_directory>\\<data_run>\\<data_run.csv>')

Notes:
- bug work-arounds (until fixed):
    - Make sure to mannually reset the corner plot and single transit to '00' option before changing targets
    - if OMC says 'error loading plot' when first loading, uncheck and recheck the OMC box at the top of the screen