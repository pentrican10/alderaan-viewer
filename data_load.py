from flask import Flask, render_template, jsonify, request, session, redirect, url_for
import os
import csv
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from astropy.io import fits
import pandas as pd
import json
import plotly.utils
import numpy as np
import sys
import lightkurve as lk
import numpy.polynomial.polynomial as poly
import glob


#data_directory = 'c:\\Users\\Paige\\Projects\\data\\'
data_directory = 'c:\\Users\\Paige\\Projects\\data\\alderaan_results'


def update_data_directory(selected_table):
    global data_directory
    data_directory = os.path.join('c:\\Users\\Paige\\Projects\\data\\alderaan_results', selected_table[:-4])

def read_table_data(table):
    """
    Reads data for the table on the left side of the web app
    Shows koi_id, kep_mag, Rstar, logrho, Teff, logg
    """
    #file_path = os.path.join(data_directory, '2023-05-19_singles.csv')
    global data_directory
    #folder = table[:-4]
    update_data_directory(table)
    file_path = os.path.join(data_directory, table)
    table_data = []
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            #round table values
            row['kep_mag'] = round(float(row['kep_mag']), 2)
            row['Rstar'] = round(float(row['Rstar']), 2)
            row['logrho'] = round(float(row['logrho']), 2)
            row['Teff'] = round(float(row['Teff']))
            row['logg'] = round(float(row['logg']), 2)
            table_data.append(row)
            
    ### Remove duplicates based on koi_id
    unique_data = []
    seen_koi_ids = set()
    for row in table_data:
        koi_id = row['koi_id']
        ### Check if koi_id is not in the set of seen ids
        if koi_id not in seen_koi_ids:
            ### Add the row to unique_data and the koi_id to the set
            unique_data.append(row)
            seen_koi_ids.add(koi_id)

    return unique_data
    #return table_data

def read_data_from_fits(file_path):
    with fits.open(file_path) as fits_file:
        time = np.array(fits_file[1].data, dtype=float)
        flux = np.array(fits_file[2].data, dtype=float)
        df = pd.DataFrame(dict(
            TIME=time,
            FLUX=flux
        ))
    return df


def get_ttv_file(koi_id, file_path):
    #star_id = koi_id.replace("K","S")
    #file_name = star_id + '_00_quick.ttvs'
    #file_path = os.path.join(data_directory,'quick_ttvs_for_paige',file_name)
    if os.path.isfile(file_path):
        index =[]
        ttime=[] 
        model = []
        out_prob = []
        out_flag = []
        # Open the file for reading
        with open(file_path, 'r') as file:
            # Iterate through each line in the file
            for line in file:
                # Split the line into columns based on the delimiter
                columns = line.strip().split('\t')

                index.append(columns[0])
                ttime.append(columns[1])
                model.append(columns[2])
                out_prob.append(columns[3])
                out_flag.append(columns[4])
                
        #df = pd.read_csv(file_path, delimiter='\t', header=None)
        return index, ttime, model, out_prob, out_flag
    
def fetch_data(koi_id, line_number):
    star_id = koi_id.replace("K","S")
    file_name_lc = star_id + '_lc_detrended.fits'
    file_path_lc = os.path.join(data_directory,'kepler_lightcurves_for_paige',file_name_lc)
    
    file_name_sc = star_id + '_sc_filtered.fits'
    file_path_sc = os.path.join(data_directory, file_name_sc)

    combined_data = None
    #get data and create detrended light curve
    if os.path.isfile(file_path_lc):
        photometry_data_lc = read_data_from_fits(file_path_lc) #descriptive names
        index, ttime, model, out_prob, out_flag = get_ttv_file(koi_id)

        if line_number < len(index):
            center_time = ttime[line_number]
            transit_number = index[line_number]
        
            start_time = float(center_time) - 0.25
            end_time= float(center_time) + 0.25

            use_lc = (photometry_data_lc['TIME'] > start_time) & (photometry_data_lc['TIME'] < end_time)
            lc_data = photometry_data_lc[use_lc]
            combined_data = lc_data
            #return filtered_data, transit_number, center_time ############
    if os.path.isfile(file_path_sc):
            photometry_data_sc = read_data_from_fits(file_name_sc)
            if combined_data is not None:
                use_sc = (photometry_data_sc['TIME'] > start_time) & (photometry_data_sc['TIME'] < end_time)
                sc_data = photometry_data_sc[use_sc]
                combined_data= pd.concat([combined_data, sc_data],ignore_index=True)
            else:
                combined_data = photometry_data_sc

    return combined_data, transit_number, center_time ############
    

def folded_data(koi_id, file_path):
    star_id = koi_id.replace("K","S")
    file_name_lc = star_id + '_lc_detrended.fits'
    #file_path_lc = os.path.join(data_directory,'kepler_lightcurves_for_paige',file_name_lc)
    file_path_lc = os.path.join(data_directory, star_id, file_name_lc)
    
    file_name_sc = star_id + '_sc_filtered.fits'
    #file_path_sc = os.path.join(data_directory, file_name_sc)
    file_path_sc = os.path.join(data_directory, star_id, file_name_sc)

    fold_data_time = []
    fold_data_flux = []
    #get data and create detrended light curve
    if os.path.isfile(file_path_lc):
        photometry_data_lc = read_data_from_fits(file_path_lc) #descriptive names
        #photometry_data_sc = read_data_from_fits(file_path_sc)
        index, ttime, model, out_prob, out_flag = get_ttv_file(koi_id, file_path)
        
        for i in range(len(index)):
            center_time = ttime[i]
        
            start_time = float(center_time) - 0.25
            end_time= float(center_time) + 0.25

            use = (photometry_data_lc['TIME'] > start_time) & (photometry_data_lc['TIME'] < end_time)
            transit_data = photometry_data_lc[use]

            #use_sc = (photometry_data_sc['TIME']>start_time) & (photometry_data_sc['TIME']<end_time)
            #transit_data_sc = photometry_data_sc[use_sc]

            # Check and ensure that 'TIME' column is of numeric type
            transit_data['TIME'] = pd.to_numeric(transit_data['TIME'], errors='coerce')
            #transit_data_sc['TIME'] = pd.to_numeric(transit_data_sc['TIME'], errors='coerce')
            
            # Check and ensure that center_time is of numeric type
            center_time = float(center_time)

            norm_time = transit_data['TIME'] - center_time
            #norm_time_sc = transit_data_sc['TIME'] - center_time
            fold_data_time.extend(norm_time)
            fold_data_flux.extend(transit_data['FLUX'])
            #fold_data_time.extend(norm_time_sc)
            #fold_data_flux.extend(transit_data_sc['FLUX'])
    if os.path.isfile(file_path_sc):
        photometry_data_sc = read_data_from_fits(file_path_sc)
        for i in range(len(index)):
            center_time = ttime[i]
        
            start_time = float(center_time) - 0.25
            end_time= float(center_time) + 0.25
            use_sc = (photometry_data_sc['TIME']>start_time) & (photometry_data_sc['TIME']<end_time)
            transit_data_sc = photometry_data_sc[use_sc]
            transit_data_sc['TIME'] = pd.to_numeric(transit_data_sc['TIME'], errors='coerce')
            center_time = float(center_time)
            norm_time_sc = transit_data_sc['TIME'] - center_time
            fold_data_time.extend(norm_time_sc)
            fold_data_flux.extend(transit_data_sc['FLUX'])
  
            

    fold_data = pd.DataFrame({
        'TIME' : fold_data_time,
        'FLUX': fold_data_flux
    })
    fold_data['TIME'] = fold_data['TIME'] * 24 ### to hours
    return fold_data

def OMC_data(koi_id,file_path):
    index, ttime, model, out_prob, out_flag = get_ttv_file(koi_id, file_path)
    index = np.asarray(index, dtype=np.int64)
    model = np.asarray(model, dtype=np.float64)
    ttime = np.asarray(ttime, dtype=np.float64)
    model = np.asarray(model, dtype=np.float64)
    out_prob = np.asarray(out_prob, dtype=np.float64)
    out_flag = np.asarray(out_flag, dtype=np.float64)
    t0, period = poly.polyfit(index, model, 1)
    omc_model = model - poly.polyval(index, [t0, period])
    omc_ttime =ttime - poly.polyval(index, [t0, period])

    omc_time_data = omc_ttime*24*60
    omc_model_data = omc_model*24*60

    OMC_data = pd.DataFrame({
        'TIME' : ttime,
        'OMC' : omc_time_data
    })

    OMC_model = pd.DataFrame({
        'TIME' : ttime,
        'OMC_MODEL' : omc_model_data
    })

    return OMC_data, OMC_model, out_prob, out_flag


### MULTI PLANET SYSTEMS

