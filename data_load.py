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
import batman



#data_directory = 'c:\\Users\\Paige\\Projects\\data\\'
data_directory = 'c:\\Users\\Paige\\Projects\\data\\alderaan_results'
k_id = True


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
    global K_id
    #folder = table[:-4]
    update_data_directory(table)
    if (table == '2023-05-19_singles.csv') or (table == '2023-05-15_doubles.csv'):
        K_id = False
    else: 
        K_id = True
    file_path = os.path.join(data_directory, table)
    table_data = []
    review_column_added = False
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        ### Check if 'review' column exists, otherwise add it
        fieldnames = reader.fieldnames
        if 'review' not in fieldnames:
            fieldnames.append('review')
            review_column_added = True

        for row in reader:
            #round table values
            row['kep_mag'] = round(float(row['kep_mag']), 2)
            row['Rstar'] = round(float(row['Rstar']), 2)
            row['logrho'] = round(float(row['logrho']), 2)
            row['Teff'] = round(float(row['Teff']))
            row['logg'] = round(float(row['logg']), 2)
            # Ensure 'review' column exists in each row
            if 'review' not in row:
                row['review'] = 'None'
            elif row['review'] == '':
                row['review'] = 'None'
            table_data.append(row)
    if review_column_added==True:
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(table_data)
            
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

def get_periods_for_koi_id(file_path, koi_id):
    koi_identifiers = []
    periods = []
    period_title = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            if row['koi_id'] == koi_id:
                koi_identifier = str(row['planet_name'])
                period_value = float(row['period'])
                rounded_period = round(period_value, 1)
                append = f'Period: {rounded_period} Days'
                period_title.append(str(append))
                periods.append(rounded_period)
                koi_identifiers.append(str(f'{koi_identifier}'))

        df = pd.DataFrame(dict(
            koi_identifiers=koi_identifiers,
            periods= periods,
            period_title = period_title
        ))

    return df if periods else None
    #return periods,koi_identifiers if periods else None

               
def read_data_from_fits(file_path):
    with fits.open(file_path) as fits_file:
        time = np.array(fits_file[1].data, dtype=float)
        flux = np.array(fits_file[2].data, dtype=float)
        err = np.array(fits_file[3].data, dtype=float)
        cadno = np.array(fits_file[4].data, dtype=int)
        quarter = np.array(fits_file[5].data, dtype=int)
        df = pd.DataFrame(dict(
            TIME=time,
            FLUX=flux,
            ERR = err,
            CADNO = cadno,
            QUARTER = quarter
        ))
    return df


def get_ttv_file(koi_id, file_path):
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
        return index, ttime, model, out_prob, out_flag
    

def get_min_max(koi_id):
    global K_id
    if K_id == False:
        star_id = koi_id.replace("K","S")
    else:
        star_id = koi_id
    file_name_lc = star_id + '_lc_filtered.fits'
    file_path_lc = os.path.join(data_directory,star_id,file_name_lc)
    
    file_name_sc = star_id + '_sc_filtered.fits'
    file_path_sc = os.path.join(data_directory, star_id, file_name_sc)

    if os.path.isfile(file_path_lc) and os.path.isfile(file_path_sc):
        photometry_data_lc = read_data_from_fits(file_path_lc) 
        photometry_data_sc = read_data_from_fits(file_path_sc)
        lc_max = photometry_data_lc['FLUX'].max()
        lc_min = photometry_data_lc['FLUX'].min()

        sc_max = photometry_data_sc['FLUX'].max()
        sc_min = photometry_data_sc['FLUX'].min()
        return lc_min,lc_max,sc_min,sc_max
    elif os.path.isfile(file_path_lc) and not os.path.isfile(file_path_sc):
        photometry_data_lc = read_data_from_fits(file_path_lc) 
        lc_max = photometry_data_lc['FLUX'].max()
        lc_min = photometry_data_lc['FLUX'].min()
        return lc_min, lc_max
    elif os.path.isfile(file_path_sc) and not os.path.isfile(file_path_lc):
        photometry_data_sc = read_data_from_fits(file_path_sc)
        sc_max = photometry_data_sc['FLUX'].max()
        sc_min = photometry_data_sc['FLUX'].min()
        return sc_min, sc_max


def single_data(koi_id, line_number, num, ttv_file):
    global K_id
    if K_id == False:
        star_id = koi_id.replace("K","S")
    else:
        star_id = koi_id
    file_name_lc = star_id + '_lc_filtered.fits'
    file_path_lc = os.path.join(data_directory,star_id,file_name_lc)
    
    file_name_sc = star_id + '_sc_filtered.fits'
    file_path_sc = os.path.join(data_directory, star_id, file_name_sc)

    file_path = os.path.join(data_directory, star_id, ttv_file)

    file_results =star_id + '-results.fits'
    file_path_results = os.path.join(data_directory, star_id, file_results)
    data_post = load_posteriors(file_path_results,num,koi_id)
    ### get max likelihood
    max_index = data_post['LN_LIKE'].idxmax()
    DUR14 = 1.5* data_post[f'DUR14_{num}'][max_index]

    combined_data = None
    #get data and create detrended light curve
    if os.path.isfile(file_path_lc) and os.path.isfile(file_path_sc):
        photometry_data_lc = read_data_from_fits(file_path_lc) 
        photometry_data_sc = read_data_from_fits(file_path_sc)
        index, ttime, model, out_prob, out_flag = get_ttv_file(koi_id, file_path)

        if line_number < len(index):
            center_time = ttime[line_number]
            transit_number = index[line_number]
        
            # start_time = float(center_time) - 0.25
            # end_time= float(center_time) + 0.25
            start_time = float(center_time) - DUR14
            end_time= float(center_time) + DUR14

            use_lc = (photometry_data_lc['TIME'] > start_time) & (photometry_data_lc['TIME'] < end_time)
            lc_data = photometry_data_lc[use_lc]
            combined_data = lc_data

            use_sc = (photometry_data_sc['TIME'] > start_time) & (photometry_data_sc['TIME'] < end_time)
            sc_data = photometry_data_sc[use_sc]

            combined_data= pd.concat([combined_data, sc_data],ignore_index=True)
        return lc_data, sc_data, transit_number, center_time
        
    elif os.path.isfile(file_path_lc):
        photometry_data_lc = read_data_from_fits(file_path_lc) #descriptive names
        index, ttime, model, out_prob, out_flag = get_ttv_file(koi_id, file_path)

        if line_number < len(index):
            center_time = ttime[line_number]
            transit_number = index[line_number]
        
            start_time = float(center_time) - 0.25
            end_time= float(center_time) + 0.25

            use_lc = (photometry_data_lc['TIME'] > start_time) & (photometry_data_lc['TIME'] < end_time)
            lc_data = photometry_data_lc[use_lc]
            combined_data = lc_data
            sc_data = None
        return lc_data,sc_data, transit_number, center_time
    elif os.path.isfile(file_path_sc):
        photometry_data_sc = read_data_from_fits(file_path_sc)
        index, ttime, model, out_prob, out_flag = get_ttv_file(koi_id, file_path)

        if line_number < len(index):
            center_time = ttime[line_number]
            transit_number = index[line_number]
        
            start_time = float(center_time) - 0.25
            end_time= float(center_time) + 0.25

            use_sc = (photometry_data_sc['TIME'] > start_time) & (photometry_data_sc['TIME'] < end_time)
            sc_data = photometry_data_sc[use_sc]
            combined_data = sc_data
            lc_data = None
        return lc_data,sc_data, transit_number, center_time


def folded_data(koi_id,planet_num, file_path):
    global K_id
    if K_id == False:
        star_id = koi_id.replace("K","S")
    else:
        star_id = koi_id
    file_name_lc = star_id + '_lc_filtered.fits'
    file_path_lc = os.path.join(data_directory, star_id, file_name_lc)
    
    file_name_sc = star_id + '_sc_filtered.fits'
    file_path_sc = os.path.join(data_directory, star_id, file_name_sc)

    file_results =star_id + '-results.fits'
    file_path_results = os.path.join(data_directory, star_id, file_results)
    data_post = load_posteriors(file_path_results,planet_num,koi_id)
    ### get max likelihood
    # sortby ln like, pick row 
    # get value after sampled
    # plot
    data_post = data_post.sort_values(by='LN_LIKE', ascending=False) 
    row = data_post.iloc[0] # pick row with highest likelihood
    ### mult by 1.5 for correct offset
    # DUR14 = 1.5 * data_post[f'DUR14_{planet_num}'][max_index]
    DUR14 = 1.5 * row[f'DUR14_{planet_num}']

    fold_data_time = []
    fold_data_flux = []
    fold_data_err = []
    fold_data_time_sc = []
    fold_data_flux_sc = []
    fold_data_err_sc = []
    #get data and create detrended light curve
    if os.path.isfile(file_path_lc):
        photometry_data_lc = read_data_from_fits(file_path_lc)
        index, ttime, model, out_prob, out_flag = get_ttv_file(koi_id, file_path)
        
        for i in range(len(index)):
            center_time = ttime[i]
        
            start_time = float(center_time) - DUR14
            end_time= float(center_time) + DUR14

            use = (photometry_data_lc['TIME'] > start_time) & (photometry_data_lc['TIME'] < end_time)
            transit_data = photometry_data_lc[use]

            ### Check and ensure that 'TIME' column is of numeric type
            transit_data['TIME'] = pd.to_numeric(transit_data['TIME'], errors='coerce')
            
            ### Check and ensure that center_time is of numeric type
            center_time = float(center_time)

            norm_time = transit_data['TIME'] - center_time
            fold_data_time.extend(norm_time)
            fold_data_flux.extend(transit_data['FLUX'])
            fold_data_err.extend(transit_data['ERR'])

    if os.path.isfile(file_path_sc):
        photometry_data_sc = read_data_from_fits(file_path_sc)
        for i in range(len(index)):
            center_time = ttime[i]
        
            start_time = float(center_time) - DUR14
            end_time= float(center_time) + DUR14
            use_sc = (photometry_data_sc['TIME']>start_time) & (photometry_data_sc['TIME']<end_time)
            transit_data_sc = photometry_data_sc[use_sc]
            transit_data_sc['TIME'] = pd.to_numeric(transit_data_sc['TIME'], errors='coerce')
            center_time = float(center_time)
            norm_time_sc = transit_data_sc['TIME'] - center_time
            fold_data_time_sc.extend(norm_time_sc)
            fold_data_flux_sc.extend(transit_data_sc['FLUX'])
            fold_data_err_sc.extend(transit_data_sc['ERR'])
  
    fold_data_lc = pd.DataFrame({
        'TIME' : fold_data_time,
        'FLUX': fold_data_flux,
        'ERR' : fold_data_err
    })
    # do this conversion in plotting, not data structure 
    # model all params in days, manipulate to hours in plotting 
    #plot days first then hours 

    fold_data_sc = pd.DataFrame({
        'TIME' : fold_data_time_sc,
        'FLUX': fold_data_flux_sc,
        'ERR' : fold_data_err_sc
    })

    bin_size = 0.02
    combined_time = np.concatenate([fold_data_lc['TIME'], fold_data_sc['TIME']])
    combined_flux = np.concatenate([fold_data_lc['FLUX'], fold_data_sc['FLUX']])
    combined_flux_err = np.concatenate([fold_data_lc['ERR'], fold_data_sc['ERR']]) 

    bin_centers_combined, weighted_avg_combined = calculate_binned_weighted_average(combined_time, combined_flux, combined_flux_err, bin_size)

    # Create DataFrame for combined binned weighted average data
    binned_weighted_avg_combined = pd.DataFrame({
        'TIME': bin_centers_combined,
        'FLUX': weighted_avg_combined
    })

    return fold_data_lc, fold_data_sc, binned_weighted_avg_combined, center_time

# Binned weighted average function
def calculate_binned_weighted_average(time, flux, flux_err, bin_size):
    bins = np.arange(time.min(), time.max(), bin_size)
    indices = np.digitize(time, bins, right=True)

    bin_centers = (bins[1:] + bins[:-1]) / 2
        
    weighted_avg = []
    for i in range(1, len(bins)):
        bin_indices = indices == i
        if any(bin_indices):
            weights = 1.0 / flux_err[bin_indices]#**2
            weighted_avg.append(np.average(flux[bin_indices], weights=weights))
        else:
            weighted_avg.append(np.nan)  # or use a different indicator for missing data?
        
    return bin_centers, weighted_avg
### mess around with binning in separate file, try just lc and just sc 
### center on midtransit point 

    
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


def load_posteriors(f,n,koi_id):
    global K_id
    if K_id == False:
        star_id = koi_id.replace("K","S")
    else:
        star_id = koi_id
    file_name = star_id + f'_0{n}_quick.ttvs'
    ttv_file = os.path.join(data_directory, star_id, file_name)
    with fits.open(f) as hduL:
        data = hduL['SAMPLES'].data
        C0 = data[f'C0_{n}']
        C1 = data[f'C1_{n}']
        ROR = data[f'ROR_{n}']
        IMPACT = data[f'IMPACT_{n}']
        DUR14 = data[f'DUR14_{n}']
        LD_Q1 = data[f'LD_Q1']
        LD_Q2 = data[f'LD_Q2']
        LN_WT = data[f'LN_WT']
        LN_LIKE = data[f'LN_LIKE']

        ### calculate P, T0, U1, U2
        LD_U1 = 2*np.sqrt(LD_Q1)*LD_Q2
        LD_U2 = np.sqrt(LD_Q1)*(1-2*LD_Q2)

        index, ttime, model, out_prob, out_flag = get_ttv_file(koi_id,ttv_file)
        # Leg0 = _legendre(koi_id,n,0)
        # Leg1 = _legendre(koi_id,n,1)
        model = np.array(model, dtype='float64')
        index = np.array(index, dtype='float64')

        # ephem = model + np.outer(C0,Leg0) + np.outer(C1, Leg1)
        # T0, P = poly.polyfit(index.flatten(),ephem.T,1)

        centered_index = (index - index[-1]) // 2
        LegX = centered_index / (index[-1]/2)
        Leg0 = np.ones_like(LegX)
        ephem = model + np.outer(C0, Leg0) + np.outer(C1,LegX)
        T0, P = poly.polyfit(index.flatten(),ephem.T,1)


        data_return = np.vstack([C0, C1, ROR, IMPACT, DUR14, T0, P, LD_Q1, LD_Q2, LD_U1, LD_U2, LN_WT, LN_LIKE]).T
        labels = f'C0_{n} C1_{n} ROR_{n} IMPACT_{n} DUR14_{n} T0 P LD_Q1 LD_Q2 LD_U1 LD_U2 LN_WT LN_LIKE'.split()
        df = pd.DataFrame(data_return, columns=labels)
        return df

def _legendre(koi_id, n, k):
        global K_id
        if K_id == False:
            star_id = koi_id.replace("K","S")
        else:
            star_id = koi_id
        
        ttv_file_name = star_id + f'_0{n}_quick.ttvs'
        ttv_file = os.path.join(data_directory, star_id, ttv_file_name)
        lc_file = star_id + '_lc_filtered.fits'
        sc_file = star_id + '_sc_filtered.fits'
        lc_path = os.path.join(data_directory, star_id, lc_file)
        sc_path = os.path.join(data_directory, star_id, sc_file)
        index, ttime, model, out_prob, out_flag = get_ttv_file(star_id,ttv_file)
        if os.path.isfile(lc_path):
            data_lc = read_data_from_fits(lc_path)
        if os.path.isfile(sc_path):
            data_sc = read_data_from_fits(sc_path)
        model= np.array(model, dtype='float64')
        t = model
        #t = t.astype(float)
        #if data_lc.TIME.min()< data_sc.TIME.min() and data_lc.TIME.max()> data_sc.TIME.max():
        x = 2 * (t-data_lc.TIME.min()) / (data_lc.TIME.max() - data_lc.TIME.min()) - 1 
        
        if k==0:
            return np.ones_like(t)
        if k==1:
            return np.zeros_like(t)
        else:
            return ValueError("only configured for 0th and 1st order Legendre polynomials")

