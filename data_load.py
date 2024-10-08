#from flask import Flask, render_template, jsonify, request, session, redirect, url_for
import os
import csv
from astropy.io import fits
import pandas as pd
import numpy as np
import numpy.polynomial.polynomial as poly


LCIT = 29.4243885           # Kepler long cadence integration time + readout time [min] 
SCIT = 58.848777            # Kepler short cadence integration time + readout time [sec]

lcit = LCIT/60/24           # Kepler long cadence integration time + readout time [days]
scit = SCIT/3600/24         # Kepler short cadence integration time + readout time [days]


k_id = True
table =  ''
data_directory = ''

# Dynamically determine the root directory of the Flask app
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Root directory of the app
# Move one level up from the root directory
PARENT_DIR = os.path.dirname(ROOT_DIR)
# Set the default directory to the parent directory's 'alderaan/Results' path
default_directory = os.path.join(PARENT_DIR, 'alderaan', 'Results')


def update_data_directory(selected_table):
    """
    Function 
    """
    global data_directory
    global default_directory
    data_directory = os.path.join(default_directory, selected_table[:-4])

def read_table_data(table):
    """
    Reads data for the table on the left side of the web app
    Shows koi_id, kep_mag, Rstar, logrho, Teff, logg
    """
    global data_directory
    global K_id
    global Table
    #folder = table[:-4]
    update_data_directory(table)
    Table = table
    if 'SIMULATION' in table:
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

    ### list of Koi IDs with data
    koi_folder_list = [f for f in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, f))]
            
    ### Remove duplicates based on koi_id
    unique_data = []
    seen_koi_ids = set()
    for row in table_data:
        koi_id = row['koi_id']
        
        ### Check if koi_id is not in the set of seen ids
        if koi_id not in seen_koi_ids:
            ### only add the row to table if there is a file for the Koi ID
            if koi_id in koi_folder_list:
                ### Add the row to unique_data and the koi_id to the set
                unique_data.append(row)
                seen_koi_ids.add(koi_id)


    return unique_data

def get_planet_properties_table(koi_id,table):
    '''
    Function retrieves relevant planet properties and passes as table info

    args:
        koi_id: string in the form "K00000" (KOI identification)
        table: string in form of 'table.csv'
    
    returns:
        planet_data: table with planet properties sorted by ascending period (name, period, lcit ratio, impact, ror, duration)
    '''
    global K_id
    if K_id==False:
        star_id = koi_id.replace("K","S")
    else:
        star_id = koi_id
    file_path_csv = os.path.join(data_directory, table)
    file_results =star_id + '-results.fits'
    file_path_results = os.path.join(data_directory, star_id, file_results)
    data_id = get_koi_identifiers(file_path_csv,koi_id)
    data_id = data_id.sort_values(by='periods') 
    koi_identifier = data_id.koi_identifiers.values
    planet_data = []
    with open(file_path_csv, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        n=0
        for row in reader:
            
            if row['koi_id'] == koi_id:
                row['planet_name'] = koi_identifier[n]

                data_post = load_posteriors(file_path_results,n,koi_id)
                row['period'] = (data_post['P'].median())
                row['lcit_ratio'] = round(row['period'] / lcit,5 )
                row['impact'] = round(data_post[f'IMPACT_{n}'].median(),4)
                row['ror'] = round(data_post[f'ROR_{n}'].median(),4)
                row['duration'] = round(data_post[f'DUR14_{n}'].median(),4) 
                n+=1
                planet_data.append(row) 
    planet_data.sort(key=lambda x: x['period']) 
    return planet_data

            

def get_koi_identifiers(file_path, koi_id):
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

               
def load_photometry_data(file_path):
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


def load_ttv_data(koi_id, file_path):
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
        ### convert to arrays
        index = np.asarray(index, dtype=np.int64)
        model = np.asarray(model, dtype=np.float64)
        ttime = np.asarray(ttime, dtype=np.float64)
        model = np.asarray(model, dtype=np.float64)
        out_prob = np.asarray(out_prob, dtype=np.float64)
        out_flag = np.asarray(out_flag, dtype=np.float64)
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
        photometry_data_lc = load_photometry_data(file_path_lc) 
        photometry_data_sc = load_photometry_data(file_path_sc)
        lc_max = photometry_data_lc['FLUX'].max()
        lc_min = photometry_data_lc['FLUX'].min()

        sc_max = photometry_data_sc['FLUX'].max()
        sc_min = photometry_data_sc['FLUX'].min()
        return lc_min,lc_max,sc_min,sc_max
    elif os.path.isfile(file_path_lc) and not os.path.isfile(file_path_sc):
        photometry_data_lc = load_photometry_data(file_path_lc) 
        lc_max = photometry_data_lc['FLUX'].max()
        lc_min = photometry_data_lc['FLUX'].min()
        return lc_min, lc_max
    elif os.path.isfile(file_path_sc) and not os.path.isfile(file_path_lc):
        photometry_data_sc = load_photometry_data(file_path_sc)
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
    data_post = data_post.sort_values(by='LN_LIKE', ascending=False) 
    row = data_post.iloc[0] # pick row with highest likelihood
    ### mult by 1.5 for correct offset
    DUR14 = row[f'DUR14_{num}']

    combined_data = None
    #get data and create detrended light curve
    if os.path.isfile(file_path_lc) and os.path.isfile(file_path_sc):
        photometry_data_lc = load_photometry_data(file_path_lc) 
        photometry_data_sc = load_photometry_data(file_path_sc)
        index, ttime, model, out_prob, out_flag = load_ttv_data(koi_id, file_path)

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
        photometry_data_lc = load_photometry_data(file_path_lc) #descriptive names
        index, ttime, model, out_prob, out_flag = load_ttv_data(koi_id, file_path)

        if line_number < len(index):
            center_time = ttime[line_number]
            transit_number = index[line_number]
        
            start_time = float(center_time) - (DUR14*1.5) 
            end_time= float(center_time) + (DUR14*1.5) 

            use_lc = (photometry_data_lc['TIME'] > start_time) & (photometry_data_lc['TIME'] < end_time)
            lc_data = photometry_data_lc[use_lc]
            combined_data = lc_data
            sc_data = 0#None
            #sc_data.TIME = 0
        else:
            lc_data = photometry_data_lc
            lc_data.TIME = 0
            sc_data = 2
            transit_number = None
            center_time = None
        return lc_data,sc_data, transit_number, center_time 
    
    elif os.path.isfile(file_path_sc):
        photometry_data_sc = load_photometry_data(file_path_sc)
        index, ttime, model, out_prob, out_flag = load_ttv_data(koi_id, file_path)

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
    

def folded_data(koi_id,planet_num, file_path,overlap):
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
    data_post = data_post.sort_values(by='LN_LIKE', ascending=False) 
    row = data_post.iloc[0] # pick row with highest likelihood
    ### mult by 1.5 for correct offset
    DUR14 = row[f'DUR14_{planet_num}']

    fold_data_time_lc = [] 
    fold_data_flux_lc = []
    fold_data_err_lc = []
    fold_data_time_sc = []
    fold_data_flux_sc = []
    fold_data_err_sc = []
    #get data and create detrended light curve
    if os.path.isfile(file_path_lc):
        photometry_data_lc = load_photometry_data(file_path_lc)
        # index, ttime, model, out_prob, out_flag = load_ttv_data(koi_id, file_path)
        # index = index[~overlap]
        # model = model[~overlap]                        # revisit and ensure using same ttime and model
        results_data = load_results_model(file_path_results,planet_num)
        
        ### revieved error about endian: ValueError: Big-endian buffer not supported on little-endian compiler
        ### Convert overlap to the correct endianness before applying the mask
        overlap = overlap.astype(np.bool_).astype(overlap.dtype.newbyteorder('='))
        ### Convert results_data to native byte order 
        for col in results_data.columns:
            if results_data[col].dtype.byteorder == '>':
                results_data[col] = results_data[col].values.astype(results_data[col].dtype.newbyteorder('='))

        
        ### mask out overlapping transits
        results_data = results_data[~overlap]
        model = np.array(results_data.model)
        index = np.array(results_data.index) 
        
        for i in range(len(index)):
            # center_time = ttime[i]
            center_time = float(model[i])
        
            start_time = (center_time) - (DUR14*1.5) 
            end_time= (center_time) + (DUR14*1.5)

            use = (photometry_data_lc['TIME'] > start_time) & (photometry_data_lc['TIME'] < end_time)
            transit_data = photometry_data_lc[use]

            folded_transit_time_lc = transit_data['TIME'] - center_time
            fold_data_time_lc.extend(folded_transit_time_lc)
            fold_data_flux_lc.extend(transit_data['FLUX'])
            fold_data_err_lc.extend(transit_data['ERR'])

    if os.path.isfile(file_path_sc):
        photometry_data_sc = load_photometry_data(file_path_sc)
        for i in range(len(index)):
            center_time = float(model[i])
        
            start_time = center_time - DUR14
            end_time= center_time + DUR14

            use_sc = (photometry_data_sc['TIME']>start_time) & (photometry_data_sc['TIME']<end_time)
            transit_data_sc = photometry_data_sc[use_sc]

            folded_transit_time_sc = transit_data_sc['TIME'] - center_time
            fold_data_time_sc.extend(folded_transit_time_sc)
            fold_data_flux_sc.extend(transit_data_sc['FLUX'])
            fold_data_err_sc.extend(transit_data_sc['ERR'])
  
    fold_data_lc = pd.DataFrame({
        'TIME' : fold_data_time_lc,
        'FLUX': fold_data_flux_lc,
        'ERR' : fold_data_err_lc
    })

    fold_data_sc = pd.DataFrame({
        'TIME' : fold_data_time_sc,
        'FLUX': fold_data_flux_sc,
        'ERR' : fold_data_err_sc
    })

    bin_size = DUR14/11 #0.02
    ### set so 11 bin points in transit, make sure it transfers to the exp time
    # DUR / 11
    combined_df = pd.concat([fold_data_lc, fold_data_sc], ignore_index=True) #EXCEPTION IF SC DATA, CODE NOT WRITTEN
    #combined_df = combined_df.sort_values(by='TIME', ascending=True)
    fold_time = np.array(combined_df.TIME)
    fold_flux = np.array(combined_df.FLUX)
    if len(fold_time)>1: 
        binned_centers, binned_data = bin_data(fold_time, fold_flux, bin_size)

        # Create DataFrame for combined binned weighted average data 
        binned_weighted_avg_combined = pd.DataFrame({
            'TIME': binned_centers,
            'FLUX': binned_data
        })
    else:
        binned_centers = [0]
        binned_data =[0]
        binned_weighted_avg_combined = pd.DataFrame({
            'TIME': binned_centers, 
            'FLUX': binned_data 
        })

    return fold_data_lc, fold_data_sc, binned_weighted_avg_combined

# Binned weighted average function
'''
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
'''


### FIX FOR SC DATA, ACCOUNT FOR ERRORS
def bin_data(time, data, binsize):
    """
    Parameters
    ----------
    time : ndarray
        vector of time values
    data : ndarray
        corresponding vector of data values to be binned
    binsize : float
        bin size for output data, in same units as time
        
    Returns
    -------
    bin_centers : ndarray
        center of each data (i.e. binned time)
    binned_data : ndarray
        data binned to selcted binsize
    """
    bin_centers = np.hstack([np.arange(time.mean(),time.min()-binsize/2,-binsize),
                            np.arange(time.mean(),time.max()+binsize/2,binsize)])
    
    bin_centers = np.sort(np.unique(bin_centers))
    binned_data = []
    
    for i, t0 in enumerate(bin_centers):
        binned_data.append(np.mean(data[np.abs(time-t0) < binsize/2]))
        
    return bin_centers, np.array(binned_data)


    
def load_OMC_data(koi_id,file_path):
    index, ttime, model, out_prob, out_flag = load_ttv_data(koi_id, file_path)
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

        index, ttime, model, out_prob, out_flag = load_ttv_data(koi_id,ttv_file)
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

        ### change to unweighted
        N_samp = 1000
        LN_WT = df['LN_WT'].values
        weight = np.exp(LN_WT- LN_WT.max())
        w = weight/ np.sum(weight)
        df = df.sample(N_samp, replace=True, ignore_index=True, weights=w)

        return df
    
def load_results_model(file_path_results,planet_num):
    with fits.open(file_path_results) as hdul:
        hdu2 = hdul[2 + planet_num]
        data = hdu2.data
        index = data['INDEX']
        ttime = data['TTIME']
        model = data['MODEL']
        out_prob = data['OUT_PROB']
        out_flag = data['OUT_FLAG']

    df = pd.DataFrame(dict(
            index=index,
            ttime=ttime,
            model = model,
            out_prob = out_prob,
            out_flag = out_flag
        ))
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
        index, ttime, model, out_prob, out_flag = load_ttv_data(star_id,ttv_file)
        if os.path.isfile(lc_path):
            data_lc = load_photometry_data(lc_path)
        if os.path.isfile(sc_path):
            data_sc = load_photometry_data(sc_path)
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

