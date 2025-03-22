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

K_id = True

### Dynamically determine the directory structure
### The viewer app should be placed in "<ROOT_DIR>/alderaan-viewer"
### Pipeline outputs should be stored in "<ROOT_DIR>/alderaan/Results/<RUN_ID>"
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEWER_DIR = os.path.join(ROOT_DIR, 'alderaan-viewer')
RESULTS_DIR = os.path.join(ROOT_DIR, 'alderaan', 'Results')


### TODO rather than tracking run_dir and table, simply track run_id
RUN_DIR = ''
table = ''


def read_star_properties_table(table):
    """
    Reads data for the table on the left side of the web app
    Shows koi_id, kep_mag, Rstar, logrho, Teff, logg
    """
    global RUN_DIR
    global K_id
    
    RUN_DIR = os.path.join(RESULTS_DIR, table[:-4])
    
    if 'SIMULATION' in table:
        K_id = False 
    else: 
        K_id = True
        
    file_path = os.path.join(RUN_DIR, table)
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
            ### Round stellar property values
            row['kep_mag'] = round(float(row['kep_mag']), 2)
            row['Rstar'] = round(float(row['Rstar']), 2)
            row['logrho'] = round(float(row['logrho']), 2)
            row['Teff'] = round(float(row['Teff']))
            row['logg'] = round(float(row['logg']), 2)
            
            ### Ensure 'review' column exists in each row
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
    koi_folder_list = [f for f in os.listdir(RUN_DIR) if
                       os.path.isdir(os.path.join(RUN_DIR, f))
                      ]
            
    ### Remove duplicates based on koi_id
    unique_rows = []
    seen_koi_ids = set()
    for row in table_data:
        koi_id = row['koi_id']
        
        if koi_id not in seen_koi_ids:
            if koi_id in koi_folder_list:
                unique_rows.append(row)
                seen_koi_ids.add(koi_id)

    return unique_rows



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
    file_path_csv = os.path.join(RUN_DIR, table)
    file_results =star_id + '-results.fits'
    file_path_results = os.path.join(RUN_DIR, star_id, file_results)
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
                row['duration'] = round((data_post[f'DUR14_{n}'].median())*24,4) 
                n+=1
                planet_data.append(row) 
    planet_data.sort(key=lambda x: x['period']) 
    return planet_data


def get_period_ratios_table(koi_id,table):
    '''
    Function retrieves relevant period ratios and passes as table info

    args:
        koi_id: string in the form "K00000" (KOI identification)
        table: string in form of 'table.csv'
    
    returns:
        planet_data: table with period ratios sorted by ascending period (name, period)
    '''

    global K_id
    if K_id==False:
        star_id = koi_id.replace("K","S")
    else:
        star_id = koi_id
    file_path_csv = os.path.join(RUN_DIR, table)
    file_results =star_id + '-results.fits'
    file_path_results = os.path.join(RUN_DIR, star_id, file_results)
    data_id = get_koi_identifiers(file_path_csv,koi_id)
    ### sort by period
    data_id = data_id.sort_values(by='periods') 
    koi_identifier = data_id.koi_identifiers.values

    periods = []
    npl = 1
    # Retrieve periods for the planets from the CSV file
    with open(file_path_csv, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        n = 0
        for row in reader:
            if row['koi_id'] == koi_id:
                # Load the period for the current planet
                data_post = load_posteriors(file_path_results, n, koi_id)
                period = data_post['P'].median()

                # Append planet name and period to the list
                periods.append({'planet_name': row['planet_name'], 'period': period})
                npl += n
                n += 1

    # Sort the periods in ascending order
    periods = sorted(periods, key=lambda x: x['period'])

    if npl > 1:

        # Calculate ratios for consecutive periods
        ratios = []
        for i in range(len(periods) - 1):
            period1 = round(periods[i+1]['period'], 4)
            period2 = round(periods[i]['period'], 4)
            ratio = round(periods[i+1]['period'] / periods[i]['period'], 3)

            ratio = {
                'planets': f"{periods[i+1]['planet_name']} / {periods[i]['planet_name']}",
                'periods': f"{period1} / {period2}",
                'ratio': ratio
            }
            ratios.append(ratio)
    else:
        # Calculate ratios for consecutive periods
        ratios = []
        ratio = {
                'planets': f"Single Planet System",
                'periods': f"Single Planet System",
                'ratio': f"Single Planet System"
            }
        ratios.append(ratio)

    # Return as a JSON-friendly list of dictionaries
    return ratios
   

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


def get_num_planets(file_path_results):
    with fits.open(file_path_results) as hdul:
        npl = int(hdul[0].header['NPL'])
        
    return npl
    
            
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
    

def load_ttv_data_from_results(file_path_results,planet_num):
    with fits.open(file_path_results) as hdul:
        data = hdul[2+planet_num].data
        index = np.array(data['INDEX'], dtype='int')
        ttime = np.array(data['TTIME'], dtype='float')
        model = np.array(data['MODEL'], dtype='float')
        out_prob = np.array(data['OUT_PROB'], dtype='float')
        out_flag = np.array(data['OUT_FLAG'], dtype='bool')

    df = pd.DataFrame(dict(
            index=index,
            ttime=ttime,
            model = model,
            out_prob = out_prob,
            out_flag = out_flag
        ))
    
    # index field in ALDERAAN.results.fits conflicts with index in pandas.df
    df.index = index
    
    return df


def get_min_max(koi_id):
    global K_id
    if K_id == False:
        star_id = koi_id.replace("K","S")
    else:
        star_id = koi_id
    file_name_lc = star_id + '_lc_filtered.fits'
    file_path_lc = os.path.join(RUN_DIR,star_id,file_name_lc)
    
    file_name_sc = star_id + '_sc_filtered.fits'
    file_path_sc = os.path.join(RUN_DIR, star_id, file_name_sc)

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
    file_path_lc = os.path.join(RUN_DIR,star_id,file_name_lc)
    
    file_name_sc = star_id + '_sc_filtered.fits'
    file_path_sc = os.path.join(RUN_DIR, star_id, file_name_sc)

    file_path = os.path.join(RUN_DIR, star_id, ttv_file)

    file_results =star_id + '-results.fits'
    file_path_results = os.path.join(RUN_DIR, star_id, file_results)
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
    ttv_file = os.path.join(RUN_DIR, star_id, file_name)
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
        
        
        
def load_posteriors_from_results(file_path_results, planet_num):
    n = planet_num
    
    with fits.open(file_path_results) as hdul:
        data = hdul['SAMPLES'].data
        
        ### Extract data and sanitize
        C0 = np.array(data[f'C0_{n}'], dtype='float')
        C1 = np.array(data[f'C1_{n}'], dtype='float')
        ROR = np.array(data[f'ROR_{n}'], dtype='float')
        IMPACT = np.array(data[f'IMPACT_{n}'], dtype='float')
        DUR14 = np.array(data[f'DUR14_{n}'], dtype='float')
        LD_Q1 = np.array(data[f'LD_Q1'], dtype='float')
        LD_Q2 = np.array(data[f'LD_Q2'], dtype='float')
        LN_WT = np.array(data[f'LN_WT'], dtype='float')
        LN_Z = np.array(data[f'LN_Z'], dtype='float')
        LN_LIKE = np.array(data[f'LN_LIKE'], dtype='float')

        ### Calculate u1, u2
        LD_U1 = 2*np.sqrt(LD_Q1)*LD_Q2
        LD_U2 = np.sqrt(LD_Q1)*(1-2*LD_Q2)

        ### Calculate P, t0 from transit times
        ttv_data = load_ttv_data_from_results(file_path_results, n)
        centered_index = np.array(ttv_data.index - ttv_data.index[-1]) // 2
        LegX = centered_index / np.array(ttv_data.index[-1]/2)
        Leg0 = np.ones_like(LegX)
        ephem = np.array(ttv_data.model) + np.outer(C0, Leg0) + np.outer(C1,LegX)
        T0, P = poly.polyfit(ttv_data.index, ephem.T, 1)

    data = np.vstack([C0, C1, ROR, IMPACT, DUR14, T0, P, LD_Q1, LD_Q2, LD_U1, LD_U2, LN_WT, LN_Z, LN_LIKE]).T
    labels = f'C0_{n} C1_{n} ROR_{n} IMPACT_{n} DUR14_{n} T0 P LD_Q1 LD_Q2 LD_U1 LD_U2 LN_WT LN_Z LN_LIKE'.split()
    df = pd.DataFrame(data, columns=labels)

    ### Resample into unweighted arrays
    nsamp = 1000
    wt = np.exp(df['LN_WT'].values - df['LN_Z'].values.max())
    wt = wt / np.sum(wt)
    df = df.sample(nsamp, replace=True, ignore_index=True, weights=wt)

    return df
        
        
        
    


def _legendre(koi_id, n, k):
        global K_id
        if K_id == False:
            star_id = koi_id.replace("K","S")
        else:
            star_id = koi_id
        
        ttv_file_name = star_id + f'_0{n}_quick.ttvs'
        ttv_file = os.path.join(RUN_DIR, star_id, ttv_file_name)
        lc_file = star_id + '_lc_filtered.fits'
        sc_file = star_id + '_sc_filtered.fits'
        lc_path = os.path.join(RUN_DIR, star_id, lc_file)
        sc_path = os.path.join(RUN_DIR, star_id, sc_file)
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

