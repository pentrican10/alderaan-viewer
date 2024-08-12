
import batman
import numpy as np
import matplotlib.pyplot as plt, mpld3
from astropy.io import fits
import pandas as pd
#from data_load import calculate_binned_weighted_average, folded_data
import os
import numpy.polynomial.polynomial as poly




cadence_type = 's' # l is long cadence, s is short cadence
koi_id = 'K01884'
planet_num = 0
file_path_results = f"c:\\Users\\Paige\\Projects\\data\\alderaan_results\\eccentricity-gap\\{koi_id}\\{koi_id}-results.fits"
file_path = f"C:\\Users\\Paige\\Projects\\data\\alderaan_results\\eccentricity-gap\\{koi_id}\\{koi_id}_{cadence_type}c_filtered.fits"
# Open the FITS file
with fits.open(file_path_results) as hdul:
    hdu2 = hdul[2]
    data = hdu2.data
    index = data['INDEX']
    ttime = data['TTIME']
    model = data['MODEL']
    out_prob = data['OUT_PROB']
    out_flag = data['OUT_FLAG']
    

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
# print(df.QUARTER)
QUARTER = df.QUARTER
# print(df[QUARTER == 16])
# print(QUARTER.max())
# Filter for quarter 16 and print associated times
quarter_16_times = df.loc[df['QUARTER'] == 16, 'TIME']
# print("Times associated with quarter 16:")
# print(quarter_16_times)
# print(quarter_16_times.min())
# print(quarter_16_times.max())

time_loc = df.loc[df['TIME'] == df.TIME.min(), 'QUARTER']
#print(time_loc.values)
data_directory = 'c:\\Users\\Paige\\Projects\\data\\alderaan_results\\eccentricity-gap'


def load_posteriors(f,n,koi_id):
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

        #index, ttime, model, out_prob, out_flag = get_ttv_file(koi_id,ttv_file)
        # Leg0 = _legendre(koi_id,n,0)
        # Leg1 = _legendre(koi_id,n,1)
        with fits.open(file_path_results) as hdul:
            hdu2 = hdul[2]
            data = hdu2.data
            index = data['INDEX']
            ttime = data['TTIME']
            model = data['MODEL']
            out_prob = data['OUT_PROB']
            out_flag = data['OUT_FLAG']
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


def folded_data(koi_id,planet_num):
    star_id = koi_id
    global cadence_type
    file_name_lc = star_id + f'_{cadence_type}c_filtered.fits'
    file_path_lc = os.path.join(data_directory, star_id, file_name_lc)
    
    
    file_results =star_id + '-results.fits'
    file_path_results = os.path.join(data_directory, star_id, file_results)
    data_post = load_posteriors(file_path_results,planet_num,koi_id)
    ### get max likelihood
    data_post = data_post.sort_values(by='LN_LIKE', ascending=False) 
    row = data_post.iloc[0] # pick row with highest likelihood
    ### mult by 1.5 for correct offset
    # DUR14 = 1.5 * data_post[f'DUR14_{planet_num}'][max_index]
    DUR14 = 1.5 * row[f'DUR14_{planet_num}']

    fold_data_time = []
    fold_data_flux = []
    fold_data_err = []
    
    #get data and create detrended light curve
    if os.path.isfile(file_path_lc):
        photometry_data_lc = df
        #index, ttime, model, out_prob, out_flag = get_ttv_file(koi_id, file_path)
        
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

    fold_data_lc = pd.DataFrame({
        'TIME' : fold_data_time,
        'FLUX': fold_data_flux,
        'ERR' : fold_data_err
    })
  
    bin_size = 0.01
    combined_time = np.concatenate([fold_data_lc['TIME']])
    combined_flux = np.concatenate([fold_data_lc['FLUX']])
    combined_flux_err = np.concatenate([fold_data_lc['ERR']]) 

    bin_centers_combined, weighted_avg_combined = calculate_binned_weighted_average(combined_time, combined_flux, combined_flux_err, bin_size)

    # Create DataFrame for combined binned weighted average data
    binned_weighted_avg_combined = pd.DataFrame({
        'TIME': bin_centers_combined,
        'FLUX': weighted_avg_combined
    })

    return fold_data_lc, binned_weighted_avg_combined, center_time


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


fold_data, binned_weighted_avg, center_time = folded_data(koi_id, 0)

data_post = load_posteriors(file_path_results,planet_num,koi_id)
### get max likelihood
max_index = data_post['LN_LIKE'].idxmax()
data_post = data_post.sort_values(by='LN_LIKE', ascending=False) 
row = data_post.iloc[0] # pick row with highest likelihood
### get most likely params {P, t0, Rp/Rs, b, T14, q1, q2}
theta = batman.TransitParams()
theta.per = row[f'P']
theta.t0 = 0.
theta.rp = row[f'ROR_{planet_num}']
theta.b = row[f'IMPACT_{planet_num}']
theta.T14 = row[f'DUR14_{planet_num}']#*24
LD_U1 = row[f'LD_U1']
LD_U2 = row[f'LD_U2']
theta.u = [LD_U1, LD_U2]
theta.limb_dark = 'quadratic'

scit = 1.15e-5
t = np.arange(fold_data.TIME.min(), fold_data.TIME.max(),scit)
m = batman.TransitModel(theta, t)    #initializes model
flux = (m.light_curve(theta))        #calculates light curve

# plt.scatter(fold_data.TIME, fold_data.FLUX, s=2,label='data')
# plt.scatter(binned_weighted_avg.TIME, binned_weighted_avg.FLUX, s=10,marker='s', label='binned')
# plt.plot(t,flux, color='red', label='model')
# plt.title(f'KOI ID: {koi_id}, Data: Long Cadence')
# plt.legend()
# plt.show()
# Calculate residuals
model_flux = np.interp(fold_data['TIME'], t, flux)
model_flux_bin = np.interp(binned_weighted_avg['TIME'], t, flux)
residuals = fold_data['FLUX'] - model_flux
residual_bin = binned_weighted_avg['FLUX'] - model_flux_bin

# Create subplots
fig, axs = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [6, 2]}, sharex=True)

# Plot the original data and model

if cadence_type == 'l':
    axs[0].scatter(fold_data['TIME'], fold_data['FLUX'], s=2,color='blue', label='lc data')
    axs[0].scatter(binned_weighted_avg['TIME'], binned_weighted_avg['FLUX'], s=10,color='orange', marker='s', label='binned')
    axs[0].plot(t, flux, color='red', label='model')
    axs[0].set_title(f'KOI ID: {koi_id}, Data: Long Cadence')
    axs[0].legend()
    axs[0].grid(True)

    # Plot the residuals
    axs[1].scatter(fold_data['TIME'], residuals, s=2, color='blue', label='Residuals data')
    axs[1].scatter(binned_weighted_avg['TIME'], residual_bin, s=10,marker='s', color='orange', label='Residuals bin')
    axs[1].hlines(0, min(fold_data['TIME']), max(fold_data['TIME']), color='red', label='Zero Residual Line')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Residuals')
    axs[1].set_title('Residuals of the Model')
    #axs[1].legend()
    axs[1].grid(True)
else:
    axs[0].scatter(fold_data['TIME'], fold_data['FLUX'], s=2,color='gray', label='sc data')
    axs[0].scatter(binned_weighted_avg['TIME'], binned_weighted_avg['FLUX'], s=10,color='orange', marker='s', label='binned')
    axs[0].plot(t, flux, color='red', label='model')
    axs[0].set_title(f'KOI ID: {koi_id}, Data: Short Cadence')
    axs[0].legend()
    axs[0].grid(True)

    # Plot the residuals
    axs[1].scatter(fold_data['TIME'], residuals, s=2, color='gray', label=' scResiduals data')
    axs[1].scatter(binned_weighted_avg['TIME'], residual_bin, s=10,marker='s', color='orange', label='Residuals bin')
    axs[1].hlines(0, min(fold_data['TIME']), max(fold_data['TIME']), color='red', label='Zero Residual Line')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Residuals')
    axs[1].set_title('Residuals of the Model')
    #axs[1].legend()
    axs[1].grid(True)




plt.tight_layout()  # Adjusts subplots to fit into the figure area.
plt.show()
