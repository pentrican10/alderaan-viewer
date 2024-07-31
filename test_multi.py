
import batman
import numpy as np
import matplotlib.pyplot as plt, mpld3
from astropy.io import fits
import pandas as pd


file_path = "c:\\Users\\Paige\\Projects\\data\\alderaan_results\\eccentricity-gap\\K00069\\K00069-results.fits"
file_path = "C:\\Users\\Paige\\Projects\\data\\alderaan_results\\eccentricity-gap\\K00084\\K00084_lc_filtered.fits"
# Open the FITS file
# with fits.open(file_path) as hdul:
#     # Print the headers
#     hdul.info()  # This will print a summary of the HDU (Header/Data Unit) list
#     print("\n")

#     # Optionally, print each header in detail
#     for i, hdu in enumerate(hdul):
#         print(f"=== HDU {i} ===")
#         print(hdu.header)
#         print("\n")


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
print(time_loc.values)


'''
params = batman.TransitParams()
params.t0 = 0.                       #time of inferior conjunction
params.per = 1.                      #orbital period
params.rp = 0.1                      #planet radius (in units of stellar radii)
params.b = 0.8                       #impact
params.T14 = .5                      #transit duration
#params.a = 15.                       #semi-major axis (in units of stellar radii)
#params.inc = 87.                     #orbital inclination (in degrees)
#params.ecc = 0.                      #eccentricity
#params.w = 90.                       #longitude of periastron (in degrees)
params.u = [0.1, 0.3]                #limb darkening coefficients [u1, u2]
params.limb_dark = "quadratic"       #limb darkening model
t = np.linspace(-0.05, 0.05, 100)
m = batman.TransitModel(params, t)    #initializes model
flux = m.light_curve(params)          #calculates light curve
plt.plot(t, flux)
plt.xlabel("Time from central transit")
plt.ylabel("Relative flux")
#plt.show()
plt.show()
'''