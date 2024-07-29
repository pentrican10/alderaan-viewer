
import batman
import numpy as np
import matplotlib.pyplot as plt, mpld3
from astropy.io import fits


file_path = "c:\\Users\\Paige\\Projects\\data\\alderaan_results\\eccentricity-gap\\K00069\\K00069-results.fits"
# Open the FITS file
with fits.open(file_path) as hdul:
    # Print the headers
    hdul.info()  # This will print a summary of the HDU (Header/Data Unit) list
    print("\n")

    # Optionally, print each header in detail
    for i, hdu in enumerate(hdul):
        print(f"=== HDU {i} ===")
        print(hdu.header)
        print("\n")


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