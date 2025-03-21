import numpy as np

def fold_data(time, flux, tts, duration):
    t_folded = []
    f_folded = []
    
    for t0 in tts:
        use = np.abs(time - t0)/duration < 1.5
        t_folded.append(time[use]-t0)
        f_folded.append(flux[use])
        
    t_folded = np.hstack(t_folded)
    f_folded = np.hstack(f_folded)

    return t_folded, f_folded
    
    
def bin_data(time, data, binsize, bin_centers=None):
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
    if bin_centers is None:
        bin_centers = np.hstack([np.arange(time.mean(),time.min()-binsize/2,-binsize)[::-1],
                                 np.arange(time.mean(),time.max()+binsize/2,binsize)[1:]])
        
    binned_data = np.zeros(len(bin_centers))
    for i, t0 in enumerate(bin_centers):
        binned_data[i] = np.mean(data[np.abs(time-t0) < binsize/2])
        
    return bin_centers, binned_data