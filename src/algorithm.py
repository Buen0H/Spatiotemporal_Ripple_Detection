'''
Outlier Detection
'''
from numpy import where, repeat, convolve, ones, tile
def detectVerticalLines(X, n=15):
    ''' Vertical Line Detection
    Detect black bars on spatiotemporal image by identifying time epochs that are either
    saturated or zero. The time epoch identified is then extended because outlier affects
    neighbor samples as well.
    INPUTS 
    X   - spatiotemporal image (N, T)
    n   - variable controlling how many nieghboring samples are also ignored
    cLim- (optional) user selected cutoff; a is ignored
    OUTPUT
    mask- binary mask of length T where 1 is an outlier and 0 is not.
    '''
    sigmas = X.std(axis=0)
    realMask = where(sigmas==0, 1, 0)
    extendedMask = convolve(realMask, ones(n), mode='same')
    imgMask = tile(extendedMask, (X.shape[0], 1))
    return imgMask.astype(dtype='bool')

from numpy import zeros_like, median, logical_or, convolve
def detectBubbles(X, a=4, cLim=None, n=3):
    ''' Detect outliers across each spatial row
    Identify time points that are outside of spatial statistics or too close to extrema
    INPUTS
    X   - spatiotemporal image (N, T)
    a   - scaling factor for outlier detection
    cLim- optional (2, ), force thresholds to some values; a is ignored
    n   - spread outlier factor
    '''
    bubbleMask = zeros_like(X)
    for i, x in enumerate(X):
        if cLim:
            tau_low = cLim[0]
            tau_high = cLim[1]
        else:
            mu, sigma = median(x), x.std()
            tau_low = mu - a*sigma; 
            if tau_low < 0: tau_low=sigma
            tau_high = mu + a*sigma; 
            if tau_high > 255: tau_high=255-sigma
        lowMask = x < tau_low
        highMask = x > tau_high
        bubbleMask[i] = logical_or(lowMask, highMask)
        bubbleMask[i] = convolve(bubbleMask[i], ones(n), mode='same')
    return bubbleMask.astype(dtype=bool)

'''
Data cleaning
'''

from scipy.ndimage import median_filter
from numpy import logical_or
def adaptiveFilter(X, size=(1, 100)):
    ''' Adaptive Filter for Spatiotemporal Image
    Filter outliers and replace by local median.
    INPUTS
    X       - spatiotemporal image (N, T)
    size    - size of median filter
    OUTPUTS
    cleanX  - filtered X
    
    '''
    X_medFilt = median_filter(X, size=size)
    outMask = logical_or(detectVerticalLines(X, ), detectBubbles(X))
    return where(outMask, X_medFilt, X)

def removeBiasTime(X):
    ''' Remove constant detrend at each spatial axis
    INPUTS 
    X   - image (N,T)
    OUTPUTS
    X   - image without detrend (N,T)
    
    '''
    return X - X.mean(axis=0) + 255//2

'''
Spatiotemporal Processing
'''
from numpy import array, zeros_like
from scipy.ndimage import sobel, gaussian_filter, gaussian_filter1d
from scipy.spatial.distance import euclidean
from scipy.signal import detrend, find_peaks
def detectClusteredRipples(X, a=2):
    ''' Algorithm for Clustered Ripple Detection.
    The key idea is to extract the energy over time of the spatiotemporal map. First, the spatiotemporal map 
    is cleaned from outliers, and a subset are used for future processing. Secondly, a sobel filter is applied to highlight
    changes along the time axis and followed by a small gaussian filter to smooth the noise. The third step, the spatial axis is
    collapsed to summarize the energy observed at a single time point, while ignoring outliers. The output is smoothed 
    using a gaussian filter with a size that highlights changes the lenght of a clustered ripple. A peak detection
    algorithm is applied to find significant changes in energy during a time epoch.
    INPUT 
    X   - image (N, T)
    a   - threshold for peak detection
    OUTPUT
    X_t - energy over time output
    X_t_filt    - X_t but filtered with a gaussian
    peaks       - output of find_peaks
    properties  - output of find_peaks
    '''
    # Data cleaning
    # X = removeBiasTime(X)
    out_vLines = detectVerticalLines(X)
    out_impulse = detectBubbles(X)

    # Data pre-processing
    X = sobel(X, axis=1)
    X[out_impulse] = 0
    # X = gaussian_filter(X, sigma=1, truncate=10)

    # Time-domain processing
    w = [1/x_space.std() if x_space.any() else 0 for x_space in X]  # collect spatial weights
    X_t = array([euclidean(u, zeros_like(u), w) for u in X.T])      # weighted distance measurement
    X_t[out_vLines[0]] = median(X_t)                                # remove detected outliers in time
    X_t = detrend(X_t, type='linear')                               # remove linear detrend

    # Epoch detection
    mu = X_t.mean(); sigma = X_t.std()   # height threshold parameters
    X_t_filt = gaussian_filter1d(X_t, sigma=15, truncate=6)              # gaussian filter for peak detection      
    peaks, properties = find_peaks(X_t_filt, height=mu+a*sigma, width=20, rel_height=0.75, wlen=250)   
    
    return X_t, X_t_filt, peaks, properties
