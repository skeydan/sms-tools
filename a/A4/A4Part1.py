import numpy as np
from scipy.signal import get_window, find_peaks
from scipy.fftpack import fft, fftshift
import math
import matplotlib.pyplot as plt
eps = np.finfo(float).eps

""" 
A4-Part-1: Extracting the main lobe of the spectrum of a window

Write a function that extracts the main lobe of the magnitude spectrum of a window given a window 
type and its length (M). The function should return the samples corresponding to the main lobe in 
decibels (dB).

To compute the spectrum, take the FFT size (N) to be 8 times the window length (N = 8*M) (For this 
part, N need not be a power of 2). 

The input arguments to the function are the window type (window) and the length of the window (M). 
The function should return a numpy array containing the samples corresponding to the main lobe of 
the window. 

In the returned numpy array you should include the samples corresponding to both the local minimas
across the main lobe. 

The possible window types that you can expect as input are rectangular ('boxcar'), 'hamming' or
'blackmanharris'.

NOTE: You can approach this question in two ways: 1) You can write code to find the indices of the 
local minimas across the main lobe. 2) You can manually note down the indices of these local minimas 
by plotting and a visual inspection of the spectrum of the window. If done manually, the indices 
have to be obtained for each possible window types separately (as they differ across different 
window types).

Tip: log10(0) is not well defined, so its a common practice to add a small value such as eps = 1e-16 
to the magnitude spectrum before computing it in dB. This is optional and will not affect your answers. 
If you find it difficult to concatenate the two halves of the main lobe, you can first center the 
spectrum using fftshift() and then compute the indexes of the minimas around the main lobe.


Test case 1: If you run your code using window = 'blackmanharris' and M = 100, the output numpy 
array should contain 65 samples.

Test case 2: If you run your code using window = 'boxcar' and M = 120, the output numpy array 
should contain 17 samples.

Test case 3: If you run your code using window = 'hamming' and M = 256, the output numpy array 
should contain 33 samples.

"""
def extractMainLobe(window, M):
    """
    Input:
            window (string): Window type to be used (Either rectangular ('boxcar'), 'hamming' or '
                blackmanharris')
            M (integer): length of the window to be used
    Output:
            The function should return a numpy array containing the main lobe of the magnitude 
            spectrum of the window in decibels (dB).
    """

    w = get_window(window, M)         # get the window 
    N = 8*M
    
    # The window functions are even sym already.
    # Zero-Phase Padding can be achieved by placing it in the center
    # of the  N sample space of zeros.
   
    hN = (N//2)+1  
    hM1 = (w.size+1)//2  
    hM2 = w.size//2   
    
    fftbuffer = np.zeros(N)     
    fftbuffer[hN-hM2 : hN+hM1] = w 
    
    X = fft(fftbuffer, N)
    
    # After the FFT you can just fftshift and the peak of the lobe will be at N/2.
    shifted = fftshift(X)
 
    mag = abs(shifted)
    
    troughs, _ = find_peaks(-mag)
    lower = troughs[troughs.shape[0]//2 - 1]
    upper = troughs[troughs.shape[0]//2]
    
    selection = mag[lower:(upper + 1)]
    return(20 * np.log10(selection + eps))
    
 
window = "blackmanharris"
M = 100
ret = extractMainLobe(window, M)
ret.shape

# {'input': {'window': 'blackmanharris', 'M': 100}, 'output': array([-290.3819269 ,  -50.47806712,  -38.74635372,  -30.40101914,
#         -23.69098922,  -18.00430867,  -13.04289587,   -8.63518336,
#          -4.67174306,   -1.07772248,    2.20074483,    5.20393943,
#           7.96289674,   10.50210474,   12.84126032,   14.99645696,
#          16.98101381,   18.80607007,   20.48102091,   22.01384356,
#          23.41134527,   24.67935459,   25.82287077,   26.84618153,
#          27.75295669,   28.54632294,   29.22892382,   29.80296767,
#          30.27026589,   30.63226306,   30.89006018,   31.04443184,
#          31.09583819,   31.04443184,   30.89006018,   30.63226306,
#          30.27026589,   29.80296767,   29.22892382,   28.54632294,
#          27.75295669,   26.84618153,   25.82287077,   24.67935459,
#          23.41134527,   22.01384356,   20.48102091,   18.80607007,
#          16.98101381,   14.99645696,   12.84126032,   10.50210474,
#           7.96289674,    5.20393943,    2.20074483,   -1.07772248,
#          -4.67174306,   -8.63518336,  -13.04289587,  -18.00430867,
#         -23.69098922,  -30.40101914,  -38.74635372,  -50.47806712,
#        -290.3819269 ])}

window = "boxcar"
M = 120
ret = extractMainLobe(window, M)
ret.shape

# {'input': {'window': 'boxcar', 'M': 120}, 'output': array([-280.30619538,   24.45801929,   31.12966036,   35.03572162,
#          37.66147543,   39.47244856,   40.67158935,   41.35923592,
#          41.58362492,   41.35923592,   40.67158935,   39.47244856,
#          37.66147543,   35.03572162,   31.12966036,   24.45801929,
#        -280.30619538])}

window = "hamming"
M = 256
ret = extractMainLobe(window, M)
ret.shape

# {'input': {'window': 'hamming', 'M': 256}, 'output': array([-293.03847366,    4.66584113,   13.46067531,   19.35490503,
#          23.88821244,   27.56775781,   30.63189891,   33.21453692,
#          35.39935603,   37.24237174,   38.78280396,   40.04889024,
#          41.06123297,   41.8348303 ,   42.38034882,   42.70492818,
#          42.8126745 ,   42.70492818,   42.38034882,   41.8348303 ,
#          41.06123297,   40.04889024,   38.78280396,   37.24237174,
#          35.39935603,   33.21453692,   30.63189891,   27.56775781,
#          23.88821244,   19.35490503,   13.46067531,    4.66584113,
#        -293.03847366])}
       
