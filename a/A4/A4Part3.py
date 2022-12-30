import os
import sys
import numpy as np
from scipy.signal import get_window
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'software/models/'))

import stft
import utilFunctions as UF

eps = np.finfo(float).eps

"""
A4-Part-3: Computing band-wise energy envelopes of a signal

Write a function that computes band-wise energy envelopes of a given audio signal by using the STFT.
Consider two frequency bands for this question, low and high. The low frequency band is the set of 
all the frequencies between 0 and 3000 Hz and the high frequency band is the set of all the 
frequencies between 3000 and 10000 Hz (excluding the boundary frequencies in both the cases). 
At a given frame, the value of the energy envelope of a band can be computed as the sum of squared 
values of all the frequency coefficients in that band. Compute the energy envelopes in decibels. 

Refer to "A4-STFT.pdf" document for further details on computing bandwise energy.

The input arguments to the function are the wav file name including the path (inputFile), window 
type (window), window length (M), FFT size (N) and hop size (H). The function should return a numpy 
array with two columns, where the first column is the energy envelope of the low frequency band and 
the second column is that of the high frequency band.

Use stft.stftAnal() to obtain the STFT magnitude spectrum for all the audio frames. Then compute two 
energy values for each frequency band specified. While calculating frequency bins for each frequency 
band, consider only the bins that are within the specified frequency range. For example, for the low 
frequency band consider only the bins with frequency > 0 Hz and < 3000 Hz (you can use np.where() to 
find those bin indexes). This way we also remove the DC offset in the signal in energy envelope 
computation. The frequency corresponding to the bin index k can be computed as k*fs/N, where fs is 
the sampling rate of the signal.

To get a better understanding of the energy envelope and its characteristics you can plot the envelopes 
together with the spectrogram of the signal. You can use matplotlib plotting library for this purpose. 
To visualize the spectrogram of a signal, a good option is to use colormesh. You can reuse the code in
sms-tools/lectures/4-STFT/plots-code/spectrogram.py. Either overlay the envelopes on the spectrogram 
or plot them in a different subplot. Make sure you use the same range of the x-axis for both the 
spectrogram and the energy envelopes.

NOTE: Running these test cases might take a few seconds depending on your hardware.

Test case 1: Use piano.wav file with window = 'blackman', M = 513, N = 1024 and H = 128 as input. 
The bin indexes of the low frequency band span from 1 to 69 (69 samples) and of the high frequency 
band span from 70 to 232 (163 samples). To numerically compare your output, use loadTestCases.py
script to obtain the expected output.

Test case 2: Use piano.wav file with window = 'blackman', M = 2047, N = 4096 and H = 128 as input. 
The bin indexes of the low frequency band span from 1 to 278 (278 samples) and of the high frequency 
band span from 279 to 928 (650 samples). To numerically compare your output, use loadTestCases.py
script to obtain the expected output.

Test case 3: Use sax-phrase-short.wav file with window = 'hamming', M = 513, N = 2048 and H = 256 as 
input. The bin indexes of the low frequency band span from 1 to 139 (139 samples) and of the high 
frequency band span from 140 to 464 (325 samples). To numerically compare your output, use 
loadTestCases.py script to obtain the expected output.

In addition to comparing results with the expected output, you can also plot your output for these 
test cases.You can clearly notice the sharp attacks and decay of the piano notes for test case 1 
(See figure in the accompanying pdf). You can compare this with the output from test case 2 that 
uses a larger window. You can infer the influence of window size on sharpness of the note attacks 
and discuss it on the forums.
"""
def computeEngEnv(inputFile, window, M, N, H):
    """
    Inputs:
            inputFile (string): input sound file (monophonic with sampling rate of 44100)
            window (string): analysis window type (choice of rectangular, triangular, hanning, 
                hamming, blackman, blackmanharris)
            M (integer): analysis window size (odd positive integer)
            N (integer): FFT size (power of 2, such that N > M)
            H (integer): hop size for the stft computation
    Output:
            The function should return a numpy array engEnv with shape Kx2, K = Number of frames
            containing energy envelop of the signal in decibles (dB) scale
            engEnv[:,0]: Energy envelope in band 0 < f < 3000 Hz (in dB)
            engEnv[:,1]: Energy envelope in band 3000 < f < 10000 Hz (in dB)
    """
    
    fs, x = UF.wavread(inputFile)
   
    w = get_window(window, M, False)   # "symmetric"
    #w = get_window(window, M, True)   
    
    Xdb = stft.stftAnal(x, w, N, H)[0]
    
    # transform to linear scale
    X = 10**(Xdb/20)
    
    # The frequency corresponding to the bin index k can be computed as
    # k/N = f/fs
    # f = k*fs/N
    # k = f * N/fs
    # low: 0 < X < 3000 
    # high: 3000 < X < 10000
    
    freqs_hz = np.arange(N) * fs / N  # f = k*fs/N
    freqs_low = np.intersect1d(np.where(freqs_hz > 0)[0], np.where(freqs_hz <= 3000)[0])
    freqs_high = np.intersect1d(np.where(freqs_hz > 3000)[0], np.where(freqs_hz <= 10000)[0])

    el = np.sum(np.power(X[:, freqs_low], 2), 1) 
    eu = np.sum(np.power(X[:, freqs_high], 2), 1)
    
    # However, once the energy is computed it can be converted back to the dB scale
    # as: EdB = 10 log10 (E)
    el_db = 10 * np.log10(el)
    eu_db = 10 * np.log10(eu)

    engEnv = np.zeros((X.shape[0],2))
    engEnv[:,0] = el_db
    engEnv[:,1] = eu_db
    return(engEnv)
     
inputFile = "../../sounds/piano.wav"
window = "blackman"
M = 513
N = 1024
H = 128
engEnv = computeEngEnv(inputFile, window, M, N, H)
engEnv

# {'input': {'H': 128, 'window': 'blackman', 'N': 1024, 'M': 513, 'inputFile': '../../sounds/piano.wav'},
# 'output': array(
#       [[-49.03116378, -69.13140778],
#        [-47.78351041, -74.82051528],
#        [-47.43848092, -72.93956067],
#        ...,
#        [-46.39025455, -77.83772359],
#        [-46.91766086, -78.20293629],
#        [-53.86249229, -75.91799936]])}

inputFile = "../../sounds/piano.wav"
window = "blackman"
M = 2047
N = 4096
H = 128
engEnv = computeEngEnv(inputFile, window, M, N, H)
engEnv
# 
# {'input': {'H': 128, 'window': 'blackman', 'N': 4096, 'M': 2047, 'inputFile': '../../sounds/piano.wav'},
# 'output': array(
#       [[-46.62640979, -72.34836483],
#        [-43.66160238, -70.71404579],
#        [-41.23971752, -69.23730287],
#        ...,
#        [-47.33078885, -76.45653921],
#        [-48.28372813, -75.79258855],
#        [-49.74139171, -75.33411474]])}
# 
inputFile = "../../sounds/sax-phrase-short.wav"
window = "hamming"
M = 513
N = 2048
H = 256
engEnv = computeEngEnv(inputFile, window, M, N, H)
engEnv
# 
# {'input': {'H': 256, 'window': 'hamming', 'N': 2048, 'M': 513, 'inputFile': '../../sounds/sax-phrase-short.wav'},
#  'output': array(
#       [[-81.52596645, -75.6881054 ],
#        [-69.76052312, -67.51607873],
#        [-66.03531521, -66.72861434],
#        ...,
#        [-66.96154527, -75.28365874],
#        [-67.00463494, -72.86384244],
#        [-67.74045677, -73.60742587]])}
#        
