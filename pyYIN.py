# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:00:39 2018

@author: Austin Mackillop

YIN Algorithm
"""
from numpy import mean, correlate, asarray, zeros, polyfit, poly1d, float32, int16, frombuffer
from numpy import sum as npsum
from scipy.signal import butter, lfilter
from wave import open as wavopen
from array import array

def getData(fname):
    """
    Get the signal information from a wav file.
    
    # Arguments
        fname: WAV file. Name of the audio file to analyze

    # Returns
        data: A 1-D int16 numpy array containing the information

    # Raises
        None
    """
     #Extract Raw Audio from Wav File
    sound_file = wavopen(fname, 'r')
    data = sound_file.readframes(-1)
    data = frombuffer(data, int16)
    sound_file.close()
    return data

def trim(signal, threshold):
    """
    Trim the silence at the beginning and the end of the signal.
    
    # Arguments
        signal: A 1-D int16 numpy array containing the signal\n
        threshold: A volume threshold for silence detection

    # Returns
        signal: A 1-D int16 numpy array containing the trimmed signal

    # Raises
        None
    """
    # This nested function allows for a simple way to run this function twice for
    # the beginning and the end.
    def _trim(x):
        snd_started = False
        arr = array('h')

        for i in signal:
            if not snd_started and abs(i) > threshold:
                snd_started = True
                arr.append(i)

            elif snd_started:
                arr.append(i)
        return arr

    # Trim the beginning
    signal = _trim(signal)

    # Trim the end
    signal.reverse()
    signal = _trim(signal)
    signal.reverse()
    signal = asarray(signal, int16)
    
    return signal

def butterLowpassFilter(signal, cutoff, Fs, order = 6):
    """
    Implementing a buttersworth lowpass filter
    
    # Arguments
        signal: A 1-D numpy array containing the signal \n
        cutoff: Cutoff frequency for the lowpass filter \n
        Fs: Sampling rate of the signal \n
        order: Order of the lowpass filter

    # Returns
        None

    # Raises
        None
    """
    nyq = 0.5 * Fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    signal = lfilter(b, a, signal)
    return signal.astype(int16)

def downSample(signal, cutoff, Fs, step):
    """
    Downsample the signal by a specified integer amount.
    
    # Arguments
        signal: A 1-D numpy array containing the signal \n
        cutoff: Cutoff frequency for the lowpass filter \n
        Fs: Sampling rate of the original signal \n
        order: Order of the buttersworth lowpass filter used
        step: Integer specificying the downsample rate. a step of 2 means throwing away
        every other sample with halves the sampling rate.

    # Returns
        signal: A 1-D numpy array containing the downsampled signal.

    # Raises
        None
    """
    butterLowpassFilter(signal, cutoff, Fs, 40)
    signal = signal[0::step]
    return signal

def crossCorr(x, tau, W, auto = False):
    """
    Autocorrelation, Step 1, Eq. (1)\n
    
    Computes the cross correlation between a signal and a shifted version. If auto = True, 
    computes the auto correlation of the signal shifted by tau.
    
    # Arguments
        x: A 1-D numpy array containing the signal\n
        tau: Integer sample lag
        W: Integer integration window size.
        auto: Boolean, set True for autocorrelation
        
    # Returns
        cross_corr_mat: A 2-D numpy array of the correlation function for each sample. 
        Each row corresponds to a sample.
    """
        
    cross_corr_mat = zeros((x.size//W, W), float32)
    x_orig = list(x)
    for i in range(cross_corr_mat.shape[0]):
        t = i*W
        # Unbias the signals
        x = x_orig[t:W+t] - mean(x_orig[t:W+t])
        x_tau = x_orig[t+tau:t+tau+W] - mean(x_orig[t+tau:t+tau+W])
        if (auto == False):
            cross_corr = correlate(x, x_tau, 'full')#/npsum(x**2)
            cross_corr_mat[i,:] = cross_corr[cross_corr.shape[0]//2:]
        else:
            cross_corr = correlate(x_tau, x_tau, 'full')
            cross_corr_mat[i,:] = cross_corr[cross_corr.shape[0]//2:]
    return cross_corr_mat


def diffEquation(x, W):
    """
    Difference Equation, Step 2, Eq. (7)\n
    
    Computes the difference equation for each sample
    
    # Arguments
        x: A 1-D numpy array containing the signal\n
        W: Integration window size
        
    # Returns
        diff_eq_mat: A 2-D numpy array of the computed difference equations for each sample.
        Each row corresponds to a sample.
    """
    
#    cross_corr_lag = crossCorr(x, 0, W, auto = True)
#    if not tau%2:
    auto_corr_mat = crossCorr(x, 0, W, auto = True)
    diff_eq_mat = zeros(auto_corr_mat.shape, float32)
    for i in range(0, diff_eq_mat.shape[0]):
        diff_eq_mat[i,:] = auto_corr_mat[i, 0] + crossCorr(x, i, W, auto = True)[i, 0] - 2 * auto_corr_mat[i,:]
    return diff_eq_mat

def cumMeanNormDiffEq(x, W):
    """
    Cumulative Mean Normal Difference Equation, Step 3, Eq. (8)\n
    
    Computes the cumulative mean normal difference equation for each sample
    
    # Arguments
        x: A 1-D numpy array containing the signal\n
        W: Integration window size
        
    # Returns
        cum_diff_mat: A 2-D numpy array of the computed difference equations
        for each sample. Each row corresponds to a sample.
    """
    diff_eq_mat = diffEquation(x, W)
    cum_diff_mat = zeros(diff_eq_mat.shape, float32)
    cum_diff_mat[:, 0] = 1
    for t in range(diff_eq_mat.shape[0]):
        for j in range(1, diff_eq_mat.shape[1]):
            cum_diff_mat[t, j] = diff_eq_mat[t, j]/((1/j)*npsum(diff_eq_mat[t, 1:j+1]))
    return cum_diff_mat
        
def absoluteThresold(x, freq_range = (40, 300), threshold = 0.1, Fs = 16e3):
    """
    Absolute Threshold Method, Step 4\n
    
    Computes the initial period predicition of each sample using the method described
    in Step 4 of the paper.
    
    #Arguments
        x: A 1-D numpy array containging the signal\n
        freq-range: A tuple containing the search range ex. (min_freq, max_freq)\n
        threshold: A floating point threshold value for picking the minimum of the
        cumulative mean normal difference equation.\n
        Fs: The sampling rate.
    
    #Returns
        taus: A 1-D numpy array containing the candidate period estimates for each sample.\n
        cum_diff_mat: A 2-D numpy array, see documentaion for `cumMeanNormDiffEq`.
    
    #Raises
        Not handed yet.
    """
    tau_min = int(Fs)//freq_range[1]
    tau_max = int(Fs)//freq_range[0]

    taus = zeros(x.size//tau_max, int16)
    tau_star = 0
    minimum = 1e9
    cum_diff_mat = cumMeanNormDiffEq(x, tau_max)
    for i in range(x.size//tau_max):
        cum_diff_eq = cum_diff_mat[i,:]
        for tau in range(tau_min, tau_max):
            if cum_diff_eq[tau] < threshold:
                taus[i] = tau
                break
            elif cum_diff_eq[tau] < minimum:
                tau_star = tau
                minimum = cum_diff_eq[tau]
        if taus[i] == 0:
            taus[i] = tau_star
    return taus, cum_diff_mat
    
def parabolicInterpolation(diff_mat, taus, freq_range, Fs):
    """
    Parabolic Interpolation, Step 4\n
    
    Applies parabolic interpolation onto the candidate period estimates using
    3 points corresponding to the estimate and it's adjacent values
    
    #Arguments
        diff_mat: A 2-D numpy array, see documentaion for `diffEquation`\n
        taus: A 1-D numpy array for the candidate estimates
        freq-range: A tuple containing the search range ex. (min_freq, max_freq)\n
        Fs: The sampling rate.
    
    #Returns
        local_min_abscissae: A 1-D numpy array containing the interpolated period estimates
        for each sample.
    
    #Raises
        Not handed yet.
    """
    abscissae = zeros((len(taus)-2, 3), float32)
    ordinates = zeros(abscissae.shape, float32)
    #if taus == []:
    for i, tau in enumerate(taus[1:-1]):
        ordinates[i-1] = diff_mat[i, tau-1:tau+2]
        abscissae[i-1] = asarray([tau-1, tau, tau+1], float32)
        
    period_min = int(Fs)//freq_range[1]
    period_max = int(Fs)//freq_range[0]
        
    coeffs = zeros((len(taus)-2, 3))
    local_min_abscissae = zeros(coeffs.shape[0])
    local_min_ordinates = zeros(coeffs.shape[0])
    for i in range(0, len(taus)-2):
        coeffs[i] = polyfit(abscissae[i,:], ordinates[i,:], 2)
        p = poly1d(coeffs[i]).deriv()
        if p.roots > period_min and p.roots < period_max:
            local_min_abscissae[i] = p.roots
        else:
            local_min_abscissae[i] = taus[i+1]
        local_min_ordinates[i] = p(local_min_abscissae[i])
        
    return local_min_abscissae

def pitchTrackingYIN(fname, freq_range = (40, 300), threshold = 0.1, timestep = 0.1, Fs = 48e3, target_Fs = 8e3, Fc = 1e3):
    """
    Putting it all together, this function is my implementation the the YIN pitch detection algorithm. /n/n
    
    
    
    Applies parabolic interpolation onto the candidate period estimates using
    3 points corresponding to the estimate and it's adjacent values
    
    #Arguments
        fname: The name of a WAV file to be analyzed. \n
        freq-range: An integer tuple containing the search range ex. (min_freq, max_freq)\n
        timestep: Tracking period in milliseconds. \n
        Fs: Sampling rate of the signal. \n
        target_Fs: Target sampling rate after downsampling. Use Fs for no downsampling. Note that the original sampling rate must
        be a multiple of the target rate. For example, this cannot downsample 44.1k to 16k.
        Fc: Cutoff frequency of the lowpass filter used in downsampling the signal. Must be less than target_Fs/2.
    
    #Returns
        f0: A 2-d numpy array containing the sample number and the ascociated frequency estimate.
    
    #Raises
        Not handed yet.
    """
    # Import the signal from a file
    signal = getData(fname)
    
    # Downsampling improves performance by reducing the total number of samples n
    # The downSample function is also applying a cutoff of 1k Hz
    step = int(Fs//target_Fs)
    if not step == 1:
        signal = downSample(signal, Fc, Fs, step)
        Fs = int(target_Fs)
    
    # This removes silence from the file, 
    signal = asarray(trim(signal))

    # The idea here is to keep only what we need to analyze.
    # If we are going to track frequency every 25ms, then we don't need the information in between
    # these points outside of the integration window.
    signal = signal.astype(float32)
    W = int(Fs)//freq_range[0]
    sampled_signal = zeros(((signal.size//int(Fs*timestep)),2*W+2), float32)
    for i in range((signal.size//int(Fs*timestep))):
        t = int(i*Fs*timestep)
        sampled_signal[i,:] = signal[t:int(t+2*W+2)]/max(signal[t:int(t+2*W+2)])
        
    sampled_signal = sampled_signal.flatten()
    signal = sampled_signal
    
    # absoluteThreshold finds the candidate period estimates
    periods_initial = absoluteThresold(signal, freq_range, threshold, Fs)[0]
    diff_mat = diffEquation(signal, W)
    
    # Apply parabolic interpolation for a better initial estimate
    periods_interp = parabolicInterpolation(diff_mat, periods_initial, freq_range, Fs)
    
    # Step 6 is still missing so I direcrtly used the interpolated estimates instead
    f0 = zeros((signal.size//W-2, 2), int16)
    for i in range(signal.size//W-2):
        f0[i,0] = i
        f0[i,1] = Fs//periods_interp[i]

    return f0
