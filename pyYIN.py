# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:00:39 2018

@author: Austin Mackillop

YIN Algorithm
"""
import numpy as np
import _tools

def auto_correlate(x, tau, W):
    """
    Autocorrelation, Step 1, Eq. (1)
    
    Computes the auto correlation of the signal shifted by tau.
    
    Arguments
    ---------
    x : ndarray
        A 1-D numpy array containing the signal
    tau : int
        Sample shift
    W : int
        Integration window size.
        
    Returns
    -------
    autocorr_mat : ndarray
        A 2-D numpy array of the correlation function for each sample. 
        Each row corresponds to a sample.
    """
    
    # Initialize cross correlation matrix
    autocorr_mat = np.zeros((x.size//(2*W), W), np.float32)
    # Store original signal
    x_orig = list(x)
    
    for i in range(autocorr_mat.shape[0]):
        t = 2*i*W
        # Unbias the signals
        x = x_orig[t:W+t] - np.mean(x_orig[t:W+t])
        x_tau = x_orig[t+tau:t+tau+W] - np.mean(x_orig[t+tau:t+tau+W])
        if len(x_tau) == 0:
            # We reached the end
            break
        
        # Auto correlation
        autocorr = np.correlate(x_tau, x_tau, 'full')
        try:
            autocorr_mat[i, :] = autocorr[autocorr.shape[0]//2:]
        except:
            padding = np.zeros((1, W-autocorr[autocorr.shape[0]//2:].size), np.float32)
            autocorr_mat[i, :] = np.append(autocorr[autocorr.shape[0]//2:], padding)
                
    return autocorr_mat


def diff_equation(x, W):
    """
    Difference Equation, Step 2, Eq. (7)\n
    
    Computes the difference equation for each sample
    
    # Arguments
        x: A 1-D numpy array containing the signal\n
        W: Integration window size\n
        
    # Returns
        diff_eq_mat: A 2-D numpy array of the computed difference equations for each sample.
        Each row corresponds to a sample.\n
    """
    
    # Get the autocorrelation matrix
    auto_corr_mat = auto_correlate(x, 0, W)
    # Initialize difference equation matrix
    diff_eq_mat = np.zeros(auto_corr_mat.shape, np.float32)
    for i in range(0, diff_eq_mat.shape[0]):
        # Implement Equation (7) from the paper
        diff_eq_mat[i,:] = (auto_corr_mat[i, 0]
                            + auto_correlate(x, i, W)[i, 0]
                            - 2 * auto_corr_mat[i,:])
    return diff_eq_mat

def cum_mean_norm_diff_eq(x, W):
    """
    Cumulative Mean Normal Difference Equation, Step 3, Eq. (8)\n
    
    Computes the cumulative mean normal difference equation for each sample
    
    # Arguments
        x: A 1-D numpy array containing the signal\n
        W: Integration window size\n
        
    # Returns
        cum_diff_mat: A 2-D numpy array of the computed difference equations
        for each sample. Each row corresponds to a sample.\n
    """
    
    # Get the difference equation matrix
    diff_eq_mat = diff_equation(x, W)
    # Initialize cumulative mean normal difference matrix
    cum_diff_mat = np.zeros(diff_eq_mat.shape, np.float32)
    cum_diff_mat[:, 0] = 1
    # Implement Eq. (8) from the paper
    for t in range(diff_eq_mat.shape[0]):
        for j in range(1, diff_eq_mat.shape[1]):
            cum_diff_mat[t, j] = diff_eq_mat[t, j] / ((1/j) * sum(diff_eq_mat[t, 1:j+1]))
    return cum_diff_mat
        
def absoluteThresold(x, freq_range = (40, 300), threshold = 0.1, Fs = 16e3, W = 400):
    """
    Absolute Threshold Method, Step 4\n
    
    Computes the initial period predicition of each sample using the method described
    in Step 4 of the paper.
    
    #Arguments
        x: A 1-D numpy array containging the signal\n
        freq-range: A tuple containing the search range ex. (min_freq, max_freq)\n
        threshold: A floating point threshold value for picking the minimum of the
        cumulative mean normal difference equation.\n
        Fs: The sampling rate.\n
    
    #Returns
        taus: A 1-D numpy array containing the candidate period estimates for each sample.\n
        cum_diff_mat: A 2-D numpy array, see documentaion for `cum_mean_norm_diff_eq`.\n
    """
    
    # Set search range
    tau_min = int(Fs)//freq_range[1]
    tau_max = int(Fs)//freq_range[0] - 1
    tau_star = 0
    minimum = 1e9
    
    # Get the cumulative mean norm difference (CMND) matrix and initialize list of taus
    cum_diff_mat = cum_mean_norm_diff_eq(x, W)
    taus = np.zeros(cum_diff_mat.shape[0], np.int16)
    
    # Implement the search as specified in step 4 of the paper
    for i in range(taus.size):
        cum_diff_eq = cum_diff_mat[i,:]
        for tau in range(tau_min, tau_max):
            # Check if the value of the CMND equation is les than the specified threshold
            if cum_diff_eq[tau] < threshold:
                taus[i] = tau
                break
            # Keep track of the global minimum if no value is below the threshold
            elif cum_diff_eq[tau] < minimum:
                tau_star = tau
                minimum = cum_diff_eq[tau]
                
        # No value found below threshold, select the global minimum        
        if taus[i] == 0:
            taus[i] = tau_star
    return taus, cum_diff_mat
    
def parabolicInterpolation(diff_matrix, taus, freq_range, Fs):
    """
    Parabolic Interpolation, Step 4
    
    Applies parabolic interpolation onto the candidate period estimates using
    3 points corresponding to the estimate and it's adjacent values
    
    #Arguments
        diff_matrix: A 2-D numpy array, see documentaion for `diffEquation`\n
        taus: A 1-D numpy array for the candidate estimates\n
        freq_range: A tuple containing the search range ex. (min_freq, max_freq)\n
        Fs: The sampling rate.\n
    
    #Returns
        local_min_abscissae: A 1-D numpy array containing the interpolated period estimates
        for each sample.\n
    
    #Raises
        Not handed yet.
    """
    
    # Initialize coordinates for interpolation
    abscissae = np.zeros((len(taus)-2, 3), np.float32)
    ordinates = np.zeros(abscissae.shape, np.float32)
    
    # Interpolation uses the minimum and its two adjacent points
    for i, tau in enumerate(taus[1:-1]):
        ordinates[i-1] = diff_matrix[i, tau-1:tau+2]
        abscissae[i-1] = np.asarray([tau-1, tau, tau+1], np.float32)
    
    # Initalize period estimates    
    period_min = int(Fs)//freq_range[1]
    period_max = int(Fs)//freq_range[0]
    
    # Initialize parabolic equation coefficients and local min abscissae
    coeffs = np.zeros((len(taus)-2, 3))
    local_min_abscissae = np.zeros(coeffs.shape[0])
    
    # Implement the parabolic interpolation
    for i in range(0, len(taus)-2):
        # Fit the parabola to each set of three points and take the derivative
        coeffs[i] = np.polyfit(abscissae[i,:], ordinates[i,:], 2)
        p = np.poly1d(coeffs[i]).deriv()
        
        # If the root of the derivative is within our range, select it
        if p.roots > period_min and p.roots < period_max:
            local_min_abscissae[i] = p.roots
            
        # Else simply take the center point pre interpolation
        else:
            local_min_abscissae[i] = taus[i+1]
        
    return local_min_abscissae

def preprocess(signal, Fs, window, target_Fs=16e3, timestep=0.1):
    # Downsample signal if desired to improve speed
    step = int(Fs//target_Fs)
    if not step == 1:
        signal = _tools.downsample(signal, 1e3, Fs, step)
        Fs = int(target_Fs)

    # Remove silence from beginning and end of the file to improve speed
    signal = np.asarray(_tools.trim(signal)).astype(np.float32)
    
    # The idea here is to keep only what we need to analyze.
    # If we are going to track frequency every 25ms, then we don't need the 
    # information in between these points outside of the integration window.
    W = window
    sampled_signal = np.zeros(((signal.size//int(Fs*timestep)),2*W), np.float32)
    if sampled_signal.shape[0] < 5:
        raise RuntimeError("Signal is too short or possibly too quiet.")
        
    for i in range((signal.size//int(Fs*timestep))):
        t = int(i*Fs*timestep)
        sampled_signal[i,:] = signal[t:int(t+2*W)]/max(signal[t:int(t+2*W)])
        
    sampled_signal = sampled_signal.flatten()

    # This new signal has been minimized in length and normalized, preprocessing is finished
    signal = sampled_signal
    if len(signal) > 0:
        pass
    else:
        raise ValueError("Detected a silent file")
        
    return signal, Fs

def get_estimates(signal, Fs, periods, window):
    W = window
    # Minues 2 because the interpolation throws away two estimates
    estimates = np.zeros((int(signal.size/(2*W)-2), 1), np.int16)
    for i in range(int(signal.size/(2*W)-2)):
        estimates[i] = Fs//periods[i]
    return estimates
        
def yin_algorithm(fname, threshold=0.1, target_Fs=16e3, freq_range=(40, 300), timestep=0.1, Fc=1e3):
    """
    Putting it all together, this function is my implementation the the YIN pitch 
    detection algorithm. 
    
    #Arguments
        fname: The name of a WAV file to be analyzed.\n
        freq-range: An integer tuple containing the search range ex. (min_freq, max_freq)\n
        threshold: A float specifying the threshold for step 4\n
        timestep: Tracking period in seconds.\n
        target_Fs: Target sampling rate after downsampling. Use Fs for no downsampling. Note that the original sampling rate must\n
        be a multiple of the target rate. For example, this cannot downsample 44.1k to 16k.\n
        Fc: Cutoff frequency of the lowpass filter used in downsampling the signal. Must be less than target_Fs/2.\n
    
    #Returns
        f: A 1-D numpy array containing the frequency esimates\n
        
    #Raises
        ValueError: If a silent audio file is used.\n
        RuntimeError: If there are not enough points for parabolic interpolation
    """
    # Extract signal from audio file
    signal, Fs = _tools.getdata(fname)
    # Integration window is dependent on the lowest frequency to be detected
    window = int(np.ceil(Fs/freq_range[0]))
    signal, Fs = preprocess(signal, Fs, target_Fs, window, threshold, freq_range, timestep)


    # Now apply the algorithm, note that step 6 is still missing
    taus = absoluteThresold(signal, freq_range, threshold, Fs, window = window)[0]
    diff_mat = diff_equation(signal, window)
    periods = parabolicInterpolation(diff_mat, taus, freq_range, Fs)
    estimates = get_estimates(signal, Fs, periods, window)

    return estimates
