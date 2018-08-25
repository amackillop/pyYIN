# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 11:44:54 2018

@author: Austin
"""
import wave
import numpy as np
from array import array
import scipy.signal as sig

def getdata(fname):
    """
    Get the audio information from a wav file.
    
    Arguments
    ---------
    fname : WAV file
        Name of the audio file to analyze.

    Returns
    -------
    data : ndarray
        A 1-D int16 numpy array containing the audio information.
    Fs : float
        A number indicating the sampling frequency of the signal.

    """
     #Extract Raw Audio from Wav File
    sound_file = wave.open(fname, 'r')
    Fs = sound_file.getframerate()
    data = sound_file.readframes(-1)
    data = wave.frombuffer(data, np.int16)
    sound_file.close()
    return data, Fs

def trim(signal, threshold=1000):
    """
    Trim the silence at the ends of the signal.
    
    Arguments
    ---------
    signal : ndarray
        A 1-D int16 numpy array containing the signal
    threshold : ndarray
        A volume threshold for silence detection

    Returns
    -------
    signal : ndarry
        A 1-D int16 numpy array containing the trimmed signal

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
    signal = np.asarray(signal, np.int16)
    
    return signal

def butter_lowpass_filter(signal, cutoff, Fs, order=6):
    """Implementing a buttersworth lowpass filter
    
    Arguments
    ---------
    signal : ndarray
        A 1-D numpy array containing the signal.
    cutoff : float
        Cutoff frequency for the lowpass filter.
    Fs : float
        Sampling rate of the signal.
    order : int
        Order of the lowpass filter.

    Returns
    -------
    ndarray:
        The original signal post filtering.
        
    """
    nyq = 0.5 * Fs
    normal_cutoff = cutoff / nyq
    b, a = sig.butter(order, normal_cutoff, btype='low', analog=False)
    signal = sig.lfilter(b, a, signal)
    return signal.astype(np.int16)

def downsample(signal, cutoff, Fs, order, step):
    """Downsample the signal by a specified integer amount.
    
    Note that the target dampling rate must be a factor of the original. 
    For example, you cannot use this to downsample 44.1k to 16k.
    
    Arguments
    ---------
    signal : ndarray
        A 1-D numpy array containing the signal
    cutoff : float
        Cutoff frequency for the lowpass filter 
    Fs : float
        Sampling rate of the original signal
    order : int
        The order of the buttersworth filter. Higher order produces a 
        sharper cutoff.
    step : int
        Integer specificying the decimation rate. a step of 2 means throwing away
        every other sample which halves the sampling rate.

    Returns
    -------
    signal : ndarray
        A 1-D numpy array containing the downsampled signal.\n

    """
    signal = butter_lowpass_filter(signal, cutoff, Fs, order)
    signal = signal[0::step]
    return signal