import os
import numpy as np
import scipy
import librosa
import scipy.fftpack as fft
from scipy.signal import get_window
import matplotlib.pyplot as plt

# Plotting
def plot(features, ncc: int,feat: str):
    if feat!='mfcc' and feat!='lpcc':
        raise ValueError('Feature must be mfcc or lpcc')
    cep_coeff, cep_coeff_weighted, cep_coeff_delta_1, cep_coeff_delta_2 = features
    plt.figure(figsize=(15,15))
    plt.subplot(4, 1, 1)
    plt.imshow(cep_coeff, aspect='auto', origin='lower')
    plt.title(f'{feat.upper()}s (n{feat}={ncc})')
    plt.grid(True)
    plt.subplot(4, 1, 2)
    plt.imshow(cep_coeff_weighted, aspect='auto', origin='lower')
    plt.title(f'{feat.upper()}s (Weighted, n{feat}={ncc})')
    plt.grid(True)
    plt.subplot(4, 1, 3)
    plt.imshow(cep_coeff_delta_1, aspect='auto', origin='lower')
    plt.title(f'{feat.upper()}s (Delta 1, n{feat}={ncc})')
    plt.grid(True)
    plt.subplot(4, 1, 4)
    plt.imshow(cep_coeff_delta_2, aspect='auto', origin='lower')
    plt.title(f'{feat.upper()}s (Delta 2, n{feat}={ncc})')
    plt.grid(True)
    return

# Add noise
def add_noise(signal, snr_db):
    signal_power = np.mean(signal**2)
    # convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    # generate white Gaussian noise
    noise = np.sqrt(noise_power) * np.random.normal(size=signal.shape)
    noisy_signal = signal + noise
    return noisy_signal

# MFCCs
def preemphasis(signal: np.ndarray, alpha=0.95) -> np.ndarray:
    signal = signal/np.max(signal)
    kernel = np.array([1, -alpha])
    ret = np.convolve(signal, kernel, mode='full')
    return ret

def frame_audio(audio, FFT_size=1024, hop_size=10, sample_rate=44100):
    # hop_size in ms
    
    audio = np.pad(audio, int(FFT_size / 2), mode='reflect')
    # convert hop_size to samples
    hop_size = np.round(sample_rate * hop_size / 1000).astype(int)
    frame_num = int((len(audio) - FFT_size) / hop_size) + 1
    frames = np.zeros((frame_num,FFT_size))
    
    for n in range(frame_num):
        frames[n] = audio[n*hop_size:n*hop_size+FFT_size]
    
    return frames

def window_frames(frames, FFT_size=1024):
    window = get_window("hann", FFT_size, fftbins=True)
    return frames*window

def fft_frames(frames, FFT_size=1024):
    audio_fft = np.empty((frames.shape[0], int(1 + FFT_size // 2)), dtype=np.complex64)
    for i in range(frames.shape[0]):
        audio_fft[i,:] = fft.fft(frames[i],axis=0)[:audio_fft.shape[1]]
    audio_power = np.square(np.abs(audio_fft))
    return audio_power

# Mel scale filter bank
def freq_to_mel(freq):
    return 2595.0 * np.log10(1.0 + freq / 700.0)

def mel_to_freq(mels):
    return 700.0 * (10.0**(mels / 2595.0) - 1.0)

def get_filter_points(fmin, fmax, mel_filter_num, FFT_size=1024, sample_rate=44100):
    fmin_mel = freq_to_mel(fmin)
    fmax_mel = freq_to_mel(fmax)
    
    print("MEL min: {0}".format(fmin_mel))
    print("MEL max: {0}".format(fmax_mel))
    
    mels = np.linspace(fmin_mel, fmax_mel, num=mel_filter_num+2)
    freqs = mel_to_freq(mels)
    
    return np.floor((FFT_size + 1) / sample_rate * freqs).astype(int), freqs

def get_filters(filter_points, FFT_size=1024):
    filters = np.zeros((len(filter_points)-2,int(FFT_size/2+1)))
    
    for n in range(len(filter_points)-2):
        filters[n, filter_points[n] : filter_points[n + 1]] = np.linspace(0, 1, filter_points[n + 1] - filter_points[n])
        filters[n, filter_points[n + 1] : filter_points[n + 2]] = np.linspace(1, 0, filter_points[n + 2] - filter_points[n + 1])
    
    return filters

def dct(dct_filter_num, filter_len):
    basis = np.empty((dct_filter_num,filter_len))
    basis[0, :] = 1.0 / np.sqrt(filter_len)
    
    samples = np.arange(1, 2 * filter_len, 2) * np.pi / (2.0 * filter_len)

    for i in range(1, dct_filter_num):
        basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_len)
        
    return basis

def parameter_weighting(cep_coeffs, Q=None):
    if not Q:
        Q = len(cep_coeffs)-1
    if len(cep_coeffs)>Q+1:
        raise ValueError("Choose a larger Q")
    bandpass_lifter = np.sin(np.arange(len(cep_coeffs))*np.pi/Q)*Q/2+1
    return bandpass_lifter[:, np.newaxis]*cep_coeffs

def mfcc(y: np.ndarray, sr: int, n_mfcc=13, mel_filter_num=40):
    preemphasized = preemphasis(signal=y)
    frames = frame_audio(preemphasized)
    windowed = window_frames(frames)
    audio_power = fft_frames(windowed)
    freq_min = 0
    freq_high = sr / 2
    # mel_filter_num = 40
    filter_points, mel_freqs = get_filter_points(freq_min, freq_high, mel_filter_num, sample_rate=sr)
    filters = get_filters(filter_points)
    enorm = 2.0 / (mel_freqs[2:mel_filter_num+2] - mel_freqs[:mel_filter_num])
    filters *= enorm[:, np.newaxis]
    audio_filtered = filters@audio_power.T
    audio_log = 10*np.log10(audio_filtered+1e-9)
    dct_filters = dct(n_mfcc, mel_filter_num)
    mfcc = dct_filters @ audio_log
    mfcc_weighted = parameter_weighting(mfcc)
    print(mfcc_weighted.shape)
    delta_1 = librosa.feature.delta(mfcc_weighted, order=1, axis=1)
    delta_2 = librosa.feature.delta(mfcc_weighted, order=2, axis=1)
    return mfcc, mfcc_weighted, delta_1, delta_2

# LPCCs
def lpcc(y, n_lpcc=13, order=12):
    lpc, error = compute_lpc(y, order)
    lpcc = compute_lpcc(lpc, error, n_lpcc)
    lpcc_weighted = []
    for i in range(lpcc.shape[1]):
        lpcc_weighted.append(parameter_weighting(lpcc[:,i]))
    lpcc_weighted = np.vstack(lpcc_weighted).T
    delta_1 = librosa.feature.delta(lpcc_weighted, order=1, axis=1)
    delta_2 = librosa.feature.delta(lpcc_weighted, order=2, axis=1)
    return lpcc, lpcc_weighted, delta_1, delta_2

def compute_lpc(audio, order=12):
    preemphasized = preemphasis(audio)
    frames = frame_audio(preemphasized)
    windowed = window_frames(frames)
    lpc, error = [], []
    for i in range(len(windowed)):
        coeffs = librosa.lpc(windowed[i], order=order)
        lpc.append(coeffs)
        error.append(compute_prediction_error_power(windowed[i], coeffs))
    lpc = np.vstack(lpc)
    error = np.array(error)
    return lpc.T, error

def compute_lpcc(lpc, error, num_cepstral=13):
    lpcc = []
    for i in range(lpc.shape[1]):
        lpcc.append(lpc_to_cc(lpc[:,i], num_cepstral, error[i]))
    lpcc = np.vstack(lpcc).T
    return lpcc

def compute_prediction_error_power(signal, lpc_coeffs):
    prediction = np.zeros(len(signal)+len(lpc_coeffs)-1)
    signal = np.concatenate([signal, np.zeros(len(lpc_coeffs)-1)], axis=0)
    shift = signal.copy()
    for i in range(len(lpc_coeffs)):
        if i>0:
            shift[1:] = shift[:-1]
            shift[0] = 0
        prediction = prediction + lpc_coeffs[i]*shift
    error = signal - prediction
    power = np.sum(np.abs(error)**2)
    return power

def lpc_to_cc(lpc_coeffs, num_cepstral, pred_error_power):
    cep_coeffs = np.zeros(num_cepstral)
    cep_coeffs[0] = np.log(pred_error_power)

    for m in range(1, num_cepstral):
        cep_coeffs[m] = -lpc_coeffs[m] if m < len(lpc_coeffs) else 0
        summation = 0
        for k in range(1, m):
            summation -= k * cep_coeffs[k] * lpc_coeffs[m - k]
        cep_coeffs[m] += summation/m

    return cep_coeffs