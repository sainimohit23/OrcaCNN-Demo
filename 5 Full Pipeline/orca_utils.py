import matplotlib.pyplot as plt
import numpy as np 
import math
from scipy.io import wavfile
import os

def graph_spectrogram(data):
    nfft = 200 # Length of each window segment
    fs = 8000 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx

def get_padded_data(data, stride):
    if data.shape[0]>=441000:
        slides_before_padding = math.ceil(((data.shape[0]-441000)/stride))
        padding = (slides_before_padding*stride) - data.shape[0] + 441000
        padded_data = np.zeros((padding + data.shape[0]))
        padded_data[0:len(data[:])] = data[:]
    else:
        padding = 441000 - data.shape[0]
        padded_data = np.zeros((padding + data.shape[0]))
        padded_data[0:len(data[:])] = data[:]
    return padded_data


def detect_calls(filename, stride_sec, model):
    """
    	This function uses sliding windows mechanism to detect orca call time.
		args: 
			filename- path of file.
			stride_sec- stride in seconds for sliding windows.
			model- orca detection model.

		returns:
			preds- array of predictions.
    """
    rate, data = wavfile.read(filename)
    stride = int(44100*stride_sec)
    pred_stride = 70   
    data = get_padded_data(data, stride)
    total_slides = int(((data.shape[0]-441000)/stride)) + 1
    
    total_preds_len = ((total_slides-1)*pred_stride) + 1375
    preds = np.zeros((1, total_preds_len, 1), dtype=np.float32)
    length = data.shape[0]
    start = 0
    pred_start = 0
    i = 0
    j=1
    while start+441000 <= data.shape[0]:
        j+=1
        segment = data[start:start+441000]
        x = graph_spectrogram(segment)
        x  = x.swapaxes(0,1)
        x = np.expand_dims(x, axis=0)
        model_preds = model.predict(x)
        for i in range(len(model_preds[0,:,0])):
            if  preds[0,pred_start+i, 0] < model_preds[0, i, 0]:
                preds[0,pred_start+i, 0] = model_preds[0, i, 0]
        
        start = start+stride
        pred_start = pred_start + pred_stride
                    
    return preds

def predict_call_time(predictions, threshold, duration):
    """
    		args:
			predictions- array of predictions.
			threshold- threshold for orca call.
			duration- duration of audio in seconds.

		returns:
			call_timestamps- list of time of orca calls in seconds.
    """
    Ty = predictions.shape[1]
    consecutive_timesteps = 0
    start_of_orca = 0
    
    call_timestamps = []
    for i in range(Ty):
        consecutive_timesteps += 1        
        if predictions[0,i,0] > threshold and consecutive_timesteps > 75:
            call_timestamps.append(((i-75) / Ty) * duration)
            consecutive_timesteps = 0
    
    return call_timestamps
