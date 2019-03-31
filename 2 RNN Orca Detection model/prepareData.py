import numpy as np
from pydub import AudioSegment
import random
import sys
import io
import os
import glob
import IPython
from td_utils import *

Ty=1375
Tx = 5511 
n_freq = 101
def get_random_time_segment(segment_ms):
    """
    Gets a random time segment of duration segment_ms in a 10,000 ms audio clip.
    
    Arguments:
    segment_ms -- the duration of the audio clip in ms ("ms" stands for "milliseconds")
    
    Returns:
    segment_time -- a tuple of (segment_start, segment_end) in ms
    """
    
    segment_start = np.random.randint(low=0, high=10000-segment_ms)   # Make sure segment doesn't run past the 10sec background 
    segment_end = segment_start + segment_ms - 1
    
    return (segment_start, segment_end)


def is_overlapping(segment_time, previous_segments):
    """
    Checks if the time of a segment overlaps with the times of existing segments.
    
    Arguments:
    segment_time -- a tuple of (segment_start, segment_end) for the new segment
    previous_segments -- a list of tuples of (segment_start, segment_end) for the existing segments
    
    Returns:
    True if the time segment overlaps with any of the existing segments, False otherwise
    """
    
    segment_start, segment_end = segment_time
    overlap = False
    for previous_start, previous_end in previous_segments:
        if previous_start<=segment_start<=previous_end or previous_start<=segment_end<=previous_end:
            overlap = True

    return overlap


def insert_audio_clip(background, audio_clip, previous_segments):
    """
    Insert a new audio segment over the background noise at a random time step, ensuring that the 
    audio segment does not overlap with existing segments.
    
    Arguments:
    background -- a 10 second background audio recording.  
    audio_clip -- the audio clip to be inserted/overlaid. 
    previous_segments -- times where audio segments have already been placed
    
    Returns:
    new_background -- the updated background audio
    """
    
    # Get the duration of the audio clip in ms
    segment_ms = len(audio_clip)
    
    # Step 1: Use one of the helper functions to pick a random time segment onto which to insert 
    segment_time = get_random_time_segment(segment_ms)
    
    # Step 2: Check if the new segment_time overlaps with one of the previous_segments. If so, keep 
    while is_overlapping(segment_time, previous_segments) == True:
        segment_time = get_random_time_segment(segment_ms)

    # Step 3: Add the new segment_time to the list of previous_segments (≈ 1 line)
    previous_segments.append(segment_time)
    
    # Step 4: Superpose audio segment and background
    new_background = background.overlay(audio_clip, position = segment_time[0])
    
    return new_background, segment_time


def insert_ones(y, segment_end_ms):
    """
    Update the label vector y. The labels of the 50 output steps strictly after the end of the segment 
    should be set to 1. By strictly we mean that the label of segment_end_y should be 0 while, the
    50 followinf labels should be ones.
    
    
    Arguments:
    y -- numpy array of shape (1, Ty), the labels of the training example
    segment_end_ms -- the end time of the segment in ms
    
    Returns:
    y -- updated labels
    """
    
    # duration of the background (in terms of spectrogram time-steps)
    segment_end_y = int(segment_end_ms * Ty / 10000.0)
    
    # Add 1 to the correct index in the background label (y)
    for i in range(segment_end_y+1, segment_end_y+51):
        if i < Ty:
            y[0, i] = 1
    
    return y

# GRADED FUNCTION: create_training_example

def create_training_example(background, activates, negatives):
    """
    Creates a training example with a given background, activates, and negatives.
    
    Arguments:
    background -- a 10 second background audio recording
    activates -- a list of audio segments of the orca calls
    negatives -- a list of audio segments of random words that are not orca calls
    
    Returns:
    x -- the spectrogram of the training example
    y -- the label at each time step of the spectrogram
    """
    
    # Set the random seed    
    # Make background quieter
    background = background - 20

    # Step 1: Initialize y (label vector) of zeros (≈ 1 line)
    y = np.zeros((1,Ty))

    # Step 2: Initialize segment times as empty list (≈ 1 line)
    previous_segments = []
    
    number_of_activates = np.random.randint(0, 5)
    random_indices = np.random.randint(len(activates), size=number_of_activates)
    random_activates = [activates[i] for i in random_indices]
    

    for random_activate in random_activates:
        # Insert the audio clip on the background
        background, segment_time = insert_audio_clip(background, random_activate, previous_segments)
        # Retrieve segment_start and segment_end from segment_time
        segment_start, segment_end = segment_time[0], segment_time[1]
        # Insert labels in "y"
        y = insert_ones(y, segment_end)

    # Select 0-2 random negatives audio recordings from the entire list of "negatives" recordings
    number_of_negatives = np.random.randint(0, 3)
    random_indices = np.random.randint(len(negatives), size=number_of_negatives)
    random_negatives = [negatives[i] for i in random_indices]

    for random_negative in random_negatives:
        # Insert the audio clip on the background 
        background, _ = insert_audio_clip(background, random_negative, previous_segments)
    
    # Standardize the volume of the audio clip 
    background = match_target_amplitude(background, -20.0)
    
    return y, background


"""
Here:
	activates - list of orca calls
	negatives - list of non-orca calls
	backgrounds - list of backgrounds
"""
activates, negatives, backgrounds = load_raw_audio()

if not os.path.exists("./generatedAUdioData/train/"):
    os.makedirs("./generatedAUdioData/train/")
# Create training samples
train_samples = 60
X_data = np.zeros((train_samples, 5511, 101))
Y_data = np.zeros((train_samples, 1375, 1))
for i in range(train_samples):
    bgNum = np.random.randint(0, len(backgrounds))
    y, background = create_training_example(backgrounds[bgNum], activates, negatives)
    fileName = str(i)+"_"+str(bgNum)+"_train" + ".wav"
    file_handle = background.export("./generatedAUdioData/train/"+fileName, format="wav")
    print("File (" + fileName + ") was saved in your directory.")
    x = graph_spectrogram("./generatedAUdioData/train/"+fileName)
    X_data[i] = x.T
    Y_data[i] = y.T
    

if not os.path.exists("./generatedAUdioData/dev/"):
    os.makedirs("./generatedAUdioData/dev/")
# Create dev samples
dev_samples = 20
X_dev= np.zeros((dev_samples, 5511, 101))
Y_dev = np.zeros((dev_samples, 1375, 1))
for i in range(dev_samples):
    bgNum = np.random.randint(0, len(backgrounds))
    y, background = create_training_example(backgrounds[bgNum], activates, negatives)
    fileName = str(i)+"_"+str(bgNum)+"_dev" + ".wav"
    file_handle = background.export("./generatedAUdioData/dev/"+fileName, format="wav")
    print("File (" + fileName + ") was saved in your directory.")
    x = graph_spectrogram("./generatedAUdioData/dev/"+fileName)
    X_dev[i] = x.T
    Y_dev[i] = y.T

if not os.path.exists("./generatedAUdioData/test/"):
    os.makedirs("./generatedAUdioData/test/")
# Create test samples
test_samples = 20
X_test= np.zeros((test_samples, 5511, 101))
Y_test = np.zeros((test_samples, 1375, 1))
for i in range(test_samples):
    bgNum = np.random.randint(0, len(backgrounds))
    y, background = create_training_example(backgrounds[bgNum], activates, negatives)
    fileName = str(i)+"_"+str(bgNum)+"_test" + ".wav"
    file_handle = background.export("./generatedAUdioData/test/"+fileName, format="wav")
    print("File (" + fileName + ") was saved in your directory.")
    x = graph_spectrogram("./generatedAUdioData/test/"+fileName)
    X_test[i] = x.T
    Y_test[i] = y.T

    
if not os.path.exists("./data/XY_train"):
    os.makedirs("./data/XY_train/")
    os.makedirs("./data/XY_dev/")
    os.makedirs("./data/XY_test/")
    
# Save the data in numpy format
np.save("./data/XY_train/X.npy", X_data)
np.save("./data/XY_train/Y.npy", Y_data)
np.save("./data/XY_dev/X_dev.npy", X_dev)
np.save("./data/XY_dev/Y_dev.npy", Y_dev)
np.save("./data/XY_test/X_test.npy", X_test)
np.save("./data/XY_test/Y_test.npy", Y_test)