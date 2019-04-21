from pydub import AudioSegment
import pydub
import os

def read_file(filename):
    if filename.endswith("wav"):
        return AudioSegment.from_wav(filename)
    elif filename.endswith("flv"):
        return AudioSegment.from_wav(filename)
    elif filename.endswith("ogg"):
        return AudioSegment.from_ogg(filename)
    elif filename.endswith("mp3"):
        return AudioSegment.from_mp3(filename)
    else:
        raise Exception("File not supported. Check if file extension in included in filename")
    return None

def set_rate(audio, rate):
    return audio.set_frame_rate(rate)

def set_audio_channels(audio, nChannel):
    return audio.set_channels(1)

def normalize_audio(audio):
    return pydub.effects.normalize(audio)

def standardize_data(filename, rate=44100, nchannels=1, chunk_size=10, target_location="./standardized/"):
    """
        arguments: 
            filename- absolute path of input audio.
            rate- rate for chunks. default 44100
            nchannels- channels for chunks. default 1
            chunk_size- size of chunks in seconds. default 10 sec.
            target_location- location where chunks will be saved.

        returns:
            filename- original file name.
            fileLength- Length of original input file in seconds.
            fileChannels- Channels of input audio
            totalChunks- Toral number of chunks generated
            chunk_size- size of each chunk in seconds.
            path_list- list containing path of each chunk.

    """
    f = read_file(filename)
    fileLength = f.duration_seconds
    fileChannels = f.channels
    
    if f.frame_rate != rate:
        f = set_rate(f, rate)
    if f.channels != nchannels:
        f = set_audio_channels(f, nchannels)
    f = normalize_audio(f)
    j = 0
    if not os.path.exists(target_location):
        os.makedirs(target_location)
    
    path_list = []
    totalChunks = 0
    while len(f[:]) >= chunk_size*1000:
        chunk = f[:chunk_size*1000]
        p = target_location+filename.strip().split('.')[0]+"_" + str(j)+ ".wav"
        chunk.export(p, format ="wav")
        path_list.append(p)
        print("File stored at "+ p)
        f = f[chunk_size*1000:]
        j += 1
        totalChunks += 1
        
    if 0 < len(f[:]) and len(f[:]) < chunk_size*1000:
        silent = AudioSegment.silent(duration=chunk_size*1000)
        p = target_location+filename.strip().split('.')[0]+"_" + str(j)+ ".wav"
        paddedData = silent.overlay(f, position=0, times=1)
        paddedData.export(p, format ="wav")
        path_list.append(p)
        print("File stored at "+ p)
        totalChunks += 1
        
    return filename, fileLength, fileChannels, totalChunks, chunk_size, path_list