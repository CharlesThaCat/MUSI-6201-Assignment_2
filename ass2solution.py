import numpy as np
from scipy.signal import medfilt, find_peaks
from matplotlib import pyplot as plt
from scipy.io.wavfile import read
import glob
import os
import math

########################### A. Feature Extraction ###########################
# 1
def extract_spectral_centroid(xb, fs):
    vsc_matrix = []
    for b in xb:
        length = len(b)
        b = np.multiply(b, compute_hann(length))
        magnitudes = np.abs(np.fft.rfft(b)) # magnitude spectrum for positive frequencies
        freqs = np.abs(np.fft.rfftfreq(length, 1.0/fs)) # positive frequencies
        vsc = np.sum(magnitudes*freqs) / np.sum(magnitudes)
        vsc_matrix.append(vsc)
    return np.array(vsc_matrix)

def extract_rms(xb):
    rms_matrix = []
    for b in xb:
        rms = np.sqrt(np.mean(np.square(b)))
        if rms < 1e-5:
            rms= 1e-5
        rms = 20*np.log10(rms)
        rms_matrix.append(rms)
    # print(rms_matrix)
    return np.array(rms_matrix)

def extract_zerocrossingrate(xb):
    vzc_matrix = []
    for b in xb:
        temp = np.mean(np.abs(np.diff(np.sign(b))))
        vzc = temp * 0.5
        vzc_matrix.append(vzc)
    return np.array(vzc_matrix)

def extract_spectral_crest(xb):
    vtsc_matrix = []
    for b in xb:
        length = len(b)
        b = np.multiply(b, compute_hann(length))
        magnitudes = np.abs(np.fft.rfft(b)) # magnitude spectrum for positive frequencies
        vtsc = np.divide(np.amax(magnitudes),np.sum(magnitudes))
        vtsc_matrix.append(vtsc)
    return np.array(vtsc_matrix)

def extract_spectral_flux(xb):
    vsf_matrix = []
    [NumOfBlocks,length] = xb.shape
    for i in range(0,NumOfBlocks):
        b_n = xb[i,:]
        b_n = np.multiply(b_n, compute_hann(length))
        if i == 0:
            b_n_1 = np.zeros_like(b_n)
        else:
            b_n_1 = xb[i-1,:]
        b_n_1 = np.multiply(b_n_1, compute_hann(length))
        magnitudes_n = np.abs(np.fft.rfft(b_n))
        magnitudes_n_1 = np.abs(np.fft.rfft(b_n_1))
        temp = np.sqrt(np.sum(np.square(magnitudes_n - magnitudes_n_1)))
        vsf = np.divide(temp, (length/2))
        vsf_matrix.append(vsf)
    return np.array(vsf_matrix)

def compute_hann(iWindowLength):
    return 0.5 - (0.5 * np.cos(2 * np.pi / iWindowLength * np.arange(iWindowLength)))


# 2
def extract_features(x, blockSize, hopSize, fs):
    blocked_x, timeInSec = block_audio(x, blockSize, hopSize, fs)
    xb = blocked_x
    NumOfBlocks = xb.shape[0]
    features = np.zeros((5,NumOfBlocks))
    features[0,:] = extract_spectral_centroid(xb, fs)
    features[1,:] = extract_rms(xb)
    features[2,:] = extract_zerocrossingrate(xb)
    features[3,:] = extract_spectral_crest(xb)
    features[4,:] = extract_spectral_flux(xb)
    return features

def block_audio(x,blockSize,hopSize,fs):
    # allocate memory
    numBlocks = math.ceil(x.size / hopSize)
    xb = np.zeros([numBlocks, blockSize])
    # compute time stamps
    t = (np.arange(0, numBlocks) * hopSize) / fs

    x = np.concatenate((x, np.zeros(blockSize)),axis=0)
    for n in range(0, numBlocks):
        i_start = n * hopSize
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])

        xb[n][np.arange(0,blockSize)] = x[np.arange(i_start, i_stop + 1)]

    return (xb,t)


# 3    
def aggregate_feature_per_file(features):
    aggFeatures = np.zeros((10,1))
    for i in range(0,5):
        aggFeatures[2*i,:] = np.mean(features[i,:])
        aggFeatures[2*i+1,:] = np.std(features[i,:])
    # print(aggFeatures)
    return aggFeatures


# 4
def get_feature_data(path, blockSize, hopSize):
    file_path = os.path.join(path, '*.wav')
    wav_files = [f for f in glob.glob(file_path)]
    print(len(wav_files))
    num_of_files = np.array(wav_files).shape[0]
    featureData = np.zeros((10,num_of_files))

    k = 0
    for wav_file in wav_files:
        fs, audio = read(wav_file)
        features = extract_features(audio, blockSize, hopSize, fs)
        aggFeatures = aggregate_feature_per_file(features)
        # featureData.append(aggFeatures)
        # print(featureData[:,0:1])
        featureData[:,k:k+1] = aggFeatures
        k = k+1

    return featureData

########################### B. Feature Normalization ###########################

def normalize_zscore(FeatureData):
    mean1= np.mean(FeatureData,axis=1)
    std1= np.std(FeatureData,axis=1)
    num_feature,num_sample=np.shape(FeatureData)
    normalized_feature=np.zeros((num_feature,num_sample))
    for i in range(num_feature):
        normalized_feature[i,:]=(FeatureData[i,:]-mean1[i])/std1[i]
    return normalized_feature

########################### C. Feature Visualization ###########################

def visualize_features(path_to_musicspeech):
    blockSize = 1024
    hopSize = 256
    path1 = os.path.join(path_to_musicspeech,'music_wav')
    path2 = os.path.join(path_to_musicspeech,'speech_wav')
    featureData_music = get_feature_data(path1, blockSize, hopSize)
    featureData_speech = get_feature_data(path2, blockSize, hopSize)
    featureData=np.concatenate((featureData_music,featureData_speech),axis=1)
    normalized_feature=normalize_zscore(featureData) 
    fig=plt.figure(figsize=(12, 8))
    plt.subplot(2,3,1)
    plt.scatter(normalized_feature[6,0:64],normalized_feature[0,0:64] , color='r')
    plt.scatter(normalized_feature[6,64:128],normalized_feature[0,64:128] , color='b')
    plt.xlabel('SCR-mean')
    plt.ylabel('SC-mean')
    plt.subplot(2,3,2)
    plt.scatter(normalized_feature[8,0:64],normalized_feature[4,0:64] , color='r')
    plt.scatter(normalized_feature[8,64:128],normalized_feature[4,64:128] , color='b')
    plt.xlabel('SF-mean')
    plt.ylabel('ZCR-mean')
    plt.subplot(2,3,3)
    plt.scatter(normalized_feature[2,0:64],normalized_feature[3,0:64] , color='r')
    plt.scatter(normalized_feature[2,64:128],normalized_feature[3,64:128] , color='b')
    plt.xlabel('RMS-mean')
    plt.ylabel('RMS-std')   
    plt.subplot(2,3,4)
    plt.scatter(normalized_feature[5,0:64],normalized_feature[7,0:64] , color='r')
    plt.scatter(normalized_feature[5,64:128],normalized_feature[7,64:128] , color='b')
    plt.xlabel('ZCR-std')
    plt.ylabel('SCR-std') 
    plt.subplot(2,3,5)
    plt.scatter(normalized_feature[1,0:64],normalized_feature[9,0:64] , color='r')
    plt.scatter(normalized_feature[1,64:128],normalized_feature[9,64:128] , color='b')
    plt.xlabel('SC-std')
    plt.ylabel('SF-std') 
    fig.tight_layout()

if __name__ == "__main__":
    path='/Users/yuyifei/Desktop/music_speech'
    visualize_features(path)
  
    
    
    """music_pair1=(normalized_feature[0:,0:64],normalized_feature[6:,0:64])
    speech_pair1=(normalized_feature[0:,64:128],normalized_feature[6:,64:128])
    data = (music_pair1, speech_pair1)
    colors = ("red", "blue")
    groups = ("music", "speech")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, axisbg="1.0")
    for data, color, group in zip(data, colors, groups):
        x, y = data
        ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)
    plt.title('Matplot scatter plot')
    plt.legend(loc=2)
    plt.show()"""

