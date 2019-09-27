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


def normalize_zscore(FeatureData):
    mean1= np.mean(FeatureData,axis=1)
    std1= np.std(FeatureData,axis=1)
    num_feature,num_sample=np.shape(FeatureData)
    normalized_feature=np.zeros((num_feature,num_sample))
    for i in range(num_feature):
        normalized_feature[i,:]=(FeatureData[i,:]-mean1[i])/std1[i]
    return normalized_feature

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

 

# def get_f0_from_acf(r, fs):
#     peaks = find_peaks(r)[0]
#     # plt.plot(r)
#     # plt.plot(peaks, r[peaks], 'rs')
#     # plt.show()
#     if len(peaks) >= 2:
#         p = sorted(r[peaks])[::-1]
#         sorted_arg = np.argsort(r[peaks])[::-1]
#         f0 = fs / abs(peaks[sorted_arg][1] - peaks[sorted_arg][0])
#         return f0
#     return 0

# # 4
# def track_pitch_acf(x, blockSize, hopSize, fs):
#     blocked_x, timeInSec = block_audio(x, blockSize, hopSize, fs)
#     frequencies = []
#     for b in blocked_x:
#         acf = comp_acf(b)
#         f0 = get_f0_from_acf(acf, fs)
#         frequencies.append(f0)
#     return [np.array(frequencies), timeInSec]

# ########################### B. Evaluation ###########################

# def gen_sin(f1=441, f2=882, fs=44100):
#     t1 = np.linspace(0, 1, fs)
#     t2 = np.linspace(1, 2, fs)
#     sin_441 = np.sin(2 * np.pi * 441 * t1)
#     sin_882 = np.sin(2 * np.pi * 882 * t2)
#     sin = np.append(sin_441, sin_882)
#     return sin

# def code_for_B1():
#     fs = 44100
#     f1 = 441
#     f2 = 882
#     sin = gen_sin(f1, f2, fs)
#     [frequencies, timeInSec] = track_pitch_acf(sin, 1024, 512, fs)
#     error = np.zeros(len(timeInSec))
#     error[:len(timeInSec) // 2] += f1
#     error[len(timeInSec) // 2 :] += f2
#     error = np.abs(error - frequencies)

#     # Plot
#     line1, = plt.plot(timeInSec, error)
#     line2, = plt.plot(timeInSec, frequencies)
#     plt.legend((line1, line2), ("error", "frequencies (f0)"))
#     plt.title("Resulting f0 and error in Hz")
#     plt.xlabel("samples (sec)")
#     plt.ylabel("Frequency (Hz)")
#     plt.show()

# # 1
# code_for_B1()

# def convert_freq2midi(freqInHz):
#     return 69 + 12 * np.log2(freqInHz / 440.0)

# def freq2cent(freqInHz):
#     return 1200 * np.log2(freqInHz / 440.0)

# def eval_pitchtrack(estimateInHz, groundtruthInHz):
#     centError = []
#     for i in range(len(groundtruthInHz)):
#         if groundtruthInHz[i] != 0:
#             if estimateInHz[i] != 0:
#                 centError.append(freq2cent(estimateInHz[i]) - freq2cent(groundtruthInHz[i]))
#             elif estimateInHz[i] == 0:
#                 centError.append(-freq2cent(groundtruthInHz[i]))
#     centError = np.array(centError)
#     rms = np.sqrt(np.mean(np.square(centError)))
#     return rms

# def run_evaluation(complete_path_to_data_folder):
#     file_path = os.path.join(complete_path_to_data_folder, '*.wav')
#     wav_files = [f for f in glob.glob(file_path)]
#     errCentRms = []
#     for wav_file in wav_files:
#         name = os.path.split(wav_file)[1].split('.')[0]
#         txt_file = os.path.join(complete_path_to_data_folder, name+'.f0.Corrected.txt')
#         with open(txt_file) as f:
#             annotations = f.readlines()
#         for i in range(len(annotations)):
#             annotations[i] = list(map(float, annotations[i][:-2].split('     ')))
#         annotations = np.array(annotations)
#         fs, audio = read(wav_file)
#         freq, timeInSec = track_pitch_acf(audio, 2048, 512, fs)
#         trimmed_freq = np.ones(freq.shape)
#         trimmed_annotations = np.ones(freq.shape)
#         for i in range(len(freq)):
#             if annotations[i, 2] > 0:
#                 trimmed_freq[i] = freq[i]
#                 trimmed_annotations[i] = annotations[i, 2]
#         # plt.plot(trimmed_freq)
#         # plt.plot(trimmed_annotations)
#         # plt.show()
#         errCentRms.append(eval_pitchtrack(trimmed_freq, trimmed_annotations))
#     errCentRms = np.array(errCentRms)
#     # print(errCentRms)
#     return np.mean(errCentRms)

# print("Overall errCentRms:",run_evaluation("trainData/"))

# ########################### C. Bonus ###########################

# def block_audio_mod(x, blockSize, hopSize, fs):
#     i = 0
#     xb = []
#     timeInSec = []
#     while i < len(x):
#         timeInSec.append(i / fs)
#         chunk = x[i: i + blockSize]
#         if len(chunk) != blockSize:
#             chunk = np.append(chunk, np.zeros(blockSize - len(chunk)))
#             xb.append(chunk*np.hamming(blockSize))
#             break
#         else:
#             xb.append(chunk*np.hamming(blockSize))
#         i += hopSize

#     return [np.array(xb), np.array(timeInSec)]

# def get_f0_from_acfmod(r, fs):
#     peaks = find_peaks(r, height=0, distance=50)[0]
#     if len(peaks) >= 2:
#         sorted_arg = np.argsort(r[peaks])[::-1]
#         px, py = parabolic(r,peaks[sorted_arg][0])
#         return fs/px
#     return 0

# def track_pitch_acfmod(x, blockSize, hopSize, fs):
#     blocked_x, timeInSec = block_audio_mod(x, blockSize, hopSize, fs)
#     frequencies = []
#     for b in blocked_x:
#         acf = comp_acf(b)
#         f0 = get_f0_from_acfmod(acf, fs)
#         frequencies.append(f0)
#     frequencies = np.array(frequencies)
#     frequencies = medfilt(frequencies,kernel_size=9)
#     return [frequencies, timeInSec]

# def parabolic(f, x):
#     """Quadratic interpolation for estimating the true position of an
#     inter-sample maximum when nearby samples are known.
#     f is a vector and x is an index for that vector.
#     Returns (vx, vy), the coordinates of the vertex of a parabola that goes
#     through point x and its two neighbors.
#     """
#     xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
#     yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
#     return (xv, yv)

# # fs = 44100
# # f1 = 441
# # f2 = 882
# # sin = gen_sin(f1, f2, fs)
# # [frequencies, timeInSec] = track_pitch_acfmod(sin, 441, 441, fs)
# # error = np.zeros(len(timeInSec))
# # error[:len(timeInSec) // 2] += f1
# # error[len(timeInSec) // 2 :] += f2
# # error = np.abs(error - frequencies)
# # plt.plot(timeInSec, error)
# # plt.plot(timeInSec, frequencies)
# # plt.show()

# def run_evaluation_mod(complete_path_to_data_folder):
#     file_path = os.path.join(complete_path_to_data_folder, '*.wav')
#     wav_files = [f for f in glob.glob(file_path)]
#     errCentRms = []
#     for wav_file in wav_files:
#         name = os.path.split(wav_file)[1].split('.')[0]
#         txt_file = os.path.join(complete_path_to_data_folder, name+'.f0.Corrected.txt')
#         with open(txt_file) as f:
#             annotations = f.readlines()
#         for i in range(len(annotations)):
#             annotations[i] = list(map(float, annotations[i][:-2].split('     ')))
#         annotations = np.array(annotations)
#         fs, audio = read(wav_file)
#         freq, timeInSec = track_pitch_acfmod(audio, 2048, 512, fs)
#         trimmed_freq = np.ones(freq.shape)
#         trimmed_annotations = np.ones(freq.shape)
#         for i in range(len(freq)):
#             if annotations[i, 2] > 0:
#                 trimmed_freq[i] = freq[i]
#                 trimmed_annotations[i] = annotations[i, 2]
#         plt.plot(trimmed_freq, label='frequency (Hz)')
#         plt.plot(trimmed_annotations, label='annotation')
#         plt.legend()
#         plt.show()
#         errCentRms.append(eval_pitchtrack(trimmed_freq, trimmed_annotations))
#     errCentRms = np.array(errCentRms)
#     # print(errCentRms)
#     return np.mean(errCentRms)

# print("Overall errCentRms (mod):",run_evaluation_mod("trainData/"))