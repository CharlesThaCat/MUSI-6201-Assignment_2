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
        # vsc = np.sum(magnitudes*freqs) / np.sum(magnitudes)
        vsc_matrix.append(vsc)
    return np.array(vsc_matrix)

def extract_rms(xb):
    rms_matrix = []
    for b in xb:
        length = len(b)
        rms = np.sqrt(np.divide(np.sum(np.square(b)),length))
        if rms < 1e-5:
            rms = 1e-5
        rms = 20*np.log10(rms)
        rms_matrix.append(rms)
    # print(rms_matrix)
    return np.array(rms_matrix)

def extract_zerocrossingrate(xb):
    vzc_matrix = []
    for b in xb:
        length = len(b)
        temp = np.sum(np.abs(np.diff(np.sign(b))))
        vzc = temp / (2*length)
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

if __name__ == "__main__":
    blockSize = 1024
    hopSize = 256
    path = 'C:/Users/bhxxl/Downloads/music_speech/music_speech/music_wav'
    featureData = get_feature_data(path, blockSize, hopSize)

    print(featureData.shape)

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