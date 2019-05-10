import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
# from pydub import AudioSegment
import librosa.display
import random


# takes start and end times in milliseconds, and a data frame of handwritten labels as arguments
# returns the label that accompanies the given timestamp
# labels are as follows:
#   * -1 means transition or other
#   *  0 means no voice
#   *  1 means one voice
#   *  2 means multiple voice
def get_label(start, end, labels):
    end = end / 1000
    start = start / 1000
    last_label = labels.shape[0]
    for i in range(0, last_label):
        if labels.loc[i, 'end'] >= end:
            if labels.loc[i, 'start'] <= start:
                return labels.loc[i, 'label']
            else:
                return -1  # this is a transition label


# splits audio segment into .5 second clips
# performs stft on each clip and the surrounding 10 seconds of that clip
# returns a data frame of both stft's, start and end (in milliseconds), and the label for each clip
def split_clip(audio_segment, sr, labels):
    # clip_len = len(audio_segment)  # length in milliseconds
    clip_len = librosa.get_duration(audio_segment,sr)*1000  # length in milliseconds
    start = 0
    data = pd.DataFrame()
    while start < clip_len:
        end = start + 500
        if end <= clip_len:
            new_clip = audio_segment[start:end]  # creates a .5 second clip
            if start - 10000 < 0:
                surr_start = start
            else:
                surr_start = start - 10000
            if end + 10000 > clip_len:
                surr_end = clip_len  # we may want to change this so that it wil ALWAYS be 10 seconds
            else:
                surr_end = end + 10000
            surr_clip = audio_segment[int(surr_start):int(surr_end)]  # creates 10 second clip of the surrounding audio
            # find label for clip
            seg_label = get_label(start, end, labels)
            fourier_res = stft_audio(new_clip).flatten()
            # data = data.append({'start': start, 'end': end, 'label': seg_label, 'fourier': fourier_res},
            #                    ignore_index=True)
            data = data.append({'start': start, 'end': end, 'label': seg_label, 'fourier': fourier_res,
                                'surr_fourier': stft_audio(surr_clip).flatten()}, ignore_index=True)
        start += 500
    # print("fourier size is: ", data.at[50, 'fourier'].shape)
    # print("surrounding fourier size is: ", data.at[20, 'surr_fourier'].shape)
    # display_stft_audio(data.at[50, 'fourier'])
    # display_stft_audio(data.at[50, 'surr_fourier'])
    # matrix1 = data.at[50, 'fourier']
    # matrix2 = data.at[50, 'surr_fourier']
    # display_stft_audio(matrix1)
    # display_stft_audio(matrix2)
    return data


# for testing purposes, displays the wave file visually
def display_segmented_audio(filename, label_filename):
    labels = pd.read_csv(label_filename, delimiter="\t", names=["start", "end", "label"])

    data, sampling_rate = librosa.load(filename)
    plt.figure(figsize=(12, 4))
    Time = np.linspace(0, len(data)/sampling_rate, num=len(data))

    plt.title('Audio Segment: ' + filename)

    last_label = labels.shape[0]
    for i in range(0,last_label):
        if labels.loc[i,'label'] == 1:
            label_color = 'blue'
        elif labels.loc[i,'label'] == 2:
            label_color = 'red'
        else:
            label_color = 'white'
        plt.axvspan(labels.loc[i, 'start'], labels.loc[i, 'end'], alpha=0.5, color=label_color, lw=0)

    plt.plot(Time, data)
    plt.show()


# returns the stft matrix for the given audio segment
# outputs a real-valued matrix  spec of size frequency x time
def stft_audio(data):
    stft_matrix = np.abs(librosa.stft(data))
    return stft_matrix


# for testing, displays the given stft matrix as a spectrogram
def display_stft_audio(stft_matrix):
    librosa.display.specshow(librosa.amplitude_to_db(stft_matrix, ref = np.max), y_axis = 'log', x_axis = 'time')
    plt.title('Power spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()


# for testing, probably won't use, gives decibels
def spectrogram_in_hz(filename):
    data, sr = librosa.load(filename)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
    librosa.display.specshow(D, y_axis='linear')
    plt.show()


def create_feature_sets_and_labels(clips, test_size=0.6):
    # random.shuffle(data)
    data = np.array(clips)

    testing_size = int(test_size*len(data))
    # print("start:\n", data[:,0][:-testing_size])
    # print("fourier:\n", data[:,1][:-testing_size])
    # print("label:\n", data[:,2][:-testing_size])
    # print("end:\n", data[:,3][:-testing_size])
    # print("fourier is: ", clips.at[0, 'fourier'])
    # print("label is: ", clips.at[0, 'label'])

    train_x = list(data[:,1][:-testing_size])  # fouriers
    train_y = list(data[:,2][:-testing_size])  # labels
    test_x = list(data[:,1][-testing_size:])
    test_y = list(data[:,2][-testing_size:])

    return train_x,train_y,test_x,test_y


def get_data():
    label_filename = 'single_double_labels.txt'
    filename = 'single_double.wav'
    audio_segment, sampling_rate = librosa.load(filename)
    labels = pd.read_csv(label_filename, delimiter="\t", names=["start", "end", "label"])
    return split_clip(audio_segment, sampling_rate, labels)


print(get_data().at[20, 'fourier'].shape)
print(type(get_data().at[20, 'fourier']))
# spectrogram_in_hz('single_double.wav')
# stft_audio('single_double.wav')
# display_segmented_audio('single_double.wav', 'single_double_labels.txt')
# display_segmented_audio('190107_001_Binaural.m4a', 'single_double_labels.txt')
# # doesnt seem to work for non-wave files
#
#
# # GOALS
#
# # open wav or m4a files
# # normalize: number of channels, sampling rate, bit depth
# # split into half second clips
# # assign each clip with its value (0 = no voice, 1 = lecturer, 2 = multiple voice, 3 = transition?)
