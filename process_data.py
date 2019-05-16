import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
# from pydub import AudioSegment
import librosa.display
import pickle
from sklearn import preprocessing


# takes start and end times in milliseconds, and a data frame of handwritten labels as arguments
# returns the label that accompanies the given timestamp
# labels are as follows:
#   *  3 means other
#   *  0 means no voice
#   *  1 means one voice
#   *  2 means multiple voice
#   * -1 means transition (unusable)
def get_label(start, end, labels):
    end = end / 1000
    start = start / 1000
    last_label = labels.shape[0]
    for i in range(0, last_label):
        if labels.loc[i, 'end'] >= end:
            if labels.loc[i, 'start'] <= start:
                return labels.loc[i, 'label']
            else:
                return -1  # this is a transition label (the clip contains multiple labels)


def split_clip_mean_sd(audio_segment, sr, labels):
    clip_len = librosa.get_duration(audio_segment,sr)*1000  # length in milliseconds
    full_rms = librosa.feature.rmse(y=audio_segment, frame_length=1, hop_length=1).flatten()
    full_sd = np.std(full_rms)
    full_mean = np.mean(full_rms)
    start = 0
    i = 0
    num_i = int(clip_len / 500)
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

            seg_label = get_label(start, end, labels)  # find label for clip

            rms = librosa.feature.rmse(y=new_clip, frame_length=1, hop_length=1).flatten()
            sd = np.std(rms)
            surr_rms = librosa.feature.rms(y=surr_clip).flatten()
            mean_surr = np.mean(surr_rms)

            if i % 100 == 0:
                print("adding row ", i, " of ", num_i, " to the DataFrame")

            data = data.append({'clip': new_clip, 'label': seg_label, 'sd': sd, 'mean': mean_surr}, ignore_index=True)
        start += 500
        i += 1
    # print(data[["label", "sd", "mean"]])
    print("overall mean is {:05.4f} and dataframe mean is {:05.4f}" .format(full_mean, np.mean(data[["mean"]].values.flatten())))
    print("overall SD   is {:05.4f} and dataframe SD   is {:05.4f}\n" .format(full_sd, np.mean(data[["sd"]].values.flatten())))
    return data


# splits audio segment into .5 second clips
# performs stft on each clip and the surrounding 10 seconds of that clip
# returns a data frame of both stft's, start and end (in milliseconds), and the label for each clip
def split_clip_fourier(audio_segment, sr, labels):
    clip_len = librosa.get_duration(audio_segment,sr)*1000  # length in milliseconds
    start = 0
    i = 0
    num_i = int(clip_len / 500)
    data = pd.DataFrame()
    while start < clip_len:
        end = start + 500
        if end <= clip_len:
            new_clip = audio_segment[start:end]  # creates a .5 second clip
            if start - 1500 < 0:
                surr_start = start
            else:
                surr_start = start - 1500
            if end + 1500 > clip_len:
                surr_end = clip_len  # we may want to change this so that it wil ALWAYS be 10 seconds
            else:
                surr_end = end + 1500
            surr_clip = audio_segment[int(surr_start):int(surr_end)]  # creates 10 second clip of the surrounding audio

            seg_label = get_label(start, end, labels)  # find label for clip

            fourier_res = np.abs(librosa.stft(new_clip)).flatten()
            fourier_surr_res = np.abs(librosa.stft(surr_clip)).flatten()

            if i % 100 == 0:
                print("adding row ", i, " of ", num_i, " to the DataFrame")
            data = data.append({'start': start, 'end': end, 'label': seg_label, 'fourier': fourier_res,
                                'surr_fourier': fourier_surr_res}, ignore_index=True)
        start += 500
        i += 1
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


def get_data(file_name, data_type):
    label_filename = file_name + '_labels.txt'
    filename = file_name + '.wav'
    print("\nloading ", filename, " from your folder...\n")
    audio_segment, sampling_rate = librosa.load(filename)
    labels = pd.read_csv(label_filename, delimiter="\t", names=["start", "end", "label"])
    if data_type == 'mean_sd':
        return split_clip_mean_sd(audio_segment, sampling_rate, labels)
    else: # if data_type is fourier
        return split_clip_fourier(audio_segment, sampling_rate, labels)


def get_all_data(file_names, data_type):
    min_max_scaler = preprocessing.MinMaxScaler()
    total_data = pd.DataFrame()
    if data_type == 'mean_sd':
        for file in file_names:
            curr_set = get_data(file, data_type)  # returns a dataframe
            sds = curr_set[["sd"]].values  # returns a numpy array
            means = curr_set[["mean"]].values  # returns a numpy array
            sds_scaled = min_max_scaler.fit_transform(sds)
            means_scaled = min_max_scaler.fit_transform(means)
            curr_set[["sd"]] = sds_scaled
            curr_set[["mean"]] = means_scaled
            total_data = total_data.append(curr_set, ignore_index=True)
        print("dropping NaN and -1 values...\n")
        for i, row in total_data.iterrows():
            if row.label == -1:
                total_data.drop(total_data.index[i])
        total_data.dropna()
    elif data_type == 'fourier':
        for file in file_names:
            curr_set = get_data(file, data_type)  # returns a dataframe
            total_data = total_data.append(curr_set, ignore_index=True)
        print("dropping NaN and -1 values...\n")
        for i, row in total_data.iterrows():
            if row.label == -1:
                total_data.drop(total_data.index[i])
        total_data.dropna()
    return total_data


def get_ann_data(data_type):
    valid_type = {'mean_sd', 'fourier'}
    if data_type not in valid_type:
        raise ValueError("get_ann_data: data_type must be one of {}.".format(valid_type))
    return get_all_data(["track1", "track2"], data_type)


def pickle_data(data_type):
    valid_type = {'mean_sd', 'fourier'}
    if data_type not in valid_type:
        raise ValueError("pickle_data: data_type must be one of {}.".format(valid_type))
    data_to_pickle = get_ann_data(data_type)
    pickle_filename = data_type + '.pickle'
    with open(pickle_filename, 'wb') as f:
        pickle.dump(data_to_pickle, f)


# pass in mean_sd or fourier
# pickle_data('fourier')
