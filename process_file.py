import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
# from pydub import AudioSegment
import librosa.display
import pickle
from sklearn import preprocessing
from keras.models import load_model
from sklearn.preprocessing import StandardScaler


def get_audio_data(file_name, data_type):
    # label_filename = file_name + '_labels.txt'
    filename = file_name + '.wav'
    print("\nloading ", filename, " from your folder...\n")
    audio_segment, sampling_rate = librosa.load(filename)
    # labels = pd.read_csv(label_filename, delimiter="\t", names=["start", "end", "label"])
    if data_type == 'mean_sd':
        return split_clip_mean_sd(audio_segment, sampling_rate)
    # else:  # if data_type is fourier
    #     return split_clip_fourier(audio_segment, sampling_rate)


def split_clip_mean_sd(audio_segment, sr):
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

            rms = librosa.feature.rmse(y=new_clip, frame_length=1, hop_length=1).flatten()
            sd = np.std(rms)
            surr_rms = librosa.feature.rms(y=surr_clip).flatten()
            mean_surr = np.mean(surr_rms)

            if i % 100 == 0:
                print("adding row ", i, " of ", num_i, " to the DataFrame")

            data = data.append({'clip': np.abs(librosa.stft(new_clip)).flatten(), 'sd': sd, 'mean': mean_surr}, ignore_index=True)
        start += 500
        i += 1
    return data


# for testing purposes, displays the wave file visually
def display_segmented_audio(filename, labels):
    filename = filename + '.wav'
    data, sampling_rate = librosa.load(filename)
    plt.figure(figsize=(12, 4))
    Time = np.linspace(0, len(data) / sampling_rate, num=len(data))
    plt.title('Audio Segment: ' + filename)
    # clip_len = librosa.get_duration(data, sampling_rate) * 1000  # length in milliseconds
    clip_len = librosa.get_duration(data, sampling_rate)  # length in milliseconds
    start = 0
    i = 0
    # num_i = int(clip_len / 500)
    while start < clip_len:
        # end = start + 500
        end = start + .5
        if end <= clip_len:
            if labels[i] == 2:
                label_color = 'blue'
            elif labels[i] == 3:
                label_color = 'red'
            else:
                label_color = 'white'
            plt.axvspan(start, end, alpha=1, color=label_color, lw=0)
        # start += 500
        start += .5
        i += 1
    import matplotlib.patches as mpatches
    blue_patch = mpatches.Patch(color='blue', label='Single Voice')
    red_patch = mpatches.Patch(color='red', label='Multiple Voice')
    white_patch = mpatches.Patch(color='white', label='Other')

    plt.legend(handles=[blue_patch, red_patch, white_patch])
    print(data)
    plt.plot(Time, data, color="black")
    plt.show()


file_name = 'track2'
classifier = load_model('mean_sd_ann.h5')
audio_data = get_audio_data(file_name, 'mean_sd')
print("Extracting mean/SD data from the DataFrame...\n\n")
# convert mean/SD results to lists so they can be passed as x values to the neural network
clips = pd.DataFrame(audio_data.iloc[:, 0].values.tolist())
sds = pd.DataFrame(audio_data[['sd']].values.tolist())  # .iloc[:, 3]
means = pd.DataFrame(audio_data[['mean']].values.tolist())  # .iloc[:, 2]
print("combining the mean/SD data...\n\n")
newData = np.concatenate((clips, sds, means), axis=1)
print("defining x and y values...\n\n")
pickle_in = open('scalar.pickle','rb')
sc = pickle.load(pickle_in)
newData = sc.fit_transform(newData)
X = newData
y_pred = classifier.predict(X)
y_pred = np.argmax(y_pred, axis=1, out=None)
display_segmented_audio(file_name, y_pred)

