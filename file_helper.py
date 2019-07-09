import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# from audio_file_obj import AudioFile

FOLDER_PATH = 'C:\\Users\\rjoyh\\Desktop\\machine learning\\'
DATA_FOLDER_PATH = 'C:\\Users\\rjoyh\\Desktop\\machine learning\\data\\'
TRIAL_FOLDER_NAMES = ['anns', 'confusion', 'loss', 'output']


def base_filename(file_with_ext):
    if file_with_ext[-4:] == '.wav':
        return file_with_ext[:-4]
    elif file_with_ext[-11:] == '_labels.txt':
        return file_with_ext[:-11]
    elif file_with_ext[-7:] == '.pickle':
        return file_with_ext[:-7]


def file_ext(file_with_ext):
    if file_with_ext[-4:] == '.wav':
        return '.wav'
    elif file_with_ext[-11:] == '_labels.txt':
        return '.txt'
    elif file_with_ext[-7:] == '.pickle':
        return '.pickle'


def add_wav(filename):
    return filename + '.wav'


def add_txt(filename):
    return filename + '_labels.txt'


def add_pickle(filename):
    return filename + '.pickle'


def get_valid_audio_files():
    has_wav = []
    has_txt = []
    has_wav_and_txt = []
    for r, d, f in os.walk(FOLDER_PATH):
        for file in f:
            if file_ext(file) == '.wav':
                has_wav.append(base_filename(file))
            elif file_ext(file) == '.txt':
                has_txt.append(base_filename(file))
        break  # don't enter subdirectories
    for file in has_wav:
        if file in has_txt:
            has_wav_and_txt.append(file)
    return has_wav_and_txt


def get_valid_data_files():
    for r, d, f in os.walk(DATA_FOLDER_PATH):
        return f


def get_last_data_num():
    num = 000
    f = None
    for r, d, f in os.walk(DATA_FOLDER_PATH):
        for file in f:
            if file_ext(file) == '.pickle' and file[0:4] == 'data':
                num = int(file[4:7])
        break  # don't enter subdirectories
    if 'data{:02d}.pickle'.format(num) not in f:
        return num
    else:
        return None


def open_pickled_file(filename):
    obj = None
    if filename[0:4] == 'data':
        pickle_in = open(DATA_FOLDER_PATH + filename, "rb")
        obj = pickle.load(pickle_in)
    return obj


def create_trial_folder(trial_num):
    print('creating folders\n')
    for folder in TRIAL_FOLDER_NAMES:
        dir_name = 'trials' + str(trial_num) + '/' + folder
        # Create target directory & all intermediate directories if don't exists
        os.makedirs(dir_name)


import librosa
MILL_TO_SEC = 1000


def get_label(start, end, labels):
    end = end / MILL_TO_SEC
    start = start / MILL_TO_SEC
    last_label = labels.shape[0]
    for i in range(0, last_label):
        if labels.loc[i, 'end'] >= end:  # if the clip ends before the current label does
            if labels.loc[i, 'start'] <= start:  # if the clip starts after the current label does
                return labels.loc[i, 'label']
            else:  # fix this to account for incomplete labels
                len_left = labels.loc[i - 1, 'end'] - labels.loc[i - 1, 'start']
                len_right = labels.loc[i, 'end'] - labels.loc[i, 'start']
                if len_left >= len_right:
                    return labels.loc[
                        i - 1, 'label']  # this is a transition label (the clip contains multiple labels)
                else:
                    return labels.loc[i, 'label']  # this is a transition label (the clip contains multiple labels)
    return -1  # remove rows with missing labels


def build_dataframe(filename, clip_len, data_types=None, clip_size=500):
    if data_types is None:
        data_types = ['label']
    labels = pd.read_csv(filename, delimiter="\t", names=["start", "end", "label"])
    start = 0
    i = 0
    num_i = int(clip_len / clip_size)
    data = pd.DataFrame()
    leftover = 0
    data_dict = {}

    while start + clip_size < clip_len:
        end = start + clip_size

        if 'label' in data_types:
            seg_label = get_label(start, end, labels)  # find label for clip
            data_dict['label'] = seg_label

        if i % 100 == 0:
            print("adding row ", i, " of ", num_i, " to the DataFrame")

        # at this point, all values should be None except those that we want
        data = data.append(data_dict, ignore_index=True)

        start += clip_size
        i += 1

    return data


def compare_labels(filename, filenames):
    audio, sr = librosa.load(add_wav(filename))
    clip_len = librosa.get_duration(audio, sr) * MILL_TO_SEC  # length in milliseconds

    labels1 = build_dataframe(filenames[0], clip_len).label
    labels2 = build_dataframe(filenames[1], clip_len).label
    # labels3 = build_dataframe(filenames[2], clip_len).label
    # labels4 = build_dataframe(filenames[3], clip_len).label

    compare_two_labels(labels1, labels2, filenames[0], filenames[1])
    # compare_two_labels(labels1, labels3, filenames[0], filenames[2])
    # compare_two_labels(labels1, labels4, filenames[0], filenames[3])
    # compare_two_labels(labels2, labels3, filenames[1], filenames[2])
    # compare_two_labels(labels2, labels4, filenames[1], filenames[3])
    # compare_two_labels(labels3, labels4, filenames[2], filenames[3])


def compare_two_labels(labels1, labels2, name1, name2):
    from sklearn.metrics import confusion_matrix
    labels = [0, 1, 2, 3]

    cm = confusion_matrix(labels1, labels2, labels)

    equal = 0
    for i in range(0,labels2.size):
        if labels1[i] == labels2[i]:
            equal += 1

    total = labels2.size

    percent_sim = equal / total

    fig2 = plt.figure(2)
    ax = fig2.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title("Comparing Labels: " + name1 + " " + name2)
    plot_acc = "{:.2f}%".format((percent_sim) * 100)
    plt.text(1.5, 0.5, plot_acc, horizontalalignment='center', color='white')
    fig2.colorbar(cax)
    ax.set_xticklabels([''] + list(labels))
    ax.set_yticklabels([''] + list(labels))
    plt.xlabel(name1)
    plt.ylabel(name2)
    plt.savefig('compare {} with {}.png'.format(name1, name2))
    plt.clf()


# compare_labels('track10', ['track10_labels_Celine.txt', 'track10_labels_Rachel.txt'])
# compare_labels('track10', ['track10_labels.txt', 'track10_labels_Jordan.txt', 'track10_labels_Shahzeb.txt', 'track10_labels_Matt.txt'])


# for testing purposes, displays the wave file visually
def display_segmented_audio(filename, label_filename):
    labels = pd.read_csv(label_filename, delimiter="\t", names=["start", "end", "label"])

    data, sampling_rate = librosa.load(filename)
    plt.figure(figsize=(20, 4))
    Time = np.linspace(0, len(data)/sampling_rate, num=len(data))

    plt.title('Audio Segment: {} and {}'.format(filename, label_filename))

    last_label = labels.shape[0]
    for i in range(0, last_label):
        if labels.loc[i, 'label'] == 1:
            label_color = 'blue'
        elif labels.loc[i, 'label'] == 2:
            label_color = 'red'
        elif labels.loc[i, 'label'] == 3:
            label_color = 'green'
        else:
            label_color = 'white'
        plt.axvspan(labels.loc[i, 'start'], labels.loc[i, 'end'], alpha=1, color=label_color, lw=0)
    import matplotlib.patches as mpatches
    white_patch = mpatches.Patch(color='white', label='Silence')
    blue_patch = mpatches.Patch(color='blue', label='Lecturer')
    red_patch = mpatches.Patch(color='red', label='Student(s)')
    green_patch = mpatches.Patch(color='green', label='Other')

    plt.legend(handles=[white_patch, blue_patch, red_patch, green_patch])
    plt.plot(Time, data, color="black")
    plt.show()


# display_segmented_audio('track2.wav', 'track2_labels.txt')

from audio_file_obj import DEFAULT_CLIP_SIZE
from audio_file_obj import MILL_TO_SEC


def create_label_track(filename, labels, clip_size=None):
    f = open(filename, "w+")
    f.close()
    if clip_size is None:
        clip_size = DEFAULT_CLIP_SIZE
    start = 0
    end = 0
    label = -1
    i = 0
    for clip in labels:
        i += 1
        if clip == label:
            end = end + clip_size
            label = clip
            if i == labels.shape[0]:
                add_label_row(filename, start / MILL_TO_SEC, end / MILL_TO_SEC, label)
        else:
            print('else')
            add_label_row(filename, start/MILL_TO_SEC, end/MILL_TO_SEC, label)
            start = end
            end = start + clip_size
            label = clip


def add_label_row(filename, start, end, label):
    if label != -1:
        f = open(filename, "a")
        print("{}\t{}\t{}\n".format(start, end, label))
        f.write("{}\t{}\t{}\n".format(start, end, label))
        f.close()

