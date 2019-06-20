import os
import pickle
FOLDER_PATH = 'C:\\Users\\rjoyh\\Desktop\\machine learning\\'
DATA_FOLDER_PATH = 'C:\\Users\\rjoyh\\Desktop\\machine learning\\data\\'


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
