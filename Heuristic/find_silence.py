import librosa
import sys
import numpy as np
import speech_recognition as sr

sys.path.append('../')
from ML.file_helper import add_wav


# Ctrl + F (TEST ME) to find functions that still need to be tested
# Ctrl + F (TO DO) to proceed to next steps

class LectureAudio:
    """
    A lecture audio object that holds information about an imported wav file

    Attributes:
        base_filename (string): the name of the file being analyzed, WITHOUT the .wav extension
        wav_filename (string): the name of the file being analyzed, WITH the .wav extension
        audio_arr (np.ndarray): array of floats representing audio signal
        sr (int): the sampling rate used to store the audio signal
        trimmed_audio (np.ndarray): audio_arr with silence trimmed off beginning and end
    """

    def __init__(self, filename, duration=None):
        """
        Initializes an instance of the LectureAudio class.

        Args:
            filename (string): without file extension, the name of the file you want to analyze
            duration (int): length in seconds of audio to be loaded in
        """

        self.base_filename = filename
        self.wav_filename = add_wav(filename)

        # if no duration was given, just load the entire wv file
        if duration is None:
            audio_arr, sr = librosa.load(self.wav_filename)

        # otherwise only load {duration} seconds of the wav file
        else:
            audio_arr, sr = librosa.load(self.wav_filename,  duration=duration)

        # entire audio
        self.audio_arr = audio_arr

        # default sr=22050 for librosa.load
        self.sr = sr

        # audio with leading and trailing silences trimmed
        self.trimmed_audio = self.trim_ends()

    def trim_ends(self, verbose=False):
        """
        Trims the silence off of the beginning and end of the audio

        Returns:
            np.ndarray: audio with leading and trailing silence removed
        """

        # Trim the beginning and ending silence
        trimmed_audio, index = librosa.effects.trim(self.audio_arr, top_db=40)

        # Print the percent of audio left after trimming
        if verbose:
            prev_dur = librosa.get_duration(self.audio_arr)
            new_dur = librosa.get_duration(trimmed_audio)
            percent_left = (new_dur / prev_dur) * 100
            print('\nThe audio has been trimmed down to {:0.2f}% the original size.\n'.format(percent_left))

        return trimmed_audio

    def split_on_silence(self, threshold, frame_length, hop_length):
        """
        Splits the audio into intervals of sound with silent beginnings and endings

        Args:
            threshold (int): the threshold (in decibels) below reference to consider as silence
            frame_length (int): the (positive integer) number of samples in an analysis window (or frame)
            hop_length (int): the number of samples between successive frames, e.g., the columns of a spectrogram

        Returns:
            np.ndarray: time intervals for lecturing chunks split on silence, given in SAMPLES
        """

        intervals = librosa.effects.split(self.trimmed_audio, top_db=threshold, frame_length=frame_length,
                                          hop_length=hop_length)

        return intervals

    def test_splits(self, frame_lengths, hop_lengths, thresholds):
        """
        Tests different parameters used to split the audio on the silence
        Calls split_on_silence() for each different set of parameters
        User can analyze the printed results and saved label files to determine best params for final_split

        Args:
            frame_lengths (list): possible frame length values to try as argument in split_on_silence()
            hop_lengths (list): possible hop length values to try as argument in split_on_silence()
            thresholds (list): possible threshold values to try as argument in split_on_silence()

        Returns:
            None
        """

        i = 0
        for frame_length in frame_lengths:
            for hop_length in hop_lengths:
                for threshold in thresholds:

                    i += 1

                    # for each set of arguments, split the audio into chunks with leading and trialing silence
                    intervals = self.split_on_silence(threshold, hop_length, frame_length)

                    # print the number of intervals and the args used
                    print('{}: {} intervals found using frame_length={}, hop_length={}, threshold={}.'
                          .format(i, len(intervals), frame_length, hop_length, threshold))

                    # for each chunk, trim the leading and trailing silence using the given threshold
                    intervals = self.trim_chunks(intervals, threshold)

                    # save a txt file with the labels for each set of args
                    self.create_labels(intervals, i)

    def final_split(self, threshold, hop_length, frame_length):
        """
        Splits the audio on silence once you determine the best parameters

        Best for BIS-2A__2019-07-17_12_10.wav:
        *** 1173 intervals found using frame_length=1024, hop_length=2048, threshold=30.
            1125 intervals found using frame_length=2048, hop_length=1024, threshold=35.
            864 intervals found using frame_length=4096, hop_length=2048, threshold=30.

        Args:
            threshold (int): the threshold (in decibels) below reference to consider as silence
            frame_length (int): the (positive integer) number of samples in an analysis window (or frame)
            hop_length (int): the number of samples between successive frames, e.g., the columns of a spectrogram

        Returns:
            np.ndarray: time intervals for start and end of each lecturing chunk, given in SAMPLES
        """

        # split the audio into chunks with leading and trialing silence
        intervals = self.split_on_silence(threshold, hop_length, frame_length)

        # print the number of intervals and the args used
        print('{} intervals found using frame_length={}, hop_length={}, threshold={}.'
              .format(len(intervals), frame_length, hop_length, threshold))

        # for each chunk, trim the leading and trailing silence using the given threshold
        new_intervals = self.trim_chunks(intervals, threshold)

        return new_intervals

    # TEST ME
    def trim_chunks(self, intervals, threshold):
        """
        Trims the leading and trailing silence off each chunk of audio

        Args:
            intervals (np.ndarray): time intervals for start and end of each lecturing chunk, given in SAMPLES
            threshold (int): the threshold (in decibels) below reference to consider as silence

        Return:
            np.ndarray: time intervals for start and end of each lecturing chunk, given in SAMPLES
        """

        # for each chunk, trim the leading and trailing silence using the given threshold
        for j in range(0, len(intervals)):
            interval = intervals[j]
            start = interval[0]
            end = interval[1]

            # grab the audio chunk
            clip = self.trimmed_audio[int(start):int(end)]

            # trim the leading and trailing silence
            trimmed_audio, index = librosa.effects.trim(clip, top_db=threshold)

            # replace the start and end time of the chunk with the trimmed ones
            intervals[j][0] += index[0]
            intervals[j][1] = intervals[j][0] + index[1]

        return intervals

    # TO DO
    # edit this so we glob together intervals without pauses as the same label
    # there might be some helpful functions in ../ML/file_helper.py
    def create_labels(self, intervals, i=None):
        """
        Creates a txt file that can be imported as labels to Audacity
        File will have the columns: start, end, label

        Args:
            intervals (np.ndarray): time intervals for start and end of each lecturing chunk, given in SAMPLES
            i (int): hold the integer val to use in the filename (for creating multiple label files for one audio file

        Returns:
            None
        """

        # if no number was given, just name the file normally
        if i is None:
            filename = '{}_labels.txt'.format(self.base_filename)

        # otherwise, add the number to the end of the file name
        # this is used when multiple tests are being run on the same audio file
        else:
            filename = '{}_labels_{}.txt'.format(self.base_filename, i)

        # open the file to make sure we create it if it isn't there
        f = open(filename, "w+")
        f.close()

        new_intervals = self.glob_labels(intervals)

        # add a label for each interval
        for row in new_intervals:
            self.add_label_row(filename, row[0] / self.sr, row[1] / self.sr, 1)

    # FIX ME
    # we might not be catching the last label
    def glob_labels(self, intervals):
        """
        Combines intervals that are directly adjacent (makes labels more readable in Audacity

        Args:
            intervals (np.ndarray): time intervals for start and end of each lecturing chunk, given in SAMPLES

        Returns:
            np.ndarray: simplified intervals
        """

        label_list = []
        label_start = intervals[0][0]
        label_end = intervals[0][1]

        for i in range(0, len(intervals) - 1):
            current_interval = intervals[i]
            next_interval = intervals[i + 1]
            if next_interval[0] == current_interval[1]:
                label_end = next_interval[1]  # set new interval end to the end of the next interval

                if i == len(intervals) - 2:
                    label_list.append(np.array([label_start, label_end]))

            else:
                label_end = current_interval[1]  # set new interval end to the end of the current interval
                label_list.append(np.array([label_start, label_end]))
                label_start = next_interval[0]

        return np.array(label_list)

    @staticmethod
    def add_label_row(filename, start, end, label):
        """
        Adds a row to the current label txt file
        Will add a row with the columns: start, end, label

        Args:
            filename (string): holds the filename of the txt file to be edited
            start (int): the start value of the label in milliseconds
            end (int): the end value of the label in milliseconds
            label (int): represents whether the current clip is silence (0) or lecture (1)

        Returns:
            None
        """

        # -1 is used as a flag for a label we want to ignore
        if label != -1:
            f = open(filename, "a")
            f.write("{}\t{}\t{}\n".format(start, end, label))
            f.close()

    def ignore_silence(self, intervals, pause):
        """
        Ignore silences that last less than the maximum pause length given

        Args:
            intervals (np.ndarray): time intervals for start and end of each lecturing chunk, given in SAMPLES
            pause (int): maximum length of silence still considered as lecturing, given in SECONDS

        Returns:
            np.ndarray: time intervals for start and end of each lecturing chunk, pauses ignored, given in SAMPLES
        """

        i = 0

        # empty list so we don't have to append an array over and over again (saves memory)
        new_intervals = []

        # add the leading silence to the lecture time if it is less than the pause length
        if intervals[0][0] <= pause and intervals[0][0] != 0:
            new_intervals.append(np.array([0, intervals[0][0]]))

        for i in range(0, len(intervals) - 1):
            speech_interval = intervals[i]
            pause_beginning = speech_interval[1]
            pause_end = intervals[i + 1][0]

            # add each old interval to the list
            new_intervals.append(speech_interval)

            # if the pause between an interval is less than the maximum, assume its still lecture
            if (pause_end - pause_beginning) / self.sr <= pause:
                new_intervals.append(np.array([pause_beginning, pause_end]))

        # add the last interval to the new list
        last_interval = intervals[len(intervals) - 1]
        new_intervals.append(last_interval)  # list must hold times in SAMPLES not SECONDS

        # add the trailing silence to the lecture time if its less than the pause length
        last_interval_end = last_interval[1]  # in samples
        audio_end = librosa.get_duration(self.trimmed_audio)  # in seconds
        audio_end_samples = audio_end * self.sr  # in samples
        if audio_end - (last_interval_end / self.sr) <= pause:
            new_intervals.append(np.array([last_interval_end, audio_end_samples]))  # list must hold times in SAMPLES not SECONDS

        return np.array(new_intervals)

    def save_trimmed_file(self):
        """
        Saves the audio with trimmed leading and trailing silence as a wav file

        Returns:
            None
        """

        filename = '{}_trimmed.wav'.format(self.base_filename)
        librosa.output.write_wav(filename, self.trimmed_audio, self.sr)
        print('{} was successfully saved!'.format(filename))

    def analyze_audio(self, intervals, pause_length):
        """
        Finds the percent of audio remaining after trailing and leading silences were removed
        Finds percent of audio that is lecture vs. silence

        Args:
            intervals (np.ndarray): time intervals for start and end of each lecturing chunk, given in SAMPLES
            pause_length (int): the max length of silence to be ignored (grouped with lecture), given in SECONDS

        Returns:
            float: the percent of time trimmed away from the beginning and end of the audio
            float: the percent of time spent in lecture as compared to the entire trimmed audio
        """

        # Print the durations
        prev_dur = librosa.get_duration(self.audio_arr)
        new_dur = librosa.get_duration(self.trimmed_audio)
        percent_trimmed = 100 - ((new_dur / prev_dur) * 100)
        print('\nRemoved {:0.2f}% of the audio as leading or trailing silence.\n'.format(percent_trimmed))

        talking = 0  # total time in SECONDS that

        # count short pauses as lecture
        adjusted_intervals = self.ignore_silence(intervals, pause_length)

        # save a txt file with the labels
        self.create_labels(adjusted_intervals)

        # add up all the time spent in lecture
        for label in adjusted_intervals:
            talking += (label[1] - label[0]) / self.sr

        # calculate the perent time spent in lecture
        percent_talking = (talking / new_dur) * 100
        print('Of the remaining audio, {:0.2f}% was lecture, and {:0.2f}% was silence.\n'.format(percent_talking,
                                                                                                 100 - percent_talking))

        return percent_trimmed, percent_talking

    # TO DO
    # edit so that it reads the words in label chunks
    # count the number of words
    def analyze_words(self, intervals):
        """
        Finds the number of words and array of words spoken in audio

        Args:
            intervals (np.ndarray): start and end in milliseconds of each non-silent clip in audio

        Returns:
            list: words spoken in lecture
            int: number of words spoken in lecture
        """

        r = sr.Recognizer()

        lecture = sr.AudioFile(self.wav_filename)

        with lecture as source:
            audio = r.record(source, duration=59)

        try:
            words = r.recognize_google(audio)

            # open the file to make sure we create it if it isn't there
            words_file = '{}_words.txt'.format(self.base_filename)

            f = open(words_file, "a+")
            f.write(words)
            f.close()

            return words

        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))

        # return words, num_words

    # TO DO
    # finish this function
    def analyze_questions(self, intervals, words):
        """
        Finds the number of questions asked based on an array of words and intervals of talking

        Args:
            intervals (np.ndarray): start and end in milliseconds of each non-silent clip in audio
            words (list): words spoken during the lecture

        Returns:
            int: number of questions asked during lecture
        """

        # return num_questions
        pass

    # TEST
    # not ready until analyze_words() and analyze_questions() are finished
    def full_analysis(self, intervals, pause_length):
        """
        Analyzes the audio and returns helpful data

        Args:
            intervals (np.ndarray): start and end in milliseconds of each non-silent clip in audio
            pause_length (int): the max length of silence to be ignored (grouped with lecture), given in SECONDS

        Returns:
            int: number of questions asked during lecture
            int: number of words spoken in lecture
            float: the percent of time trimmed away from the beginning and end of the audio
            float: the percent of time spent in lecture as compared to the entire trimmed audio
        """

        percent_trimmed, percent_talking = self.analyze_audio(intervals, pause_length)

        words, num_words = self.analyze_words(intervals)
        num_questions = self.analyze_questions(intervals, words)

        return num_questions, num_words, percent_trimmed, percent_talking


if __name__ == '__main__':

    # select the base filename to be analyzed
    audio_file = 'speech_tester'

    # # create an instance of the LectureAudio class
    # # extract audio info from wav file and trim leading and trailing silence
    # lecture = LectureAudio(audio_file)

    # create an instance of the LectureAudio class
    # only load first 1200 seconds to make tests run faster
    lecture = LectureAudio(audio_file, duration=1200)

    # # run and analyze this test if unsure what params will work best for final split
    # frame_lengths = [1024, 2048, 4096]
    # hop_lengths = [1024, 2048]
    # thresholds = [30, 35]
    # lecture.test_splits(frame_lengths, hop_lengths, thresholds)

    # outputs time intervals for start and end of each lecturing chunk
    intervals = lecture.final_split(threshold=30, hop_length=2048, frame_length=1024)

    # # find the percent silence removed and percent of lecture spent talking
    # # save label .txt file(s)
    # # ignore pauses of pause_length number of SECONDS
    # pause_length = 1
    # percent_trimmed, percent_talking = lecture.analyze_audio(intervals, pause_length)

    # # save a wav file of the audio with leading and trailing silences trimmed
    # lecture.save_trimmed_file()

    words = lecture.analyze_words(intervals)

# THINGS TO BE AWARE OF
# bis 2c hands mics to students who ask questions
# there is some student background noise
# I think we would need to tweak the system too much every time to recognize students from lecturer

# use google speech recognition for each chunk of lecturing
# reading writing level analysis
# determine if questions were asked
# tell if a different person is speaking
# find words per minute
