import librosa
import sys

sys.path.append('../')
from ML.file_helper import add_wav


class LectureAudio:

    def __init__(self, filename, duration=None):
        """
        Initializes an instance of the LectureAudio class.

        Args:
            filename:string: without file extension, the name of the file you want to analyze
            duration:int: length in seconds of audio to be loaded in
        Attributes:
            wav_filename:string: the name of the file being analyzed, WITH the .wav extension
            audio_arr:array: array of floats representing audio signal
            sr:int: the sampling rate used to store the audio signal
            trimmed_audio:array: audio_arr with silence trimmed off beginning and end
        """

        self.base_filename = filename
        self.wav_filename = add_wav(filename)

        # default sr=22050
        if duration is None:
            audio_arr, sr = librosa.load(self.wav_filename)
        else:
            audio_arr, sr = librosa.load(self.wav_filename,  duration=duration)

        self.audio_arr = audio_arr
        self.sr = sr
        self.trimmed_audio = self.trim_ends()

    def trim_ends(self):
        """
        Trims the silence off of the beginning and end of the audio
        """

        # Trim the beginning and ending silence
        trimmed_audio, index = librosa.effects.trim(self.audio_arr, top_db=40)

        # Print the durations
        prev_dur = librosa.get_duration(self.audio_arr)
        print(trimmed_audio)
        new_dur = librosa.get_duration(trimmed_audio)
        percent_left = (new_dur / prev_dur) * 100
        print('\nThe audio has been trimmed down to {:0.2f}% the original size.\n'.format(percent_left))

        return trimmed_audio

    def split_on_silence(self, threshold, frame_length, hop_length):
        """
        Splits the audio into invervals of sound (removes the silence)

        Args:
            threshold:int: the threshold (in decibels) below reference to consider as silence
            frame_length:int: the (positive integer) number of samples in an analysis window (or frame)
            hop_length:int: the number of samples between successive frames, e.g., the columns of a spectrogram
        """

        intervals = librosa.effects.split(self.trimmed_audio, top_db=threshold, frame_length=frame_length,
                                          hop_length=hop_length)
        # print('{} intervals of sound have been found.'.format(len(intervals)))

        return intervals

    def test_splits(self):
        """
        Tests different parameters used to split the audio on the silence
        Calls split_on_silence() for each different set of parameters
        """

        i = 0
        for frame_length in [1024, 2048, 4096]:
            for hop_length in [1024, 2048]:
                for threshold in [30, 35]:
                    i += 1
                    intervals = self.split_on_silence(threshold, hop_length, frame_length)
                    print('{}: {} intervals found using frame_length={}, hop_length={}, threshold={}.'
                          .format(i, len(intervals), frame_length, hop_length, threshold))
                    for j in range(0, len(intervals)):
                        interval = intervals[j]
                        start = interval[0]
                        end = interval[1]
                        clip = self.trimmed_audio[int(start):int(end)]
                        trimmed_audio, index = librosa.effects.trim(clip, top_db=threshold)
                        intervals[j][0] += index[0]  # / self.sr
                        intervals[j][1] = intervals[j][0] + index[1]  # / self.sr
                    self.create_labels(intervals, i)

    def final_split(self, threshold, hop_length, frame_length):
        """
        Splits the audio on silence once you determine the best parameters

        Best for BIS-2A__2019-07-17_12_10.wav:
        *** 1173 intervals found using frame_length=1024, hop_length=2048, threshold=30.
            1125 intervals found using frame_length=2048, hop_length=1024, threshold=35.
            864 intervals found using frame_length=4096, hop_length=2048, threshold=30.

        Args:
            threshold:int: the threshold (in decibels) below reference to consider as silence
            frame_length:int: the (positive integer) number of samples in an analysis window (or frame)
            hop_length:int: the number of samples between successive frames, e.g., the columns of a spectrogram
        """
        intervals = self.split_on_silence(threshold, hop_length, frame_length)
        print('{} intervals found using frame_length={}, hop_length={}, threshold={}.'
              .format(len(intervals), frame_length, hop_length, threshold))
        for j in range(0, len(intervals)):
            interval = intervals[j]
            start = interval[0]
            end = interval[1]
            clip = self.trimmed_audio[int(start):int(end)]
            trimmed_audio, index = librosa.effects.trim(clip, top_db=threshold)
            intervals[j][0] += index[0]  # / self.sr
            intervals[j][1] = intervals[j][0] + index[1]  # / self.sr

        self.create_labels(intervals)
        return intervals

    def create_labels(self, intervals, i=None):
        """
        Creates a txt file that can be imported as labels to Audacity
        File will have the columns: start, end, label

        Args:
            intervals:array: holds tuples with start and end times of intervals of sound
            i:int: hold the integer val to use in the filename (for creating multiple label files for one audio file
        """

        if i is None:
            filename = '{}_labels.txt'.format(self.base_filename)
        else:
            filename = '{}_labels_{}.txt'.format(self.base_filename, i)

        f = open(filename, "w+")
        f.close()

        for row in intervals:
            self.add_label_row(filename, row[0] / self.sr, row[1] / self.sr, 1)

    @staticmethod
    def add_label_row(filename, start, end, label):
        """
        Adds a row to the current label txt file
        Will add a row with the columns: start, end, label

        Args:
            filename:string: holds the filename of the txt file to be edited
            start:int: the start value of the label in milliseconds
            end:int: the end value of the label in milliseconds
            label:int: represents whether the current clip is silence (0) or lecture (1)
        """

        if label != -1:
            f = open(filename, "a")
            # print("{}\t{}\t{}\n".format(start, end, label))
            f.write("{}\t{}\t{}\n".format(start, end, label))
            f.close()

    def save_trimmed_file(self):
        """
        Saves the audio with trimmed leading and trailing silence as a wav file
        """

        librosa.output.write_wav('{}_trimmed.wav'.format(self.base_filename), self.trimmed_audio, self.sr)
        print('file successfully saved!')

    def analyze_audio(self, intervals):
        """
        Finds the percent of audio remaining after trailing and leading silences were removed
        Finds percent of audio that is lecture vs. silence

        Args:
            intervals:array: start and end in milliseconds of each non-silent clip in audio
        """

        # Print the durations
        prev_dur = librosa.get_duration(self.audio_arr)
        new_dur = librosa.get_duration(self.trimmed_audio)
        percent_trimmed = 100 - ((new_dur / prev_dur) * 100)
        print('\nRemoved {:0.2f}% of the audio as leading or trailing silence.\n'.format(percent_trimmed))

        talking = 0
        for label in intervals:
            talking += (label[1] - label[0]) / self.sr
        percent_talking = (talking / new_dur) * 100
        print('Of the remaining audio, {:0.2f}% was lecture, and {:0.2f}% was silence.\n'.format(percent_talking,
                                                                                               100 - percent_talking))

        return percent_trimmed, percent_talking

    def analyze_words(self, intervals):
        """
        Finds the number of words and array of words spoken in audio

        Args:
            intervals:array: start and end in milliseconds of each non-silent clip in audio
        """

        # return words
        pass

    def analyze_questions(self, intervals, words):
        """
        Finds the number of questions asked based on an array of words and intervals of talking

        Args:
            intervals:array: start and end in milliseconds of each non-silent clip in audio
        """

        # return num_questions
        pass


if __name__ == '__main__':
    audio_file = 'BIS-2A__2019-07-17_12_10'
    lecture = LectureAudio(audio_file)
    # lecture.test_splits()
    intervals = lecture.final_split(threshold=30, hop_length=2048, frame_length=1024)
    lecture.analyze_audio(intervals)
    # lecture.save_trimmed_file()
