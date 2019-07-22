import librosa
import sys

sys.path.append('../')
from ML.file_helper import add_wav


class LectureAudio:

    def __init__(self, filename):
        """
        Initializes an instance of the LectureAudio class.

        Args:
            filename:string: without file extension, the name of the file you want to analyze
        Attributes:
            wav_filename:string: the name of the file being analyzed, WITH the .wav extension
            audio_arr:array: array of floats representing audio signal
            sr:int: the sampling rate used to store the audio signal
            trimmed_audio:array: audio_arr with silence trimmed off beginning and end
        """

        self.base_filename = filename
        self.wav_filename = add_wav(filename)
        # default sr=22050
        audio_arr, sr = librosa.load(self.wav_filename)
        self.audio_arr = audio_arr
        self.sr = sr
        self.trimmed_audio = self.trim_ends()

    def trim_ends(self):
        """
        Trims the silence off of the beginning and end of the audio
        """

        # Trim the beginning and ending silence
        trimmed_audio, index = librosa.effects.trim(self.audio_arr, top_db=50)

        # Print the durations
        prev_dur = librosa.get_duration(self.audio_arr)
        print(trimmed_audio)
        new_dur = librosa.get_duration(trimmed_audio)  # FIX MEEEEEEEEEEEEEEEE
        percent_left = (new_dur / prev_dur) * 100
        print('\nThe audio has been trimmed down to {:03f}% the original size.\n'.format(percent_left))

        return trimmed_audio

    def split_on_silence(self, threshold, frame_length, hop_length):
        """
        Splits the audio into invervals of sound (removes the silence)

        Args:
            threshold:int: the highest value to still be considered as silence
            frame_length:int:
            hop_length:int:
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
            for hop_length in [256, 512, 1024]:
                for threshold in [20, 60, 80]:
                    i += 1
                    intervals = self.split_on_silence(threshold, hop_length, frame_length)
                    print('{}: {} intervals found using frame_length={}, hop_length={}, threshold={}.'
                          .format(i, len(intervals), frame_length, hop_length, threshold))
                    print(intervals)
                    self.create_labels(intervals, i)

    def create_labels(self, intervals, i=None):
        """
        Creates a txt file that can be imported as labels to Audacity
        File will have the columns: start, end, label

        Args:
            intervals:array: holds tuples with start and end times of intervals of sound
            i:int: hold the integer val to use in the filename (for creating multiple label files for one audio file
        """

        filename = '{}_labels_{}.txt'.format(self.base_filename, i)
        f = open(filename, "w+")
        f.close()

        for row in intervals:
            self.add_label_row(filename, row[0] / 1000, row[1] / 1000, 1)

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


if __name__ == '__main__':
    audio_file = 'output_audio'
    lecture = LectureAudio(audio_file)
    lecture.test_splits()
    lecture.save_trimmed_file()
