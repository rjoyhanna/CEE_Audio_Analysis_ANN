import librosa
import numpy as np
import json
import math
from textstat import lexicon_count, syllable_count, dale_chall_readability_score
__all__ = [lexicon_count, syllable_count, dale_chall_readability_score]

from download_convert_files import DownloadConvertFiles, remove_file_type, VALID_TXT, VALID_AUDIO, return_file_type

# Ctrl + F (TEST ME) to find functions that still need to be tested
# Ctrl + F (TO DO) to proceed to next steps


class LectureAudio:
    """
    A lecture audio object that holds information about an imported wav file

    Attributes:
        original_audio_filename (string): the name of the file being analyzed, WITH the original extension
        wav_filename (string): the name of the file being analyzed, WITH the .wav extension
        transcript_filename (string): the name of the transcript file, WITH the .txt extension
        file_handler (object): the object that downloads, converts, and deletes audio files from AWS s3
        audio_arr (np.ndarray): array of floats representing audio signal
        sr (int): the sampling rate used to store the audio signal
        trimmed_audio (np.ndarray): audio_arr with silence trimmed off beginning and end
        trimmed_filename (string): the name of the trimmed file, WITH the .wav extension
        trimmed_offset (int): the point of audio_arr where trimmed_audio begins, in SAMPLES
    """

    def __init__(self, filename, transcript, duration=None, download_needed=True):
        """
        Initializes an instance of the LectureAudio class.

        Args:
            filename (string): WITH .mp4 file extension, the name of the file you want to analyze
            transcript (string): WITH .txt file extension, the name of the transcript file
            duration (int): length in seconds of audio to be loaded in
        """

        if return_file_type(filename) in VALID_AUDIO:
            self.original_audio_filename = filename
            self.wav_filename = remove_file_type(filename) + '.wav'
        else:
            raise ValueError('filename must be one of: {}'.format(VALID_AUDIO))

        if return_file_type(transcript) == VALID_TXT:
            self.transcript_filename = transcript
        else:
            raise ValueError('transcript must be of type {}'.format(VALID_TXT))

        if download_needed:
            self.file_handler = DownloadConvertFiles(self.original_audio_filename, self.transcript_filename)

        # if no duration was given, just load the entire wav file
        if duration is None:
            audio_arr, sr = librosa.load('s3/{}'.format(self.wav_filename))

        # otherwise only load {duration} seconds of the wav file
        else:
            audio_arr, sr = librosa.load('s3/{}'.format(self.wav_filename),  duration=duration)

        # entire audio
        self.audio_arr = audio_arr

        # default sr=22050 for librosa.load
        self.sr = sr

        # audio with leading and trailing silences trimmed
        self.trimmed_audio, index = self.trim_ends()
        self.trimmed_offset = index[0]

        self.trimmed_filename = '{}_trimmed.wav'.format(remove_file_type(self.original_audio_filename))
        librosa.output.write_wav(filename, self.trimmed_audio, self.sr)
        print('{} was successfully saved!'.format(filename))

        # save the new trimmed wav file to the s3 bucket
        if download_needed:
            self.file_handler.upload_file(self.trimmed_filename)

        self.silence_threshold = self.get_silence_threshold()

    def trim_ends(self, verbose=False):
        """
        Trims the silence off of the beginning and end of the audio

        Returns:
            np.ndarray: audio with leading and trailing silence removed
            list: the interval of audio_arr corresponding to the non-silent region, in SAMPLES
        """

        # Trim the beginning and ending silence
        trimmed_audio, index = librosa.effects.trim(self.audio_arr, top_db=40)

        # Print the percent of audio left after trimming
        if verbose:
            prev_dur = librosa.get_duration(self.audio_arr)
            new_dur = librosa.get_duration(trimmed_audio)
            percent_left = (new_dur / prev_dur) * 100
            print('\nThe audio has been trimmed down to {:0.2f}% the original size.\n'.format(percent_left))

        return trimmed_audio, index

    def get_silence_threshold(self):
        mean = np.mean(self.trimmed_audio)
        max = np.amax(self.trimmed_audio)
        min = np.amin(self.trimmed_audio)
        std = np.std(self.trimmed_audio)

        frame_length = 2048
        hop_length = 1024
        pause_length = 2

        i = 0

        for threshold in [20, 25, 30, 35, 40]:
            intervals = self.split_on_silence(threshold, hop_length, frame_length, pause_length)
            self.create_labels(intervals, str(threshold), i)
            i += 1

        # print('{}, {}, {}, {}'.format(mean, max, min, std))

        return threshold

    def split_on_silence(self, threshold, hop_length, frame_length, pause):
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

        # ignore pauses less than the max pause length
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
            new_intervals.append(np.array([last_interval_end, audio_end_samples]))
            # list must hold times in SAMPLES not SECONDS

        return np.array(new_intervals)

    # IMPORTANT: STUDENT INTERVALS MUST BE GIVEN, NOT ALL INTERVALS
    def count_time_spent(self, professor_intervals, student_intervals=None):
        """
        Finds the percent of audio remaining after trailing and leading silences were removed
        Finds percent of audio that is lecture vs. silence

        Args:
            professor_intervals (np.ndarray): time intervals for start and end of each lecturing chunk, given in SAMPLES
            all_talking_intervals (np.ndarray): time intervals for start and end of each talking chunk, given in SAMPLES

        Returns:
            float: the percent of time trimmed away from the beginning and end of the audio
            float: the SECONDS spent in lecture
            float: the length of the trimmed audio in SECONDS
            np.ndarray: time intervals for start and end of each lecturing chunk, given in SAMPLES, with pauses ignored
        """

        # find the percent of the original file that was trimmed off as leading or trailing silence
        prev_dur = librosa.get_duration(self.audio_arr)
        new_dur = librosa.get_duration(self.trimmed_audio)
        percent_trimmed = 100 - ((new_dur / prev_dur) * 100)

        # add up all the time spent in lecture
        professor_talking = 0  # total time in SECONDS
        for label in professor_intervals:
            professor_talking += (label[1] - label[0]) / self.sr

        student_talking = 0  # total time in SECONDS
        if student_intervals is not None:
            for label in student_intervals:
                student_talking += (label[1] - label[0]) / self.sr

        return percent_trimmed, professor_talking, student_talking, new_dur

    def analyze_words(self):
        """
        Finds the number of words and array of words spoken in audio

        Returns:
            int: number of words spoken in lecture
            int: average number of syllables per word
            string: the grade level using the New Dale-Chall Formula
        """

        with open('s3/{}'.format(self.transcript_filename), 'r') as myfile:
            data = myfile.read()

        num_words = lexicon_count(data, removepunct=True)
        num_syllables = syllable_count(data, lang='en_US')

        grade_level = dale_chall_readability_score(data)

        if grade_level < 5:
            grade_level_string = 'average 4th-grade student or lower'
        elif grade_level < 6:
            grade_level_string = 'average 5th or 6th-grade student'
        elif grade_level < 7:
            grade_level_string = 'average 7th or 8th-grade student'
        elif grade_level < 8:
            grade_level_string = 'average 9th or 10th-grade student'
        elif grade_level < 9:
            grade_level_string = 'average 11th or 12th-grade student'
        elif grade_level < 10:
            grade_level_string = 'average 13th to 15th-grade (college) student'
        else:
            grade_level_string = 'above average college student'

        return num_words, num_syllables / num_words, grade_level_string

    def integrate_interval_sets(self, professor_intervals, all_intervals=None):
        if all_intervals is None:
            all_intervals = np.array([])

        student_intervals = self.get_student_intervals(professor_intervals, all_intervals)
        # all_intervals has 3 values for each chunk: start, end, label
        # labels:
        # 0 is silence
        # 1 is professor
        # 2 is student
        all_intervals = self.combine_interval_sets(professor_intervals, student_intervals)

        return professor_intervals, student_intervals

    @staticmethod
    def get_student_intervals(lecture_intervals, all_intervals):
        """
        determines the intervals of student participation

        Args:
            lecture_intervals (np.ndarray): the intervals for lecture
            all_intervals (np.ndarray): the intervals for all talking in lecture

        Returns:
            np.ndarray: the intervals for student participation in lecture
        """

        label_list = []

        for chunk in all_intervals:  # 35 dbs
            lectures_found = 0
            for i in range(0, len(lecture_intervals)):  # 20 dbs

                lecture = lecture_intervals[i]

                if chunk[1] > lecture[1] >= chunk[0]:

                    if lectures_found == 0 and lecture[0] > chunk[0]:
                        label_list.append(np.array([chunk[0], lecture[0]]))
                        lectures_found += 1

                    lectures_found += 1
                    start = lecture[1]
                    lecture_end = lecture_intervals[i + 1][0]
                    chunk_end = chunk[1]
                    if chunk_end < lecture_end:
                        label_list.append(np.array([start, chunk_end]))
                    else:
                        label_list.append(np.array([start, lecture_end]))
                elif lecture[1] == lecture_intervals[len(lecture_intervals) - 1][1] and chunk[1] == lecture[1]:
                    start = lecture_intervals[len(lecture_intervals) - 2][1]
                    end = lecture[0]
                    if chunk[0] > end:
                        label_list.append(np.array([start, chunk[0]]))
                    else:
                        label_list.append(np.array([start, end]))
                    lectures_found += 1

            if lectures_found == 0:
                label_list.append(chunk)

        return np.array(label_list)

    @staticmethod
    def combine_interval_sets(lecture_intervals, student_intervals):
        label_list = []

        i = 0
        j = 0
        lecture_intervals_finished = False
        student_intervals_finished = False

        if student_intervals.size == 0:
            student_intervals_finished = True

        if lecture_intervals.size == 0:
            lecture_intervals_finished = True

        while not lecture_intervals_finished or not student_intervals_finished:
            if lecture_intervals_finished:
                lecture_chunk = [math.inf, math.inf]
            else:
                lecture_chunk = np.append(lecture_intervals[i], 1)

            if student_intervals_finished:
                student_chunk = [math.inf, math.inf]
            else:
                student_chunk = np.append(student_intervals[j], 2)

            if lecture_chunk[0] < student_chunk[0]:
                lecture_chunk_comes_first = True
            else:
                lecture_chunk_comes_first = False

            if lecture_chunk_comes_first:
                label_list.append(lecture_chunk)
                i += 1
            else:
                label_list.append(student_chunk)
                j += 1

            if (i == len(lecture_intervals)):
                i = -1
                lecture_intervals_finished = True
            elif (j == len(student_intervals)):
                j = -1
                student_intervals_finished = True

        return np.array(label_list)

    # FINISH MEEEEEEEEEEE
    @staticmethod
    def fill_in_label_gaps(intervals):
        for i in range(0, len(intervals)):
            chunk = intervals[i]


    # TEST
    def full_analysis(self, threshold_all, threshold_lecture, hop_length, frame_length, pause_length):
        """
        Analyzes the audio and returns helpful data

        Args:
            threshold_all (int): the threshold (in decibels) below reference to consider as silence
            threshold_lecture (int): the threshold (in decibels) below reference to consider as silence or class
            hop_length (int): the number of samples between successive frames, e.g., the columns of a spectrogram
            frame_length (int): the (positive integer) number of samples in an analysis window (or frame)
            pause_length (int): the max length of silence to be ignored (grouped with lecture), given in SECONDS

        Returns:
            float: the percent of time trimmed away from the beginning and end of the audio
            float: the percent of time spent in lecture as compared to the entire trimmed audio
            float: the amount of time spent with students or professor talking, given in SECONDS
            float: the duration of the total audio, with leading and trailing silences trimmed, given in SECONDS
            int: the total number of words spoken in lecture
            int: the average syllables per word spoken in lecture
            string: description of the grade level of the transcription
            float: the amount of words spoken per minute of lecture
            float: the amount of words spoken per second of lecture
            np.ndarray: the time intervals where students are talking
            np.ndarray: the time intervals where the professor is talking
        """

        # outputs time intervals for start and end of each speech chunk (includes students)
        intervals_all = self.split_on_silence(threshold=threshold_all, hop_length=hop_length, frame_length=frame_length, pause=pause_length)

        # outputs time intervals for start and end of each lecturing chunk
        lecture_intervals = self.split_on_silence(threshold=threshold_lecture, hop_length=hop_length, frame_length=frame_length, pause=pause_length)

        lecture_intervals, student_intervals = self.integrate_interval_sets(lecture_intervals, intervals_all)

        # find the percent silence removed and percent of lecture spent talking
        percent_trimmed, professor_talking, student_talking, new_dur = lecture.count_time_spent(lecture_intervals, student_intervals)

        # FIX ME! Here we need to create one big label set for both student and professor intervals

        num_words, num_syllables, grade_level = lecture.analyze_words()

        words_per_second = num_words / professor_talking
        words_per_minute = words_per_second * 60

        response = {"percent_leading_trailing_silence_trimmed": percent_trimmed,
                    "student_talking_time": student_talking,
                    "professor_talking_time": professor_talking,
                    "class_duration": new_dur,
                    "words_spoken": num_words,
                    "average_syllables_per_word": num_syllables,
                    "grade_level": grade_level,
                    "words_per_minute": words_per_minute,
                    "words_per_second": words_per_second,
                    "all_labels": [
                        # FIX MEEEEE! do this dynamically
                        {"start": 0.09287981859410431, "end": 0.3250793650793651, "label": "student"},
                        {"start": 0.3250793650793651, "end": 4.4117913832199545, "label": "professor"},
                        {"start": 200.38820861678005, "end": 214.1907029478458, "label": "professor"}
                    ]
                    }

        # convert into JSON:
        response = json.dumps(response)

        return percent_trimmed, student_talking, professor_talking, new_dur, num_words, num_syllables, grade_level, words_per_minute, words_per_second, student_intervals, lecture_intervals

    # eventually we want to do this as the last step, once we have both student and professor intervals
    def create_labels(self, intervals, threshold, i=None):
        """
        Creates a txt file that can be imported as labels to Audacity
        File will have the columns: start, end, label

        Args:
            intervals (np.ndarray): time intervals for start and end of each lecturing chunk, given in SAMPLES
            threshold (string): the threshold of the intervals, also used to add strings to end of the filename
            i (int): hold the integer val to use in the filename (for creating multiple label files for one audio file

        Returns:
            np.ndarray: time intervals for start and end of each lecturing chunk, given in SAMPLES, with pauses ignored
        """

        # if no number was given, just name the file normally
        if i is None:
            filename = '{}_labels_{}.txt'.format(remove_file_type(self.original_audio_filename), threshold)

        # otherwise, add the number to the end of the file name
        # this is used when multiple tests are being run on the same audio file
        else:
            filename = '{}_labels_{}_{}.txt'.format(remove_file_type(self.original_audio_filename), threshold, i)

        # open the file to make sure we create it if it isn't there
        f = open(filename, "w+")
        f.close()

        new_intervals = self.glob_labels(intervals)

        # add a label for each interval
        for row in new_intervals:
            self.add_label_row(filename, row[0] / self.sr, row[1] / self.sr, 1)

        return new_intervals

    # make max_length variable available in parameters?
    # can we make this a part of create_labels? Or even remove it and complete this BEFORE making labels
    def glob_labels(self, intervals):
        """
        Combines intervals that are directly adjacent (makes labels more readable in Audacity

        Args:
            intervals (np.ndarray): time intervals for start and end of each lecturing chunk, given in SAMPLES

        Returns:
            np.ndarray: simplified intervals
        """

        label_list = []
        if len(intervals) > 0:
            label_start = intervals[0][0]
        # label_end = intervals[0][1]

        max_length = 20000 * self.sr  # this keeps the intervals at a good length for speech recognition

        for i in range(0, len(intervals) - 1):
            current_interval = intervals[i]
            next_interval = intervals[i + 1]
            if next_interval[0] == current_interval[1]:
                future_length = next_interval[1] - label_start
                if future_length >= max_length:
                    label_end = current_interval[1]  # set new interval end to the end of the current interval
                    label_list.append(np.array([label_start, label_end]))
                    label_start = next_interval[0]
                else:
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


if __name__ == '__main__':

    # select the base filename to be analyzed
    transcript_file = 'BIS-2A_ 2019-07-17 12_10_transcript.txt'
    audio_file = 'BIS-2A__2019-07-17_12_10.mp4'

    # create an instance of the LectureAudio class
    # extract audio info from wav file and trim leading and trailing silence
    lecture = LectureAudio(audio_file, transcript_file, download_needed=False)

    # lecture.get_silence_threshold()

    # # create an instance of the LectureAudio class
    # # only load first 1200 seconds to make tests run faster
    # lecture = LectureAudio(audio_file, duration=1200)

    threshold_all = 35
    threshold_lecture = 30
    hop_length = 1024
    frame_length = 2048
    pause_length = 2

    lecture.full_analysis(threshold_all, threshold_lecture, hop_length, frame_length, pause_length)


# THINGS TO BE AWARE OF
# 20 db is a good setting to get all speaking sound (includes background noise and student discussion
# 35 db is a good setting for just lecturing
# try to combine this data to get student participation time
