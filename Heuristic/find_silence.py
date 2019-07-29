import librosa
import sys
import numpy as np
import speech_recognition as sr
import datetime
from textstat import lexicon_count, syllable_count, dale_chall_readability_score
__all__ = [lexicon_count, syllable_count, dale_chall_readability_score]

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
        transcript_filename (string): the name of the transcript file, WITH the .txt extension
        audio_arr (np.ndarray): array of floats representing audio signal
        sr (int): the sampling rate used to store the audio signal
        trimmed_audio (np.ndarray): audio_arr with silence trimmed off beginning and end
        trimmed_filename (string): the name of the trimmed file, WITH the .wav extension
        trimmed_offset (int): the point of audio_arr where trimmed_audio begins, in SAMPLES
    """

    def __init__(self, filename, transcript, duration=None):
        """
        Initializes an instance of the LectureAudio class.

        Args:
            filename (string): without file extension, the name of the file you want to analyze
            transcript (string): without file extension, the name of the transcript file
            duration (int): length in seconds of audio to be loaded in
        """

        self.base_filename = filename
        self.wav_filename = add_wav(filename)
        self.transcript_filename = transcript + '.txt'

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
        self.trimmed_audio, index = self.trim_ends()
        self.trimmed_offset = index[0]

        self.trimmed_filename = self.save_trimmed_file()

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
                    # print('{}: {} intervals found using frame_length={}, hop_length={}, threshold={}.'
                    #       .format(i, len(intervals), frame_length, hop_length, threshold))

                    # for each chunk, trim the leading and trailing silence using the given threshold
                    intervals = self.trim_chunks(intervals, threshold)

                    # save a txt file with the labels for each set of args
                    self.create_labels(intervals, threshold, i)

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
        # print('{} intervals found using frame_length={}, hop_length={}, threshold={}.'
        #       .format(len(intervals), frame_length, hop_length, threshold))

        # for each chunk, trim the leading and trailing silence using the given threshold
        new_intervals = self.trim_chunks(intervals, threshold)

        return new_intervals

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
            filename = '{}_labels_{}.txt'.format(self.base_filename, threshold)

        # otherwise, add the number to the end of the file name
        # this is used when multiple tests are being run on the same audio file
        else:
            filename = '{}_labels_{}_{}.txt'.format(self.base_filename, threshold, i)

        # open the file to make sure we create it if it isn't there
        f = open(filename, "w+")
        f.close()

        new_intervals = self.glob_labels(intervals)

        # add a label for each interval
        for row in new_intervals:
            self.add_label_row(filename, row[0] / self.sr, row[1] / self.sr, 1)

        return new_intervals

    # make max_length variable available in parameters?
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
            new_intervals.append(np.array([last_interval_end, audio_end_samples]))
            # list must hold times in SAMPLES not SECONDS

        return np.array(new_intervals)

    def save_trimmed_file(self):
        """
        Saves the audio with trimmed leading and trailing silence as a wav file

        Returns:
            string: the name of the trimmed file
        """

        filename = '{}_trimmed.wav'.format(self.base_filename)
        librosa.output.write_wav(filename, self.trimmed_audio, self.sr)
        print('{} was successfully saved!'.format(filename))

        return filename

    def analyze_audio(self, intervals, pause_length, threshold):
        """
        Finds the percent of audio remaining after trailing and leading silences were removed
        Finds percent of audio that is lecture vs. silence

        Args:
            intervals (np.ndarray): time intervals for start and end of each lecturing chunk, given in SAMPLES
            pause_length (int): the max length of silence to be ignored (grouped with lecture), given in SECONDS
            threshold (int): the threshold (in decibels) below reference to consider as silence

        Returns:
            float: the percent of time trimmed away from the beginning and end of the audio
            float: the SECONDS spent in lecture
            float: the length of the trimmed audio in SECONDS
            np.ndarray: time intervals for start and end of each lecturing chunk, given in SAMPLES, with pauses ignored
        """

        # Print the durations
        prev_dur = librosa.get_duration(self.audio_arr)
        new_dur = librosa.get_duration(self.trimmed_audio)
        percent_trimmed = 100 - ((new_dur / prev_dur) * 100)

        talking = 0  # total time in SECONDS

        # count short pauses as lecture
        adjusted_intervals = self.ignore_silence(intervals, pause_length)

        # save a txt file with the labels
        final_intervals = self.create_labels(adjusted_intervals, threshold)

        # add up all the time spent in lecture
        for label in adjusted_intervals:
            talking += (label[1] - label[0]) / self.sr

        return percent_trimmed, talking, new_dur, final_intervals

    # TO DO
    def analyze_words_speech_recognition(self, intervals):
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

        # open the file to make sure we create it if it isn't there
        words_file = '{}_words.txt'.format(self.base_filename)

        words = ''
        total_text = []

        for interval in intervals:
            with lecture as source:  # should this be outside or inside of the intervals for loop????
                start = interval[0] / self.sr
                end = interval[1] / self.sr

                length = end - start
                if length < 60:
                    audio = r.record(source, offset=start+(self.trimmed_offset/self.sr), duration=length)
                else:
                    print('Interval starting at {} is too long to recognize.\n'.format(start))
                    continue

            try:
                words = r.recognize_google(audio)

                total_text.append(words)

                print('start: {}\tend: {}\n'.format(start, end))
                print(words)

                f = open(words_file, "a+")
                # f.write('start: {}\tend: {}\n'.format(start, end))
                f.write(words)
                f.write('\n')
                f.close()

            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")

                total_text.append('')

                f = open(words_file, "a+")
                # f.write('start: {}\tend: {}\n'.format(start, end))
                # f.write('Google Speech Recognition could not understand audio')
                f.write('\n\n')
                f.close()
            except sr.RequestError as e:
                print("Could not request results from Google Speech Recognition service; {0}".format(e))

                total_text.append('')

        word_string = ''
        for phrase in total_text:
            word_string = word_string + ' ' + phrase

        num_words = lexicon_count(word_string, removepunct=True)
        num_syllables = syllable_count(word_string, lang='en_US')

        # 4.9 or lower	average 4th-grade student or lower
        # 5.0–5.9	average 5th or 6th-grade student
        # 6.0–6.9	average 7th or 8th-grade student
        # 7.0–7.9	average 9th or 10th-grade student
        # 8.0–8.9	average 11th or 12th-grade student
        # 9.0–9.9	average 13th to 15th-grade (college) student
        grade_level = dale_chall_readability_score(word_string)

        return num_words, num_syllables, grade_level, total_text

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

    def analyze_words(self):
        """
        Finds the number of words and array of words spoken in audio

        Returns:
            int: number of words spoken in lecture
            int: average number of syllables per word
            string: the grade level using the New Dale-Chall Formula
        """

        with open(self.transcript_filename, 'r') as myfile:
            data = myfile.read()

        num_words = lexicon_count(data, removepunct=True)
        num_syllables = syllable_count(data, lang='en_US')

        # 4.9 or lower	average 4th-grade student or lower
        # 5.0–5.9	average 5th or 6th-grade student
        # 6.0–6.9	average 7th or 8th-grade student
        # 7.0–7.9	average 9th or 10th-grade student
        # 8.0–8.9	average 11th or 12th-grade student
        # 9.0–9.9	average 13th to 15th-grade (college) student
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

    # TEST
    def full_analysis(self, pause_length, threshold_all, threshold_lecture, hop_length, frame_length):
        """
        Analyzes the audio and returns helpful data

        Args:
            pause_length (int): the max length of silence to be ignored (grouped with lecture), given in SECONDS
            threshold_all (int): the threshold (in decibels) below reference to consider as silence
            threshold_lecture (int): the threshold (in decibels) below reference to consider as silence or class
            hop_length (int): the number of samples between successive frames, e.g., the columns of a spectrogram
            frame_length (int): the (positive integer) number of samples in an analysis window (or frame)

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
        intervals_all = self.final_split(threshold=threshold_all, hop_length=hop_length, frame_length=frame_length)

        # find the percent silence removed and percent of lecture spent talking
        # save label .txt file(s)
        # ignore pauses of pause_length number of SECONDS
        _, total_speech, _, all_final_intervals = lecture.analyze_audio(intervals_all, pause_length, threshold_all)

        # each lecture chunk is more or less included in the set of speech chunks

        # outputs time intervals for start and end of each lecturing chunk
        intervals = self.final_split(threshold=threshold_lecture, hop_length=hop_length, frame_length=frame_length)

        # find the percent silence removed and percent of lecture spent talking
        # save label .txt file(s)
        # ignore pauses of pause_length number of SECONDS
        percent_trimmed, talking, new_dur, final_intervals = lecture.analyze_audio(intervals, pause_length,
                                                                                   threshold_lecture)

        # save a txt file with student participation labels
        student_intervals = self.get_student_intervals(final_intervals, all_final_intervals)

        self.create_labels(student_intervals, 'student')

        num_words, num_syllables, grade_level = lecture.analyze_words()

        print('\nRemoved {:0.2f}% of the audio as leading or trailing silence.\n'.format(percent_trimmed))

        # calculate the perent time spent in lecture
        percent_talking = (talking / new_dur) * 100

        if talking < total_speech:
            student_participation = total_speech - talking
        else:
            student_participation = 0

        percent_student = (student_participation / new_dur) * 100

        print('Of the remaining audio, '
              '{:0.2f}% was lecture, '
              '{:0.2f}% was student participation, '
              'and {:0.2f}% was silence.'.format(percent_talking, percent_student,
                                                 100 - percent_talking - percent_student))

        lecture_time = str(datetime.timedelta(hours=new_dur/60/60))
        talking_time = str(datetime.timedelta(hours=talking/60/60))
        student_time = str(datetime.timedelta(hours=student_participation/60/60))

        print('Of the {} of lecture, you spent {} talking, and the class spent {} talking.\n'.format(lecture_time,
                                                                                                     talking_time,
                                                                                                     student_time))

        print('Words: {}\nAverage syllables per word: {:0.2f}\nTranscript grade level: {}\n'.format(num_words,
                                                                                                    num_syllables,
                                                                                                    grade_level))
        # would it be more accurate to use talking time instead of total duration?
        words_per_second = num_words / talking
        words_per_minute = words_per_second * 60
        print("Words per minute: {:0.2f}".format(words_per_minute))
        print('That\'s about {:0.2f} words per second!'.format(words_per_second))

        if words_per_minute < 120:
            print("That is considered a slower-than-average speech rate.")
        elif words_per_minute < 200:
            print("That is considered a normal conversational speech rate.")
        else:
            print("That is considered a faster_than_average speech rate.")

        return percent_trimmed, percent_talking, talking, new_dur, num_words, num_syllables, grade_level, words_per_minute, words_per_second, student_intervals, final_intervals

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
                        lectures_found +=1

                    lectures_found += 1
                    start = lecture[1]
                    lecture_end = lecture_intervals[i+1][0]
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

    def set_db_levels(self, pause_length, threshold_all=35, threshold_lecture=20, hop_length=2048, frame_length=1024):
        # outputs time intervals for start and end of each speech chunk (includes students)
        intervals_all = self.final_split(threshold=threshold_all, hop_length=hop_length, frame_length=frame_length)

        # find the percent silence removed and percent of lecture spent talking
        # save label .txt file(s)
        # ignore pauses of pause_length number of SECONDS
        _, total_speech, _, _ = lecture.analyze_audio(intervals_all, pause_length, threshold_all)

        # each lecture chunk is more or less included in the set of speech chunks

        # outputs time intervals for start and end of each lecturing chunk
        intervals = self.final_split(threshold=threshold_lecture, hop_length=hop_length, frame_length=frame_length)

        # save a txt file with student participation labels
        student_intervals = self.get_student_intervals(intervals, intervals_all)

        self.create_labels(student_intervals, 'student')

        # find the percent silence removed and percent of lecture spent talking
        # save label .txt file(s)
        # ignore pauses of pause_length number of SECONDS
        percent_trimmed, talking, new_dur, final_intervals = lecture.analyze_audio(intervals, pause_length,
                                                                                   threshold_lecture)


if __name__ == '__main__':

    # select the base filename to be analyzed
    transcript_file = 'BIS-2A_ 2019-07-17 12_10_transcript'
    audio_file = 'BIS-2A__2019-07-17_12_10_tester'

    # create an instance of the LectureAudio class
    # extract audio info from wav file and trim leading and trailing silence
    lecture = LectureAudio(audio_file, transcript_file)

    # # create an instance of the LectureAudio class
    # # only load first 1200 seconds to make tests run faster
    # lecture = LectureAudio(audio_file, duration=1200)

    # # run and analyze this test if unsure what params will work best for final split
    # frame_lengths = [1024]
    # hop_lengths = [2048]
    # thresholds = [30, 10]
    # lecture.test_splits(frame_lengths, hop_lengths, thresholds)

    # threshold = 10
    # hop_length = 2048
    # frame_length = 1024
    # pause_length = 2
    #
    # lecture.full_analysis(pause_length, threshold, hop_length, frame_length)

    threshold_all = 35
    threshold_lecture = 20
    hop_length = 2048
    frame_length = 1024
    pause_length = 2

    lecture.full_analysis(pause_length, threshold_all, threshold_lecture, hop_length, frame_length)


# THINGS TO BE AWARE OF
# 20 db is a good setting to get all speaking sound (includes background noise and student discussion
# 35 db is a good setting for just lecturing
# try to combine this data to get student participation time
