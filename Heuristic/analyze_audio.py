import librosa
import numpy as np
import json
import math
import sys
import collections
import string
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
        librosa.output.write_wav(self.trimmed_filename, self.trimmed_audio, self.sr)
        print('{} was successfully saved!'.format(self.trimmed_filename))

        # save the new trimmed wav file to the s3 bucket
        if download_needed:
            self.file_handler.upload_file(self.trimmed_filename)

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

        frame_length = 2048
        hop_length = 1024

        # outputs time intervals for start and end of each speech chunk (includes students)
        intervals_all = self.split_on_silence(threshold=35, hop_length=hop_length, frame_length=frame_length)

        overall_mean = 0  # this will ignore all silence gaps
        num_samples_seen = 0
        overall_max = 0
        overall_min = math.inf
        # overall_std = 0
        for chunk in intervals_all:
            # should intervals be converted to seconds instead of samples?
            length = chunk[1] - chunk[0]
            audio_chunk = self.trimmed_audio[int(chunk[0]):int(chunk[1])]
            mean = np.mean(audio_chunk)  # FIX MEEEE must find the mean of the data, not the interval start/end
            max = np.amax(audio_chunk)
            min = np.amin(audio_chunk)
            # std = np.std(audio_chunk)

            if max > overall_max:
                overall_max = max

            if min < overall_min:
                overall_min = min

            if num_samples_seen == 0:
                overall_mean = mean
                num_samples_seen = length
                # print(overall_mean)
                # print(num_samples_seen)
            else:
                overall_mean = ((overall_mean * num_samples_seen) + (mean * length)) / (num_samples_seen + length)
                # print(overall_mean)
                num_samples_seen += length
                # print(num_samples_seen)

        print("\nOVERALL:")
        print('mean: {}\tmax: {}\tmin: {}'.format(overall_mean, overall_max, overall_min))

        _, _, _, _, _, _, _, _, _, _, intervals = self.full_analysis(30, 30, hop_length, frame_length, 1, 2)
        self.create_labels(intervals, 2)

        _, _, _, _, _, _, _, _, _, _, intervals = self.full_analysis(31, 30, hop_length, frame_length, 2, 1)
        self.create_labels(intervals, 3)

        _, _, _, _, _, _, _, _, _, _, intervals = self.full_analysis(32, 30, hop_length, frame_length, 2, 1)
        self.create_labels(intervals, 4)

        _, _, _, _, _, _, _, _, _, _, intervals = self.full_analysis(33, 30, hop_length, frame_length, 2, 1)
        self.create_labels(intervals, 5)

        _, _, _, _, _, _, _, _, _, _, intervals = self.full_analysis(34, 30, hop_length, frame_length, 2, 1)
        self.create_labels(intervals, 6)

    def split_on_silence(self, threshold, hop_length, frame_length):
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

        # # ignore pauses less than the max pause length
        # i = 0
        #
        # # empty list so we don't have to append an array over and over again (saves memory)
        # new_intervals = []
        #
        # # add the leading silence to the lecture time if it is less than the pause length
        # if intervals[0][0] <= pause and intervals[0][0] != 0:
        #     new_intervals.append(np.array([0, intervals[0][0]]))
        #
        # for i in range(0, len(intervals) - 1):
        #     speech_interval = intervals[i]
        #     pause_beginning = speech_interval[1]
        #     pause_end = intervals[i + 1][0]
        #
        #     # add each old interval to the list
        #     new_intervals.append(speech_interval)
        #
        #     # if the pause between an interval is less than the maximum, assume its still lecture
        #     if (pause_end - pause_beginning) / self.sr <= pause:
        #         new_intervals.append(np.array([pause_beginning, pause_end]))
        #
        # # add the last interval to the new list
        # last_interval = intervals[len(intervals) - 1]
        # new_intervals.append(last_interval)  # list must hold times in SAMPLES not SECONDS
        #
        # # add the trailing silence to the lecture time if its less than the pause length
        # last_interval_end = last_interval[1]  # in samples
        # audio_end = librosa.get_duration(self.trimmed_audio)  # in seconds
        # audio_end_samples = audio_end * self.sr  # in samples
        # if audio_end - (last_interval_end / self.sr) <= pause:
        #     new_intervals.append(np.array([last_interval_end, audio_end_samples]))
        #     # list must hold times in SAMPLES not SECONDS

        return np.array(intervals)

    # IMPORTANT: STUDENT INTERVALS MUST BE GIVEN, NOT ALL INTERVALS
    def count_time_spent(self, intervals):
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
        student_talking = 0
        silence = 0
        for label in intervals:
            if label[2] == 1:
                professor_talking += (label[1] - label[0]) / self.sr
            elif label[2] == 0:
                silence += (label[1] - label[0]) / self.sr
            elif label[2] == 2:
                student_talking += (label[1] - label[0]) / self.sr

        return percent_trimmed, professor_talking, student_talking, silence, new_dur

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
        total_num_syllables = syllable_count(data, lang='en_US')

        difficult_words = []

        lower = data.lower()
        no_punctuation = ''.join(char for char in lower if char not in string.punctuation)
        words = no_punctuation.split()
        syllable_dist = [0] * 10
        for word in words:
            num_syllables = syllable_count(word, lang='en_US')
            syllable_dist[num_syllables] = syllable_dist[num_syllables] + 1
            if num_syllables > 4:
                difficult_words.append(word)
        print(syllable_dist)

        punctuation = string.punctuation.replace("'", "")
        lower = lower.replace("all right", "alright")
        lower = lower.replace("you see", "you_see")
        lower = lower.replace("you know", "you_know")
        lower = lower.replace("i mean", "i_mean")
        lower = lower.replace("or something", "or_something")
        lower = lower.replace("thank you", "thank_you")
        no_punctuation = ''.join(char for char in lower if char not in punctuation)
        no_punctuation = no_punctuation.replace("and and", "and_and")
        words = no_punctuation.split()

        # get a distribution of the most common words used, ignoring filler words
        ignore = {'the', 'a', 'if', 'in', 'it', 'of', 'or', 'and', 'to', 'that', 'this', 'so', 'is', 'some', 'on', 'these', 'those', 'one', 'can', 'are', 'they', 'like', '2', 'two', 'for', 'have', 'from', 'with', 'it\'s', 'that\'s'}
        Counter = collections.Counter(word for word in words if word not in ignore)
        most_common = Counter.most_common(20)
        # print(most_common)

        # get distribution of most used filler words
        fillers = {'um', 'uh', 'ah', 'like', 'okay', 'ok', 'alright', 'and_and', 'really', 'well', 'you_see', 'you_know', 'i_mean', 'so', 'or_something', 'right'}
        Counter_fillers = collections.Counter(word for word in words if word in fillers)
        common_fillers = Counter_fillers.most_common(20)
        # print(common_fillers)

        # get distribution of most used positive words
        positives = {'good', 'happy', 'thank_you'}
        Counter_positives = collections.Counter(word for word in words if word in positives)
        common_positives = Counter_positives.most_common(20)
        # print(common_positives)

        # get a distribution of the letters per word
        word_length_dist = [0] * 30
        other_difficult_words = []

        for word in words:
            length = len(word)
            word_length_dist[length] = word_length_dist[length] + 1
            if length > 12:
                other_difficult_words.append(word)

        final_difficult_words = other_difficult_words.copy()
        for word in other_difficult_words:
            if word not in difficult_words:
                final_difficult_words.append(word)

        Counter_difficult = collections.Counter(final_difficult_words)
        most_difficult = Counter_difficult.most_common(20)
        print(most_difficult)

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

        return num_words, total_num_syllables / num_words, grade_level_string, most_common, common_fillers, common_positives, word_length_dist, syllable_dist, most_difficult

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

        return all_intervals

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
        used = 0

        # find the next professor end (skip adjacent labels)
        # check if the all ConnectionRefusedError

        for i in range(0, len(lecture_intervals)):
            lecture_chunk = lecture_intervals[i]

            if i < len(lecture_intervals) - 1:
                next_lecture_chunk = lecture_intervals[i + 1]
            else:
                next_lecture_chunk = np.array([math.inf, math.inf])

            if lecture_chunk[1] == next_lecture_chunk[0]:
                continue

            student_start = lecture_chunk[1]
            student_end = next_lecture_chunk[0]

            # print('\nstudent: [{} {}]'.format(student_start, student_end))

            for j in range(used, len(all_intervals)):
                all_chunk = all_intervals[j]
                # print('all: {}'.format(all_chunk))

                if (student_end >= all_chunk[1] > student_start):
                    label_list.append(np.array([student_start, all_chunk[1]]))

                if student_start < all_chunk[0]:
                    student_start = all_chunk[0]

                elif (student_end <= all_chunk[1]) and (student_start >= all_chunk[0]):
                    if (student_end != math.inf):
                        label_list.append(np.array([student_start, student_end]))
                    else:
                        label_list.append(np.array([student_start, all_chunk[1]]))

                elif student_end < all_chunk[1]:
                    used = j
                    break

        return np.array(label_list)

    # WORKS
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

            if i == len(lecture_intervals):
                i = -1
                lecture_intervals_finished = True
            elif j == len(student_intervals):
                j = -1
                student_intervals_finished = True

        return np.array(label_list)

    @staticmethod
    def fill_in_label_gaps(intervals, pause_length):
        label_list = []

        for i in range(0, len(intervals) - 1):
            chunk = intervals[i]
            next_chunk = intervals[i+1]
            if (next_chunk[0] - chunk[1] < pause_length) and (next_chunk[0] - chunk[1] != 0):
                if chunk[2] == next_chunk[2]:
                    new_chunk = [chunk[1], next_chunk[0], chunk[2]]
                    label_list.extend([chunk, new_chunk])
                else:
                    new_chunk = [chunk[1], next_chunk[0], 1]
                    label_list.extend([chunk, new_chunk])
            else:
                label_list.append(chunk)

        label_list.append(intervals[len(intervals) - 1])

        return np.array(label_list)
    
    @staticmethod
    def ignore_short_student_intervals(intervals, pause_length):
        for i in range(0, len(intervals)):
            chunk = intervals[i]
            if (chunk[2] == 2) and (chunk[1] - chunk[0] < pause_length):
                if i > 0:
                    prev_chunk = intervals[i-1]
                    if (prev_chunk[2] == 1) and (prev_chunk[1] == chunk[0]):
                        intervals[i, 2] = 1
                if i < len(intervals) - 1:
                    next_chunk = intervals[i+1]
                    if (next_chunk[2] == 1) and (next_chunk[0] == chunk[1]):
                        intervals[i, 2] = 1

        chunk = intervals[0]
        next_chunk = intervals[1]

        return intervals
    
    @staticmethod
    def combine_same_labels(intervals):
        """
        Combines intervals that are directly adjacent (makes labels more readable in Audacity
        Args:
            intervals (np.ndarray): time intervals for start and end of each lecturing chunk, given in SAMPLES
        Returns:
            np.ndarray: simplified intervals
        """

        # sr = 22050

        label_list = []
        if len(intervals) > 0:
            label_start = intervals[0][0]

            # max_length = 20000 * self.sr  # this keeps the intervals at a good length for speech recognition

            for i in range(0, len(intervals) - 1):
                current_interval = intervals[i]
                next_interval = intervals[i + 1]
                # check that the adjacent intervals are connected, and that they have the same label
                if (next_interval[0] == current_interval[1]) and (current_interval[2] == next_interval[2]):
                    label_end = next_interval[1]  # set new interval end to the end of the next interval

                    if i == len(intervals) - 2:
                        label_list.append(np.array([label_start, label_end, next_interval[2]]))

                else:
                    label_list.append(np.array([label_start, current_interval[1], current_interval[2]]))
                    label_start = next_interval[0]

                    if i == len(intervals) - 2:
                        label_list.append(next_interval)

        return np.array(label_list)

    @staticmethod
    def ignore_short_intervals(intervals, min_time):
        label_list = []

        for i in range(0, len(intervals)):
            chunk = intervals[i]
            if chunk[1] - chunk[0] > min_time:
                label_list.append(chunk)

        return np.array(label_list)

    @staticmethod
    def add_silent_labels(intervals):
        label_list = []

        for i in range(0, len(intervals) - 1):
            chunk = intervals[i]
            next_chunk = intervals[i + 1]

            if chunk[1] == next_chunk[0]:
                label_list.append(chunk)
            else:
                new_chunk = np.array([chunk[1], next_chunk[0], 0])
                label_list.extend([chunk, new_chunk])
            if i == len(intervals) - 2:
                label_list.append(next_chunk)

        return np.array(label_list)

    def full_analysis(self, threshold_all, threshold_lecture, hop_length, frame_length, pause_length, min_time):
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
        intervals_all = self.split_on_silence(threshold=threshold_all, hop_length=hop_length, frame_length=frame_length)

        # outputs time intervals for start and end of each lecturing chunk
        lecture_intervals = self.split_on_silence(threshold=threshold_lecture, hop_length=hop_length, frame_length=frame_length)

        intervals = self.integrate_interval_sets(lecture_intervals, intervals_all)

        intervals = self.fill_in_label_gaps(intervals, pause_length * self.sr)

        intervals = self.ignore_short_student_intervals(intervals, min_time * self.sr)

        intervals = self.combine_same_labels(intervals)

        intervals = self.ignore_short_intervals(intervals, min_time * self.sr)

        intervals = self.add_silent_labels(intervals)
        self.create_labels(intervals, 1)

        # find the percent silence removed and percent of lecture spent talking
        percent_trimmed, professor_talking, student_talking, silence, new_dur = lecture.count_time_spent(intervals)

        num_words, num_syllables, grade_level, most_common, common_fillers, common_positives, word_length_dist, syllable_dist, most_difficult = lecture.analyze_words()

        words_per_second = num_words / professor_talking
        words_per_minute = words_per_second * 60

        all_labels_arr = []

        for chunk in intervals:
            if chunk[2] == 0:
                label_name = "silence"
            elif chunk[2] == 1:
                label_name = "professor"
            elif chunk[2] == 2:
                label_name = "student"
            else:
                label_name = ""

            all_labels_arr.append({"start": (int(chunk[0]) + self.trimmed_offset)/ self.sr, "end": (int(chunk[1]) + self.trimmed_offset) / self.sr, "label": label_name})

        most_common_arr = []
        for word in most_common:
            most_common_arr.append({"word": word[0], "number": word[1]})

        common_fillers_arr = []
        for word in common_fillers:
            common_fillers_arr.append({"word": word[0], "number": word[1]})

        common_positives_arr = []
        for word in common_positives:
            common_positives_arr.append({"word": word[0], "number": word[1]})

        most_difficult_arr = []
        for word in most_difficult:
            most_difficult_arr.append({"word": word[0], "number": word[1]})

        response = {"percent_leading_trailing_silence_trimmed": percent_trimmed,
                    "student_talking_time": student_talking,
                    "professor_talking_time": professor_talking,
                    "silence_time": silence,
                    "class_duration": new_dur,
                    "words_spoken": num_words,
                    "average_syllables_per_word": num_syllables,
                    "grade_level": grade_level,
                    "words_per_minute": words_per_minute,
                    "words_per_second": words_per_second,
                    "most_common_words": most_common_arr,
                    "common_fillers": common_fillers_arr,
                    "common_positives": common_positives_arr,
                    "most_difficult": most_difficult_arr,
                    "word_lengths": word_length_dist,
                    "syllable_lengths": syllable_dist,
                    "all_labels": all_labels_arr
                    }

        # for chunk in response["all_labels"]:
        #     print(chunk)

        # convert into JSON:
        with open('{}.json'.format(remove_file_type(self.wav_filename)), 'w') as outfile:
            json.dump(response, outfile)

        response = json.dumps(response)

        return percent_trimmed, student_talking, professor_talking, silence, new_dur, num_words, num_syllables, grade_level, words_per_minute, words_per_second, intervals, most_common, common_fillers, common_positives, word_length_dist

    # eventually we want to do this as the last step, once we have both student and professor intervals
    def create_labels(self, intervals, i=None):
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
            filename = '{}_labels.txt'.format(remove_file_type(self.original_audio_filename))

        # otherwise, add the number to the end of the file name
        # this is used when multiple tests are being run on the same audio file
        else:
            filename = '{}_labels_{}.txt'.format(remove_file_type(self.original_audio_filename), i)

        # open the file to make sure we create it if it isn't there
        f = open(filename, "w+")
        f.close()

        # add a label for each interval
        for row in intervals:
            self.add_label_row(filename, row[0] / self.sr, row[1] / self.sr, row[2])

        return intervals

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
    transcript_file = sys.argv[1]
    audio_file = sys.argv[2]

    # select the base filename to be analyzed
    # transcript_file = 'BIS-2A_ 2019-07-17 12_10_transcript.txt'
    # audio_file = 'BIS-2A__2019-07-17_12_10.mp4'

    # create an instance of the LectureAudio class
    # extract audio info from wav file and trim leading and trailing silence
    lecture = LectureAudio(audio_file, transcript_file)

    # lecture.get_silence_threshold()

    # # create an instance of the LectureAudio class
    # # only load first 1200 seconds to make tests run faster
    # lecture = LectureAudio(audio_file, duration=1200)

    threshold_all = 30  # should be >= threshold_lecture
    threshold_lecture = 30
    hop_length = 1024
    frame_length = 2048
    pause_length = 2
    min_time = 2

    lecture.full_analysis(threshold_all, threshold_lecture, hop_length, frame_length, pause_length, min_time)


# THINGS TO BE AWARE OF
# 20 db is a good setting to get all speaking sound (includes background noise and student discussion
# 35 db is a good setting for just lecturing
# try to combine this data to get student participation time
