# Lecture Audio Analysis
_Center for Educational Effectiveness at UC Davis_

We began with the following goals:
- Split audio clips into **single voice**, **multiple voic**e, **no voice**, and **other**
- Determine amount of **interactive learning** in audio file
- Determine whether a question has been asked by the professor

The project was broken into two approaches; one using **machine learning** and the other using a **heuristic** approach.

Using the `librosa` Python module, we split the audio into silence and lecturing. This works for high quality audio with very little ambient sound, such as from **UC Davis lecture capture**. This provides a rough idea of how many minutes were spent with the professor lecturing. In the future we could also explore number of words and/or questions asked.

Using machine learning, we explored a **Sklearn** `Random Forest Classifier` and a **Keras** `Sequential model`. These were initially meant to split audio into 4 labels: **silence** (0), **lecturing** (1), **student discussion** (2), and **other** (3). However, with insufficient data from ~50 lectures (as opposed to the 1,500 we wanted) we may shift our approach to only recognizing student participation (discussion, asking questions, etc.)

## Compatible Audio

To use this library, all files must be in `.wav` format. The easiest way to convert them is to download ffmpeg:

[ffmpeg download](https://ffmpeg.org/download.html)

Then run this command to convert files:

```console
ffmpeg -i input_file.mp4 output_file.wav
```

All label `.txt` files are compatible with [Audacity](https://www.audacityteam.org/). They can be imported as labels for the matching audio `.wav` file.
## Getting Started with Machine Learning analysis

### 1. Test many versions of Random Forest Classifiers on different data sets
```python
data_candidates = None                  # default in grid_search_obj.py is DATA_CANDIDATES
data_candidate_strings = None           # default in grid_search_obj.py is DATA_CANDIDATE_STRINGS
parameter_candidates = None             # default in grid_search_obj.py is PARAMETER_CANDIDATES
test_track = False,                     # default in grid_search_obj.py is False
data_filename = None                    # default in grid_search_obj.py is DEFAULT_DATA_FILENAME = 'data017.pickle'
best_models = grid_search_test(data_candidates, data_candidate_strings, parameter_candidates, test_track, data_filename)
```

### 2. Test one Random Forest Classifier
```python
data_filename = None                    # default in random_forest_test.py is DATA_FILENAME = 'data017.pickle'
test_split = None                       # default in random_forest_test.py is DEFAULT_TEST_SPLIT = .4
feat_labels = None                      # default in random_forest_test.py is FEAT_LABELS = ...
data_set_name = None                    # default in random_forest_test.py is DEFAULT_NAME = 'RFC test data'
test_rfc(data_filename, test_split, feat_labels, data_set_name)
```

### 3. Save a confusion matrix with % similarity between label sets as a png file
```python
label1 = 'track10_labels.txt'
label2 = 'track10_labels_Rachel.txt'
compare_labels('track10', [label1, label2])
```

### 4. For testing purposes, displays the wave file visually with labels filled
```python
display_segmented_audio('track2.wav', 'track2_labels.txt')
```

### 5. Create and save a new data file
```python
my_clip_size = None                 # default in audio_file_obj.py is DEFAULT_CLIP_SIZE = 500
my_window_size = None               # default in audio_file_obj.py is DEFAULT_WINDOW_SIZE = 2500 ie 5 seconds total
my_window_size2 = None              # default in audio_file_obj.py is DEFAULT_WINDOW_SIZE2 = 10000 ie 20 seconds total
data_types = ['mean', 'sd', 'mean_surr', 'sd_surr', 'mean_surr2', 'sd_surr2', 'sd_full', 'mean_full', 'label',
              'mean_left', 'mean_right', 'sd_left', 'sd_right', 'mean_surr_left', 'sd_surr_left']
data = AudioData(data_types, clip_size=my_clip_size, window_size=my_window_size, window_size2=my_window_size2)
data.print()
data.pickle_data()
```

## Getting Started with Heuristic analysis

For quick use of `analyze_audio.py` just type the following into the command line:
```console
analyze_audio.py MY_FILENAME.txt MY_FILENAME.wav
```
Transcript files must be a text file. Audio/video files can be in `.mp3`, `.mp4`, or `.wav` format.

All audio files must be in `.wav` format since this module uses `librosa`. All transcript files must be in `.txt` format and include accurate words, punctuation, and a timestamp of when speaking begins. The easiest way to get the correct format is to take the transcription directly from Aggie Video. To load a file, use a string without the file extension. So if your file was named `input_audio.wav`, load it as shown:

```python
audio_file = 'input_audio'
transcript_file = 'input_audio_transcript'

# create an instance of the LectureAudio class
# extract audio info from wav file and trim leading and trailing silence
# download_needed (boolean) is true if files need to be downloaded from an s3 bucket
lecture = LectureAudio(audio_file, transcript_file, download_needed=False)
```

If the file is really long and you want to speed up testing, select only a limited number of seconds as shown:
```python
# Only load first 5 seconds to make tests run faster
# MAKE SURE YOUR TRANSCRIPTION ENDS WHEN THE AUDIO FILE ENDS
lecture_first_5_seconds = LectureAudio(audio_file, transcript_file, duration=5)
```

Once the LectureAudio object is initialized you can produce sample label files for different silence trimming thresholds. You can compare the accuracy in [Audacity](https://www.audacityteam.org/) and then run final_split() with the most accurate threshold(s).
```python
# run and analyze this test if unsure what params will work best for final split
lecture.get_silence_threshold()
```

Now that you found the best parameters, run all available tests using the `full_analysis()` method:
```python
threshold_all = 35
threshold_lecture = 30
hop_length = 1024
frame_length = 2048
pause_length = 2
min_time = 1

lecture.full_analysis(threshold_all, threshold_lecture, hop_length, frame_length, pause_length, min_time)
```
### If you want to only use specific methods, don't use `full_analysis()`:
