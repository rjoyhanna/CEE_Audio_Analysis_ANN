import subprocess
import boto3


# redefine these values when we change to the CEE AWS account
S3_BUCKET_NAME = 'cee-audio-analysis'
# the server that runs this python code must have a folder for the audio
AUDIO_FOLDER_NAME = 's3'

VALID_WAV = '.wav'
VALID_MP3 = '.mp3'
VALID_TXT = '.txt'
VALID_MP4 = '.mp4'
VALID_AUDIO = [VALID_MP4, VALID_WAV, VALID_MP3]


def return_file_type(string):
    return string[-4:]


def remove_file_type(string):
    return string[:-4]


class DownloadConvertFiles:
    def __init__(self, video_name, transcript_name):
        audio_file_ext = return_file_type(video_name)

        if audio_file_ext not in VALID_AUDIO:
            raise ValueError('video_name must be one of: {}'.format(VALID_AUDIO))
        if return_file_type(transcript_name) != VALID_TXT:
            raise ValueError('transcript_name must be of type {}'.format(VALID_TXT))

        # if audio_file_ext == VALID_MP3:
        # elif audio_file_ext == VALID_WAV:
        #
        # elif audio_file_ext == VALID_MP4:

        self.audio_filename = video_name
        self.wav_filename = remove_file_type(video_name) + '.wav'
        self.transcript_filename = transcript_name

        self.video_convert_command = 'ffmpeg -i s3/{} s3/{}'.format(self.audio_filename, self.wav_filename)

        self.download_files()
        self.convert_file()

        self.s3 = boto3.resource('s3')

    def download_files(self):
        s3 = boto3.resource('s3')
        s3.Bucket(S3_BUCKET_NAME).download_file(self.transcript_filename,
                                                      '{}/{}'.format(AUDIO_FOLDER_NAME, self.transcript_filename))
        s3.Bucket(S3_BUCKET_NAME).download_file(self.audio_filename,
                                                      '{}/{}'.format(AUDIO_FOLDER_NAME, self.audio_filename))

    def convert_file(self):
        subprocess.call(self.video_convert_command, shell=True)

    def upload_file(self, new_filename):
        print('{} has started its upload to your s3 bucket!'.format(new_filename))
        s3 = boto3.resource('s3')
        s3.Bucket(S3_BUCKET_NAME).upload_file(new_filename, new_filename)
        print('{} has finished uploading to your s3 bucket!'.format(new_filename))


if __name__ == '__main__':
    # _tester.wav
    converter = DownloadConvertFiles('BIS-2A__2019-07-17_12_10.mp4', 'BIS-2A_ 2019-07-17 12_10_transcript.txt')
    # converter.download_files()
    converter.convert_file()
