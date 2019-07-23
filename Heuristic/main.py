# import find_silence
import librosa


audio_arr, sr = librosa.load('output_audio.wav', duration=600)
trimmed_audio, index = librosa.effects.trim(audio_arr, top_db=45)

print('from {} to {}'.format(index[0] / sr, index[1] / sr))
librosa.output.write_wav('{}_trimmed.wav'.format('output_audio'), trimmed_audio, sr)
