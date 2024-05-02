import wave
import pyaudio
import librosa
import matplotlib.pyplot as plt
import numpy as np


def percentage_of_time(session_duration: float, total_frames: float, frames: float):
    """
    :param session_duration: int
    :param total_frames: int
    :param frames: int
    :return: Amount of time spent, percentage of time spent during the session
    """
    chunk = ((session_duration*frames)/total_frames)
    percentage = round((chunk/session_duration)*100, 1)
    percentage = percentage if percentage < 100 else 100
    return chunk, percentage


def save_audio(filename, audio, channels, rate, frames: list):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))


def plot_spectrogram(audio_file):
    audio_file = audio_file
    signal, sr = librosa.load(audio_file, sr=44100)
    db = librosa.amplitude_to_db(abs(librosa.stft(signal)), ref=np.max)
    plt.figure(figsize=(15, 7))
    librosa.display.specshow(db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.savefig('plots/spectrogram.png')


def softmax(emotion_dict: dict):
    keys = [i for i in emotion_dict.keys()]
    values = [i for i in emotion_dict.values()]
    total = sum(values) if sum(values) > 0 else 0.000001
    for key, value in emotion_dict.items():
        emotion_dict[key] = str(round((value / total) * 100, 2)) + "%"
    return emotion_dict
