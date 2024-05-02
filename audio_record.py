import pyaudio
import wave
from tqdm.auto import tqdm
from utilis import plot_spectrogram


def record_audio(filename, duration: int = 60, chunk=1024, channels=1, rate=44100):
    audio = pyaudio.PyAudio()

    # Open recording stream
    stream = audio.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=rate,
                        input=True,
                        frames_per_buffer=chunk)


    print("Recording...")
    frames = []

    # Record audio
    for i in tqdm(range(0, int(rate / chunk * duration))):
        data = stream.read(chunk)
        frames.append(data)

    print("Finished recording.")

    # Stop recording stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save recorded audio to file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))


if __name__ == "__main__":
    duration = int(input("Enter the duration: "))
    record_audio(r"audio/output.wav", duration=duration)
    plot_spectrogram(r'audio/output.wav')
