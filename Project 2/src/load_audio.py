"""
This file loads the .mat file and convolves .wav files with sources, then each sources at each mic can be listened to.
The main goal of this source file is to load the audio and impulse responses and convolve. The listening part can be deleted and so the 
python libraries required for this also do not have to be installed.

To install the "soundfile" module, run: "python -m pip install soundfile"
To install "simpleaudio" module, run: "python -m pip install simpleaudio". I needed Microsoft Visual C++ 14.0 or greater to build the wheels, installing the build tools fixed it for me:https://visualstudio.microsoft.com/visual-cpp-build-tools/
"""

import os
import numpy as np
import scipy.io as sio
import scipy.signal as signal
import soundfile as sf 
import simpleaudio as sa
import time
import threading

class PlayerThread(threading.Thread):
    def __init__(self, audio_data, samplerate):
        threading.Thread.__init__(self, daemon=True)
        self.audio_data = audio_data
        self.samplerate = samplerate
        self.play_obj = None
        self.stopped = False

    def run(self):
        self.audio_data = np.int16(self.audio_data / np.max(np.abs(self.audio_data)) * 32767)  # Normalize audio to int16 range
        self.play_obj = sa.play_buffer(self.audio_data, 1, 2, self.samplerate)  # 1 channel, 2 bytes per sample
        self.play_obj.wait_done()

    def stop(self):
        if self.play_obj is not None:
            self.play_obj.stop()
        self.stopped = True

def load_mat_file(file_path):
    mat_contents = sio.loadmat(file_path)
    impulse_responses = np.stack((mat_contents['h_inter1'], mat_contents['h_inter2'], mat_contents['h_inter3'], mat_contents['h_inter4'], mat_contents['h_target']))
    return impulse_responses

def load_wav_file(file_path):
    data, samplerate = sf.read(file_path)
    return data, samplerate

def convolve_signal(signal_data, impulse_response):

    # Ensure signal_data is 1-dimensional
    if signal_data.ndim != 1:
        signal_data = np.ravel(signal_data)

    return signal.convolve(signal_data, impulse_response, mode='full')

def get_user_input(player):
    while True:
        user_input = input()
        if user_input == 'q':
            player.stop()
            return

def convolve_audio_sources(wav_file_paths, mat_file_path='data/impulse_responses.mat', play_path_audio=False):
    impulse_responses = load_mat_file(mat_file_path)

    num_sources = impulse_responses.shape[0]
    num_mics = impulse_responses.shape[1]
    path_signals = [None] * num_sources

    for source_idx in range(num_sources):
        wav_file_path = wav_file_paths[source_idx]
        signal_data, samplerate = load_wav_file(wav_file_path)

        mic_signals = [None] * num_mics
        for mic_idx in range(num_mics):
            print(f"Convolve audio sample {wav_file_path} on source {source_idx+1} to microphone {mic_idx+1}")
            impulse_response = impulse_responses[source_idx, mic_idx]
            convolved_signal = convolve_signal(signal_data, impulse_response)

            if play_path_audio:
                print(f'Playing sound from source {source_idx+1} to microphone {mic_idx+1} for up to 5 seconds (Press "q" to go to next path)')
                player = PlayerThread(convolved_signal, samplerate)
                player.start()

                input_thread = threading.Thread(target=get_user_input, args=(player,), daemon=True)
                input_thread.start()

                start_time = time.time()
                while player.is_alive() and time.time() - start_time < 5:
                    pass
                
                if player.is_alive():
                    player.stop()
                    player.join()
            mic_signals[mic_idx] = convolved_signal
        path_signals[source_idx] = mic_signals
    return path_signals

if __name__ == "__main__":
    wav_file_1 = "data/clean_speech.wav"
    wav_file_2 = "data/babble_noise.wav"
    wav_file_3 = "data/speech_shaped_noise.wav"
    wav_file_4 = "data/aritificial_nonstat_noise.wav"
    wav_file_5 = "data/clean_speech_2.wav"
    wav_file_paths = [wav_file_1, wav_file_2, wav_file_3, wav_file_4, wav_file_5]
    try:
        path_signals = convolve_audio_sources(wav_file_paths, play_path_audio = False)
    except KeyboardInterrupt:
        print("KeyboardInterrupt: Stopping all threads and exiting...")
