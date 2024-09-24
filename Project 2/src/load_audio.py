import numpy as np
import scipy.io as sio
import scipy.signal as signal
import soundfile as sf 
import matplotlib.pyplot as plt

# STFT specs
N_PER_SEG = 400
N_OVERLAP = 200
SAMPLE_FREQ = 16000
STFT_WINDOW = 'hann'
STFT_TIME = N_PER_SEG/SAMPLE_FREQ
STFT_OVERLAP = STFT_TIME/2

def show_spectrum(audio_signal, sample_rate=16000):
    # Compute the frequency spectrum of the audio signal
    freq_spectrum = np.fft.fft(audio_signal)
    freq_spectrum = np.abs(freq_spectrum[:len(freq_spectrum) // 2])  # Take only the positive frequencies

    # Create the frequency axis
    freq_axis = np.linspace(0, sample_rate / 2, len(freq_spectrum))

    # Plot the real-time frequency spectrum
    fig = plt.figure()
    plt.plot(freq_axis, freq_spectrum)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Real-time Frequency Spectrum')
    return fig

def load_mat_file(file_path):
    mat_contents = sio.loadmat(file_path)
    impulse_responses = np.stack((mat_contents['h_inter1'], mat_contents['h_inter2'], mat_contents['h_inter3'], mat_contents['h_inter4'], mat_contents['h_target']))
    return impulse_responses

def load_wav_file(file_path):
    data, sample_rate = sf.read(file_path)
    return data, sample_rate

def convolve_signal(signal_data, impulse_response):
    # Ensure signal_data is 1-dimensional
    if signal_data.ndim != 1:
        signal_data = np.ravel(signal_data)

    return signal.convolve(signal_data, impulse_response, mode='full')

def convolve_audio_sources(data_folder ="data/"):
    wav_file_1 = data_folder + "clean_speech_2.wav"
    wav_file_2 = data_folder + "babble_noise.wav"
    wav_file_3 = data_folder + "speech_shaped_noise.wav"
    wav_file_4 = data_folder + "aritificial_nonstat_noise.wav"
    wav_file_5 = data_folder + "clean_speech.wav"
    wav_file_paths = [wav_file_1, wav_file_2, wav_file_3, wav_file_4, wav_file_5]
    
    mat_file_path = data_folder + "impulse_responses.mat"

    # Load impulse responses from .mat file
    print("Loading impulse responses from .mat file...")
    impulse_responses = load_mat_file(mat_file_path) 

    # Calculate the number of sources and microphones
    num_sources = impulse_responses.shape[0]
    num_mics = impulse_responses.shape[1]
    print("Number of sources: ", num_sources)
    print("Number of microphones: ", num_mics)

    # Upsample impulse reso=ponses to 16 kHz
    # impulse_responses = np.array([signal.resample(impulse_response, 16000) for impulse_response in impulse_responses])

    # Find the minimum length of the audio signals
    min_length = (SAMPLE_FREQ * 5) + 800  # hard coded to the minimum length of the signals, approx 5 seconds, and fits the STFT settings
    print("Truncated length of audio signals: ", min_length)

    # # Plot impulse response per source and microphone
    # fig, axs = plt.subplots(num_sources, num_mics, figsize=(15, 10))
    # for source_idx in range(num_sources):
    #     for mic_idx in range(num_mics):
    #         impulse_response = impulse_responses[source_idx, mic_idx]
    #         axs[source_idx, mic_idx].plot(impulse_response)
    #         axs[source_idx, mic_idx].set_title(f"Source {source_idx + 1} - Mic {mic_idx + 1}")
    # plt.tight_layout()
    # plt.show()
    
    # Initialize the combined signals, path signals and clean signals
    combined_signals = np.array([None] * num_mics)
    path_signals = np.array([[None] * num_mics] * num_sources)
    mic_signals = np.array([None] * num_mics)
    source_signal = np.array([None] * num_mics)
    noise_signal = np.array([None] * num_mics)

    # Convolve audio sources with impulse responses
    print("Convolving audio sources with impulse responses...")

    max_amplitude = 0
    for source_idx in range(num_sources):
        wav_file_path = wav_file_paths[source_idx]
        signal_data, sample_rate = load_wav_file(wav_file_path)

        combined_signals = [None] * num_mics
        for mic_idx in range(num_mics):
            impulse_response = impulse_responses[source_idx, mic_idx]
            convolved_signal = convolve_signal(signal_data, impulse_response)
            combined_signals[mic_idx] = convolved_signal
            path_signals[source_idx][mic_idx] = convolved_signal[:min_length]
    
            if combined_signals[mic_idx] is None:
                combined_signals[mic_idx] = convolved_signal
            else:
                # Truncate to the length of the samples to save on computation time
                shaped_combined_signals = combined_signals[mic_idx][:min_length]
                shaped_convolved_signal = convolved_signal[:min_length]
                
                # Now you can add the signals together
                combined_signals[mic_idx] = np.sum([shaped_convolved_signal, shaped_combined_signals], axis=0)

                # Find the maximum sound amplitude over alle signals and mics
                max_amplitude_in_signal = np.max(combined_signals[mic_idx])
                max_amplitude = max(max_amplitude, max_amplitude_in_signal)
    
    # Calculate the RMS of the signals
    rms = np.sqrt(np.mean(np.square(combined_signals)))
    print("RMS of the audio at the reference microphone: ", rms)

    # Gain the signals to with 1e3 to prevent floating point errors
    print(f"Apply {np.round((1/rms),2)} gain to the signals to normalize...")
    for mic_idx in range(num_mics):
        mic_signals[mic_idx] =  np.sum(path_signals[:5, mic_idx]) * (1/rms)
        source_signal[mic_idx] = path_signals[4][mic_idx] * (1/rms)
        noise_signal[mic_idx] = np.sum(path_signals[:4, mic_idx]) * (1/rms)
    
    # Calculate the RMS of the signals after gain
    rms = np.sqrt(np.mean(np.square(mic_signals[0])))
    print("RMS of the reference microphone after gain: ", rms)
    rms = np.sqrt(np.mean(np.square(source_signal[0])))
    print("RMS of the source signal after gain: ", rms)
    rms = np.sqrt(np.mean(np.square(noise_signal[0])))
    print("RMS of the noise signal after gain: ", rms)

    return impulse_responses, num_sources, num_mics, sample_rate, mic_signals, noise_signal, source_signal

if __name__ == "__main__":
    try:
        _,mic_signals,_,_ = convolve_audio_sources()
    except KeyboardInterrupt:
        print("KeyboardInterrupt: Stopping all threads and exiting...")
