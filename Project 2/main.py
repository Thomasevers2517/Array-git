"""
This script implements an audio beamforming system using an array of microphones.
The goal of the project is to enhance the desired audio signal while suppressing noise and interference from other sources.
The script uses various beamforming techniques, including Minimum Variance Distortionless Response (MVDR), 
Multiple Signal Classification (MUSIC), and Wiener filter.

The main function of the script performs the following steps:
1. Loads the audio data from multiple sources.
2. Plays the combined audio signal.
3. Displays the real-time frequency spectrum of the audio signal using Pygame and Matplotlib.
4. Applies the MVDR beamformer to enhance the audio signal.
5. Plays the enhanced audio signal.
6. Repeats steps 3-5 for the MUSIC beamformer and Wiener filter.
7. Compares the results with the original microphone signals and calculates the Mean Squared Error (MSE).

ToDos:
    1. Check if the microphone array can be inferred from the impulse responses.
    2. Implement the beamforming algorithms (MVDR, MUSIC, Wiener filter), with the inter microphone distance and the speed of sound as parameters.
    3. Find a way to measure intelligibility and signal-to-noise ratio of the enhanced signals.
    4. Steer the beamformer to a specific direction, according to the intellegibility and signal-to-noise ratio.
    5. (Optional) Implement a GUI to visualize the beamforming process and the results.

Note: The script assumes that the audio data is stored in the 'src/data/' folder and follows a specific naming convention.
"""

import numpy as np
import simpleaudio as sa
import threading
import time
import matplotlib.pyplot as plt
import pygame
from pygame.locals import *
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import queue
from scipy.signal import butter, lfilter

import src.load_audio as load_audio
import src.beam_former as beam_former
import src.rtf as rtf
import src.intel_metric as intel_metric

# Settings for signal processing
LOW_CUT = 200  # Low cut-off frequency for bandpass filter
HIGH_CUT = 7000  # High cut-off frequency for bandpass filter

# Create a queue to hold the data
data_queue = queue.Queue()

class PlayerThread(threading.Thread):
    def __init__(self, audio_data, sample_rate, volume=1.0):
        threading.Thread.__init__(self, daemon=True)
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.volume = volume
        self.play_obj = None
        self.stopped = False

    def run(self):
        # Apply volume
        self.audio_data = np.int16(self.audio_data / np.max(np.abs(self.audio_data)) * 32767 * self.volume)  # Normalize audio to int16 range
        self.play_obj = sa.play_buffer(self.audio_data, 1, 2, self.sample_rate)  # 1 channel, 2 bytes per sample
        
        start_time = time.time()
        while not self.stopped:
            elapsed_time = time.time() - start_time
            start_sample = int(elapsed_time * self.sample_rate)
            end_sample = start_sample + int(0.1 * self.sample_rate)  # 100ms window

            if end_sample < len(self.audio_data):
                window_data = self.audio_data[start_sample:end_sample]
                data_queue.put((window_data, self.sample_rate))  # Put the data in the queue

            time.sleep(0.1)  # Sleep for 100ms

        self.play_obj.wait_done()

    def stop(self):
        if self.play_obj is not None:
            self.play_obj.stop()
        self.stopped = True

def show_spectrum(screen, ymax, audio_signal, sample_rate):
    # Compute the frequency spectrum of the audio signal
    freq_spectrum = np.fft.fft(audio_signal)
    freq_spectrum = np.abs(freq_spectrum[:len(freq_spectrum) // 2])  # Take only the positive frequencies

    # Create the frequency axis
    freq_axis = np.linspace(0, sample_rate / 2, len(freq_spectrum))

    # Create the figure and plot the spectrum
    dpi = 100  # Adjust this value if necessary
    fig = plt.figure(figsize=(800/dpi, 600/dpi), dpi=dpi)
    plt.plot(freq_axis, freq_spectrum)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.ylim(0, ymax)
    plt.title('Real-time Frequency Spectrum')

    # Convert the figure to a Pygame surface
    canvas = FigureCanvas(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    size = canvas.get_width_height()

    surf = pygame.image.fromstring(raw_data, size, "RGB")

    # Display the surface
    screen.blit(surf, (0,0))
    pygame.display.flip()

    # Close the figure
    plt.close(fig)

def play_audio_show_spectrum(audio_data, sample_rate, play_volume=0.33):
    ymax = 2 * 1e6  # Set the maximum value for the y-axis

    player = PlayerThread(audio_data, sample_rate, volume=play_volume)
    player.start()
    start_time = time.time()

    # Create a Pygame window
    screen = pygame.display.set_mode((800, 600))  # Set your desired size

    # Plot the data in the main thread
    while player.is_alive() and time.time() - start_time < 5:
        try:
            # Get the data from the queue
            window_data, sample_rate = data_queue.get(timeout=0.1)  # Wait for up to 100ms

            # Show the spectrum
            show_spectrum(screen, ymax, window_data, sample_rate)

        except queue.Empty:
            # If the queue is empty, check if the player thread is still running
            if not player.is_alive():
                break  # If the player thread has stopped, exit the loop
            else:
                # Wait for the player thread to finish
                player.join()
                break
    
    if player.is_alive():
        # Stop audio playback
        player.stop()
    
    # Close the Pygame window
    pygame.quit()

def bandpass_filter(signal, lowcut, highcut, fs, order=5):
    # Normalize the frequencies
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    # Design the Butterworth filter
    b, a = butter(order, [low, high], btype='band')

    # Apply the filter
    filtered_signal = np.zeros_like(signal)
    filtered_signal = lfilter(b, a, signal)

    return filtered_signal

def audio_compressor(signal, threshold, gain, ratio):
    threshold = 10
    gain = 0.5
    ratio = 2
    compressed_signal = np.zeros_like(signal)

    # Apply the volume compression
    if signal.shape[0] > 5: # 1D signal
        for i in range(len(signal)):
            if np.abs(signal[i]) > threshold:
                compressed_signal[i] = np.sign(signal[i]) * threshold + (np.abs(signal[i]) - threshold) * gain  # Hard clipping
            else:
                compressed_signal[i] = signal[i]
    
    if signal.shape[0] == 4: # 2D signal
        mic_num = signal.shape[0]
        for mic in range(mic_num):
            for i in range(len(signal[0])):
                if np.abs(signal[mic][i]) > threshold:
                    compressed_signal[mic][i] = threshold #np.sign(signal[mic][i]) * threshold + (np.abs(signal[mic][i]) - threshold) * gain
                else:
                    print("signal[mic][i]: ", signal[mic][i])
                    compressed_signal[mic][i] = signal[mic][i]
    
    if signal.shape[0] == 5: # 3D signal
        mic_num = signal.shape[1]
        source_num = signal.shape[0]
        for mic in range(mic_num):
            for source in range(source_num):
                for i in range(len(signal[0])):
                    if np.abs(signal[mic][source][i]) > threshold:
                        compressed_signal[mic][source][i] = threshold #np.sign(signal[mic][source][i]) * threshold + (np.abs(signal[mic][source][i]) - threshold) * gain
                    else:
                        compressed_signal[mic][source][i] = signal[mic][source][i]

    return compressed_signal

def eval_metrics(source_signal, enhanced_signal, sample_rate):
    stoi_score = np.round(intel_metric.calculate_stoi(source_signal[:len(enhanced_signal)], enhanced_signal, sample_rate), 2)
    estoi_score = np.round(intel_metric.calculate_estoi(source_signal[:len(enhanced_signal)], enhanced_signal, sample_rate), 2)
    siib_score = 0 #np.round(intel_metric.compute_siib(source_signal[:len(enhanced_signal)], enhanced_signal, sample_rate), 2)
    ncm_score = np.round(intel_metric.calculate_ncm(source_signal[:len(enhanced_signal)], enhanced_signal), 2)
    return stoi_score, estoi_score, siib_score, ncm_score

def main():
    # Initialize Pygame
    pygame.init()

    mic_signals = np.array(np.zeros((4, 80800)))
    source_signal = np.array(np.zeros((4, 80800)))
    path_signals = np.array(np.zeros((5, 4, 80800)))

    # ======================================== Load Audio Data ========================================
    impulse_responses, num_sources, num_mics, sample_rate, mic_signals, source_signal, path_signals = load_audio.convolve_audio_sources(data_folder="src/data/")
    # ======================================== Audio preprocessing ========================================

    # Add all the signals from the microphones with equal weigths to create the combined signal
    combined_signal = np.zeros_like(mic_signals[0])
    for i in range(0, num_mics):
        combined_signal += mic_signals[i]
    combined_signal = combined_signal / num_mics

    # Measure the threshold for source precense using the source signal
    # plot the audio sample in time domain
    # plt.figure()
    # plt.plot(source_signal)
    # plt.title('Source Signal')
    # plt.xlabel('Time')
    # plt.ylabel('Amplitude')
    # plt.grid()
    # plt.show()

    # Metric
    stoi_score, estoi_score, siib_score, ncm_score = eval_metrics(source_signal, combined_signal, sample_rate)
    print("combined_signal scores: STOI: ", stoi_score, " ESTOI: ", estoi_score, " SIIB: ", siib_score, " NCM: ", ncm_score)

    # # # Play the combined signal
    # play_audio_show_spectrum(combined_signal, sample_rate)
    # input("Press Enter to continue...")

    # Filter the all signals with a bandpass filter
    # print("Filtering the audio signals with a bandpass filter at ", LOW_CUT, "Hz and ", HIGH_CUT, "Hz...")
    # combined_signal = bandpass_filter(combined_signal, lowcut=LOW_CUT, highcut=HIGH_CUT, fs=sample_rate)
    # mic_signals[0] = bandpass_filter(mic_signals[0], lowcut=LOW_CUT, highcut=HIGH_CUT, fs=sample_rate)
    # mic_signals[1] = bandpass_filter(mic_signals[1], lowcut=LOW_CUT, highcut=HIGH_CUT, fs=sample_rate)
    # mic_signals[2] = bandpass_filter(mic_signals[2], lowcut=LOW_CUT, highcut=HIGH_CUT, fs=sample_rate)
    # mic_signals[3] = bandpass_filter(mic_signals[3], lowcut=LOW_CUT, highcut=HIGH_CUT, fs=sample_rate)
    # # source_signal = bandpass_filter(source_signal, lowcut=LOW_CUT, highcut=HIGH_CUT, fs=sample_rate)
    # # path_signals = bandpass_filter(path_signals, lowcut=LOW_CUT, highcut=HIGH_CUT, fs=sample_rate)

    # # Volume Compress the audio signal with a threshold at 20000, gain of 0.5 and a ratio of 2
    # print("Volume compressing the audio signals with a threshold at 20000, gain of 0.5 and a ratio of 2...")
    # combined_signal = audio_compressor(combined_signal, threshold=20000, gain=0.5, ratio=2)
    # mic_signals[0] = audio_compressor(mic_signals[0], threshold=20000, gain=0.5, ratio=2)
    # mic_signals[1] = audio_compressor(mic_signals[1], threshold=20000, gain=0.5, ratio=2)
    # mic_signals[2] = audio_compressor(mic_signals[2], threshold=20000, gain=0.5, ratio=2)
    # mic_signals[3] = audio_compressor(mic_signals[3], threshold=20000, gain=0.5, ratio=2)
    # #source_signal = audio_compressor(source_signal, threshold=20000, gain=0.5, ratio=2)
    # #path_signals = audio_compressor(path_signals, threshold=20000, gain=0.5, ratio=2)

    # # Play the combined signal
    # play_audio_show_spectrum(combined_signal, sample_rate)
    # input("Press Enter to continue...")

    # Metric
    # stoi_score, estoi_score, siib_score, ncm_score = eval_metrics(source_signal, combined_signal, sample_rate)
    # print("filtered_signal scores: STOI: ", stoi_score, " ESTOI: ", estoi_score, " SIIB: ", siib_score, " NCM: ", ncm_score)

    # ======================================== Relative Transfer Function (RTF) ========================================
    # Determine the relative transfer function (RTF) for each microphone
    rtf_cheating, Rn, sig_var, stft_mic_signals = rtf.determine_rtf_a_priori_CPSD(mic_signals, path_signals)  
    
    #  Estimate the relative transfer function (RTF) for each microphone using prewhitening
    rtf_estimated_prewhiten, Rn, sig_var, stft_mic_signals = rtf.estimate_rtf_Rs_prewhiten(mic_signals, path_signals, source_signal)

    #  Estimate the relative transfer function (RTF) for each microphone using prewhitening
    #rtf_estimated_prewhiten, Rn, sig_var, stft_mic_signals = rtf.estimate_rtf_Rs_prewhiten_working(mic_signals, path_signals, source_signal)
    #  Estimate the relative transfer function (RTF) for each microphone using GEVD
    #rtf_estimated_GEVD, Rn, sig_var, stft_mic_signals = rtf.estimate_rtf_GEVD(mic_signals, path_signals)

    # Infer geometry from impulse responses
    # geometry = rtf.infer_geometry_ULA(impulse_responses)
    # print("Geometry: ", geometry)

    # # Visualize geometry in a 2D plot
    # plt.figure()
    # plt.scatter(geometry[:num_sources, 0], geometry[:num_sources, 1], label='Sources')
    # # Highlight the last source in red
    # plt.scatter(geometry[num_sources-1, 0], geometry[num_sources-1, 1], color='red')
    # plt.scatter(geometry[num_sources:, 0], geometry[num_sources:, 1], label='Microphones')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('Geometry')
    # plt.legend()
    # plt.show()

    # ======================================== Delay-and-Sum Beamformer ========================================
    # Calculate a delay-and-sum beamformer
    print("Calculating delay-and-sum beamformer with cheating...")
    delay_and_sum_weights = beam_former.calculate_delay_and_sum_weights(rtf_cheating)

    # Apply the delay-and-sum beamformer to enhance the audio signal
    print("Applying delay-and-sum beamformer...")
    enhanced_signal_delay_and_sum = beam_former.apply_beamforming_weights(stft_mic_signals, delay_and_sum_weights)

    # Metric
    stoi_score, estoi_score, siib_score, ncm_score = eval_metrics(source_signal, enhanced_signal_delay_and_sum, sample_rate)
    print("Delay-and-Sum scores: STOI: ", stoi_score, " ESTOI: ", estoi_score, " SIIB: ", siib_score, " NCM: ", ncm_score)

    # # Play the enhanced signal
    # play_audio_show_spectrum(enhanced_signal_delay_and_sum, sample_rate)
    # input("Press Enter to continue...")
    
    # ======================================== Minimum Variance Distortionless Response (MVDR) ========================================
    # Calculate weigths for the MVDR beamformer
    print("Calculating MVDR beamformer with cheating...")
    w_mvdr = beam_former.calculate_mvdr_weights(rtf_cheating, Rn)

    # Apply the MVDR beamformer to enhance the audio signal
    print("Applying MVDR beamformer...")
    enhanced_signal_mvdr = beam_former.apply_beamforming_weights(stft_mic_signals, w_mvdr)

    # Metric
    stoi_score, estoi_score, siib_score, ncm_score = eval_metrics(source_signal, enhanced_signal_mvdr, sample_rate)
    print("MVDR scores: STOI: ", stoi_score, " ESTOI: ", estoi_score, " SIIB: ", siib_score, " NCM: ", ncm_score)

    # # Play the enhanced signal
    # play_audio_show_spectrum(enhanced_signal_mvdr, sample_rate)
    # input("Press Enter to continue...")

    # ======================================== Multi-Channel Wiener Filter ========================================
    # Calculate weigths for the Multi-Channel Wiener

    print("Calculating Multi-Channel Wiener beamformer with cheating...")
    w_MCWiener = beam_former.calculate_Multi_channel_Wiener_weigths(rtf_cheating, Rn, sig_var, w_mvdr)

    # Apply the Multi-Channel Wiener beamformer to enhance the audio signal
    print("Applying Multi-Channel Wiener beamformer...")
    enhanced_signal_MCWiener = beam_former.apply_beamforming_weights(stft_mic_signals, w_MCWiener)
    print("Enhanced signal (Multi-Channel Wiener Beamformer):", enhanced_signal_MCWiener)

    # Metric
    stoi_score, estoi_score, siib_score, ncm_score = eval_metrics(source_signal, enhanced_signal_MCWiener, sample_rate)
    print("Multi-Channel Wiener scores: STOI: ", stoi_score, " ESTOI: ", estoi_score, " SIIB: ", siib_score, " NCM: ", ncm_score)

    # # Play the enhanced signal
    # play_audio_show_spectrum(enhanced_signal_MCWiener, sample_rate)


    # ======================================== Delay-and-Sum Beamformer ========================================
    # Calculate a delay-and-sum beamformer
    print("Calculating delay-and-sum beamformer with detector...")
    delay_and_sum_weights = beam_former.calculate_delay_and_sum_weights(rtf_estimated_prewhiten)

    # Apply the delay-and-sum beamformer to enhance the audio signal
    print("Applying delay-and-sum beamformer...")
    enhanced_signal_delay_and_sum = beam_former.apply_beamforming_weights(stft_mic_signals, delay_and_sum_weights)

    # Metric
    stoi_score, estoi_score, siib_score, ncm_score = eval_metrics(source_signal, enhanced_signal_delay_and_sum, sample_rate)
    print("Delay-and-Sum scores: STOI: ", stoi_score, " ESTOI: ", estoi_score, " SIIB: ", siib_score, " NCM: ", ncm_score)

    # # Play the enhanced signal
    # play_audio_show_spectrum(enhanced_signal_delay_and_sum, sample_rate)
    # input("Press Enter to continue...")
    
    # ======================================== Minimum Variance Distortionless Response (MVDR) ========================================
    # Calculate weigths for the MVDR beamformer
    print("Calculating MVDR beamformer with detector...")
    w_mvdr = beam_former.calculate_mvdr_weights(rtf_estimated_prewhiten, Rn)

    # Apply the MVDR beamformer to enhance the audio signal
    print("Applying MVDR beamformer...")
    enhanced_signal_mvdr = beam_former.apply_beamforming_weights(stft_mic_signals, w_mvdr)

    # Metric
    stoi_score, estoi_score, siib_score, ncm_score = eval_metrics(source_signal, enhanced_signal_mvdr, sample_rate)
    print("MVDR scores: STOI: ", stoi_score, " ESTOI: ", estoi_score, " SIIB: ", siib_score, " NCM: ", ncm_score)

    # # Play the enhanced signal
    # play_audio_show_spectrum(enhanced_signal_mvdr, sample_rate)
    # input("Press Enter to continue...")

    # ======================================== Multi-Channel Wiener Filter ========================================
    # Calculate weigths for the Multi-Channel Wiener
    print("Calculating Multi-Channel Wiener beamformer with detector...")
    w_MCWiener = beam_former.calculate_Multi_channel_Wiener_weigths(rtf_estimated_prewhiten, Rn, sig_var, w_mvdr)

    # Apply the Multi-Channel Wiener beamformer to enhance the audio signal
    print("Applying Multi-Channel Wiener beamformer...")
    enhanced_signal_MCWiener = beam_former.apply_beamforming_weights(stft_mic_signals, w_MCWiener)
    print("Enhanced signal (Multi-Channel Wiener Beamformer):", enhanced_signal_MCWiener)

    # Metric
    stoi_score, estoi_score, siib_score, ncm_score = eval_metrics(source_signal, enhanced_signal_MCWiener, sample_rate)
    print("Multi-Channel Wiener scores: STOI: ", stoi_score, " ESTOI: ", estoi_score, " SIIB: ", siib_score, " NCM: ", ncm_score)

    # # Play the enhanced signal
    # play_audio_show_spectrum(enhanced_signal_MCWiener, sample_rate)

    # ======================================== Comparison and Evaluation ========================================
    # # Compare the enhanced signals with the original microphone signals
    # print("Comparing the enhanced signals with the original microphone signals...")
    # # Plot the original microphone signals
    # plt.figure()
    # for i in range(num_mics):
    #     plt.plot(np.abs(stft_mic_signals[i, 0, :]), label=f'Microphone {i+1}')
    # plt.xlabel('Time')
    # plt.ylabel('Magnitude')
    # plt.title('Original Microphone Signals')
    # plt.legend()
    # plt.show()

    # # Plot the enhanced signals
    # plt.figure()
    # plt.plot(np.abs(enhanced_signal_delay_and_sum), label='Delay-and-Sum')
    # plt.plot(np.abs(enhanced_signal_mvdr), label='MVDR')
    # plt.plot(np.abs(enhanced_signal_MCWiener), label='Multi-Channel Wiener')
    # plt.xlabel('Time')
    # plt.ylabel('Magnitude')
    # plt.title('Enhanced Signals')
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    main()