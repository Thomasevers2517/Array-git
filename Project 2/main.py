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

import src.load_audio as load_audio
import src.beam_former as beam_former
import src.atf as atf

# Create a queue to hold the data
data_queue = queue.Queue()

# Constants
NUM_OF_MIC = 4
NUM_OF_SRC = 5
STFT_WINDOW_TIME = 0.02 #seconds
NOISE_STD_DEV = 1

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

def main():
    # Initialize Pygame
    pygame.init()

    # Load audio data
    num_sources, num_mics, sample_rate, mic_signals, path_signals = load_audio.convolve_audio_sources(data_folder="src/data/")
    
    # Add all the signals from the microphones with equal weigths to create the combined signal
    combined_signal = np.zeros_like(mic_signals[0])
    for i in range(0, NUM_OF_MIC):
        combined_signal += mic_signals[i]
    combined_signal = combined_signal / NUM_OF_MIC

    # Determine the relative transfer function (RTF) for each microphone
    # atf_cheating, Rx, Rn, sig_var, stft_mic_signals = atf.determine_atf_a_priori(num_sources, num_mics, path_signals)
    # print("atf_cheating.shape: ", atf_cheating.shape)
    
    # Estimate the relative transfer function (RTF) for each microphone using prewhitening
    rtf_estimated_prewhiten, Rx, Rn, sig_var, stft_mic_signals = atf.estimate_rtf_prewhiten(num_mics, mic_signals)
    print("rtf_estimated_prewhiten: ", rtf_estimated_prewhiten)

    # # Estimate the relative transfer function (RTF) for each microphone using generalized eigenvalue decomposition (GEVD)
    # rtf_estimated_GEVD = atf.estimate_rtf_GEVD(impulse_responses)
    # print("rtf_estimated_GEVD: ", rtf_estimated_GEVD)

    # Infer geometry from impulse responses
    # geometry = atf.infer_geometry_ULA(impulse_responses)
    # print("Geometry: ", geometry)

    # # Visualize geometry in a 2D plot
    # plt.figure()
    # plt.scatter(geometry[:num_sources, 0], geometry[:num_sources, 1], label='Sources')
    # plt.scatter(geometry[num_sources:, 0], geometry[num_sources:, 1], label='Microphones')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('Geometry')
    # plt.legend()
    # plt.show()

    # Play the Original signal
    play_audio_show_spectrum(combined_signal, sample_rate, play_volume=0.33)
    input("Press Enter to continue...")

    # Calculate a dealy-and-sum beamformer
    print("Calculating delay-and-sum beamformer...")
    delay_and_sum_weights = beam_former.calculate_delay_and_sum_weights(rtf_estimated_prewhiten)

    # Visualize the beamforming weights
    #beam_former.calculate_output_SNR(delay_and_sum_weights, Rx, Rn)
    #beam_former.visualize_beamforming_weights(delay_and_sum_weights)
    #beam_former.visualize_beamforming_weights_polar(delay_and_sum_weights)

    # Apply the delay-and-sum beamformer to enhance the audio signal
    print("Applying delay-and-sum beamformer...")
    enhanced_signal_delay_and_sum = beam_former.apply_beamforming_weights(stft_mic_signals, delay_and_sum_weights)

    # Play the enhanced signal
    play_audio_show_spectrum(enhanced_signal_delay_and_sum, sample_rate)
    input("Press Enter to continue...")
    
    # Calculate weigths for the MVDR beamformer
    print("Calculating MVDR beamformer...")
    w_mvdr = beam_former.calculate_mvdr_weights(rtf_estimated_prewhiten, Rn)
    
    # Measure Visualize the beamforming weights
    #beam_former.calculate_output_SNR(w_mvdr, Rx, Rn)
    #beam_former.visualize_beamforming_weights(w_mvdr)
    #beam_former.visualize_beamforming_weights_polar(w_mvdr)

    # Apply the MVDR beamformer to enhance the audio signal
    print("Applying MVDR beamformer...")
    enhanced_signal_mvdr = beam_former.apply_beamforming_weights(stft_mic_signals, w_mvdr)
    print("Enhanced signal (MVDR Beamformer):", enhanced_signal_mvdr)

    # Play the enhanced signal
    play_audio_show_spectrum(enhanced_signal_mvdr, sample_rate)
    input("Press Enter to continue...")

    # Apply the Multi-Channel Wiener
    print("Calculating Multi-Channel Wiener beamformer...")
    w_MCWiener = beam_former.calculate_Multi_channel_Wiener_weigths(rtf_estimated_prewhiten, Rn, sig_var, w_mvdr)

    # Visualize the beamforming weights
    #beam_former.visualize_beamforming_weights(w_MCWiener)
    #beam_former.visualize_beamforming_weights_polar(w_MCWiener)

    # Apply the Multi-Channel Wiener beamformer to enhance the audio signal
    print("Applying Multi-Channel Wiener beamformer...")
    enhanced_signal_MCWiener = beam_former.apply_beamforming_weights(stft_mic_signals, w_MCWiener)
    print("Enhanced signal (Multi-Channel Wiener Beamformer):", enhanced_signal_MCWiener)

    # Play the enhanced signal
    play_audio_show_spectrum(enhanced_signal_MCWiener, sample_rate)
    input("Press Enter to continue...")

    # Compare the results with the original signals
    print("Original microphone signals:", combined_signal)
    
    # Calculate the mean squared error (MSE) between the original and enhanced signals
    mse_delay_and_sum = np.mean((enhanced_signal_delay_and_sum[:len(combined_signal)] - combined_signal)**2)
    mse_mvdr = np.mean((enhanced_signal_mvdr[:len(combined_signal)] - combined_signal)**2)
    mse_wiener_filter = np.mean((enhanced_signal_MCWiener[:len(combined_signal)] - combined_signal)**2)
    
    # Print the results
    print("MSE (Delay-and-Sum Beamformer):", mse_delay_and_sum)
    print("MSE (MVDR Beamformer):", mse_mvdr)
    print("MSE (Wiener Filter):", mse_wiener_filter)

    # Play the combined signal
    play_audio_show_spectrum(combined_signal, sample_rate, play_volume=0.33)
    input("Press Enter to continue...")

    # TODO: Add code here to measure intelligibility and signal-to-noise ratio of the enhanced signals

if __name__ == "__main__":
    main()