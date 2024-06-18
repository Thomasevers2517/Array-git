import numpy as np
import pystoi
from scipy.signal import hilbert
from gammatone.gtgram import gtgram
from scipy.signal import stft

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# STFT specs
N_PER_SEG = 400
N_OVERLAP = 200
SAMPLE_FREQ = 16000
STFT_WINDOW = 'hann'
STFT_TIME = N_PER_SEG/SAMPLE_FREQ
STFT_OVERLAP = STFT_TIME/2

# Gamma filter specs
GAMMA_NUM_BANDS = 28
GAMMA_LOW_FREQ = 100
GAMMA_HIGH_FREQ = 8000 # half of sample frequency is standard, but in research this was 6500?

def visualize_beamforming_weights(beamforming_weights):
    # Sum the beamforming weights across the microphone dimension
    print("beamforming_weights shape: ", beamforming_weights.shape)
    beamforming_weights_sum = np.sum(beamforming_weights, axis=2)
    print("beamforming_weights shape: ", beamforming_weights.shape)
    # Calculate the magnitude of the beamforming weights
    beamforming_weights_magnitude = np.abs(beamforming_weights_sum)

    # Normalize the beamforming weights to the range [0, 1]
    beamforming_weights_magnitude = (beamforming_weights_magnitude - np.min(beamforming_weights_magnitude)) / (np.max(beamforming_weights_magnitude) - np.min(beamforming_weights_magnitude))

    # Create a heatmap of the beamforming weights
    plt.figure(figsize=(10, 10))
    plt.imshow(beamforming_weights_magnitude, aspect='auto', cmap='jet', vmin=np.min(beamforming_weights_magnitude), vmax=np.max(beamforming_weights_magnitude))
    plt.colorbar(format='%+2.0f dB')
    plt.title('Beamforming Weights (Magnitude)')
    plt.xlabel('Time Slots')
    plt.ylabel('Wave Numbers')
    plt.show()

def visualize_beamforming_weights_polar(beamforming_weights):
    # Sum the beamforming weights across the microphone dimension
    beamforming_weights_sum = np.sum(beamforming_weights, axis=2)

    # Calculate the magnitude and phase of the summed beamforming weights
    beamforming_weights_magnitude = np.abs(beamforming_weights_sum)
    beamforming_weights_phase = np.angle(beamforming_weights_sum)

    # Create a figure for the animation
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)

    # Initialize the plot
    line, = ax.plot([], [], 'o-')

    # Animation update function
    def update(i):
        line.set_ydata(beamforming_weights_magnitude[i])
        line.set_xdata(beamforming_weights_phase[i])
        return line,

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=range(len(beamforming_weights_magnitude)), blit=True)

    plt.title('Beamforming Weights (Polar Plot)')
    plt.show()

def calculate_output_SNR(rtf, Rx, Rn):
    # Get the number of wave numbers and timeslots
    num_wave_numbers = rtf.shape[0]
    num_timeslots = rtf.shape[1]

    # Calculate the output SNR of the enhanced signal
    output_SNR = np.zeros((num_wave_numbers, num_timeslots))
    for l in range(num_timeslots):
        for k in range(num_wave_numbers):
            # Calculate the output SNR of the enhanced signal
            output_SNR[k][l] = 10 * np.log10(rtf[k][l].conj().T @ Rx[k][l] @ rtf[k][l] / (rtf[k][l].conj().T @ Rn[k][l] @ rtf[k][l]))
    
    # Plot the output SNR of the enhanced signal
    plt.figure(figsize=(10, 10))
    plt.imshow(output_SNR, aspect='auto', cmap='jet', vmin=np.min(output_SNR), vmax=np.max(output_SNR))
    plt.colorbar(format='%+2.0f dB')
    plt.title('Output SNR of Enhanced Signal')
    plt.xlabel('Time Slots')
    plt.ylabel('Wave Numbers')
    plt.show()

    return output_SNR

def calculate_stoi(clean_signal, processed_signal, fs):
    stoi_score = pystoi.stoi(clean_signal, processed_signal, fs, extended=False)
    return stoi_score

def auditory_filtering(signal, fs, num_bands=GAMMA_NUM_BANDS, low_freq=GAMMA_LOW_FREQ, high_freq=GAMMA_HIGH_FREQ):
    # Gammatone spectrogram
    gtgram_result = gtgram(signal, fs, STFT_TIME, STFT_OVERLAP, num_bands, low_freq) # High frequency is taken as half of the sample frequency by default, can be worked around
    return gtgram_result

def modulation_spectrum_analysis(filtered_signal):
    num_bands, num_frames = filtered_signal.shape
    modulation_spectra = []

    for band in range(num_bands):
        # Calculate the envelope using the Hilbert transform
        envelope = np.abs(hilbert(filtered_signal[band, :]))

        # Perform FFT on the envelope
        modulation_spectrum = np.abs(np.fft.fft(envelope))[:num_frames // 2]
        modulation_spectra.append(modulation_spectrum)
    
    return np.array(modulation_spectra)

def compare_modulation_spectra(mod_spec_clean, mod_spec_processed):
    # Compare modulation spectra and compute SIIB score
    # This is a simplified example. In practice, you might need to calculate mutual information or other metrics.
    # Assuming mod_spec_clean and mod_spec_processed are numpy arrays of the same shape
    siib_score = np.corrcoef(mod_spec_clean.flatten(), mod_spec_processed.flatten())[0, 1]
    return siib_score

def compute_siib(clean_signal, processed_signal, fs):
    # STFT
    _, _, stft_clean_signal = stft(clean_signal, fs=SAMPLE_FREQ, window=STFT_WINDOW, nperseg=N_PER_SEG, noverlap=N_OVERLAP)
    _, _, stft_processed_signal = stft(processed_signal, fs=SAMPLE_FREQ, window=STFT_WINDOW, nperseg=N_PER_SEG, noverlap=N_OVERLAP)
    
    # Apply auditory filtering
    filtered_clean = auditory_filtering(stft_clean_signal, fs)
    filtered_processed = auditory_filtering(stft_processed_signal, fs)

    # Modulation spectrum analysis
    mod_spec_clean = modulation_spectrum_analysis(filtered_clean)
    mod_spec_processed = modulation_spectrum_analysis(filtered_processed)

    # Calculate SIIB score based on modulation spectra
    siib_score = compare_modulation_spectra(mod_spec_clean, mod_spec_processed)
    return siib_score