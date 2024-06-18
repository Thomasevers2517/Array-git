import numpy as np
from scipy.signal import stft, istft
import matplotlib.pyplot as plt
import matplotlib.animation as animation

N_PER_SEG = 512
N_OVERLAP = 256

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
    
def apply_beamforming_weights(stft_mic_signals, beamforming_weights, fs=16000, nperseg=N_PER_SEG, noverlap=N_OVERLAP):
    # Get the number of microphones
    num_mics = stft_mic_signals.shape[0]

    # Get the number of wave numbers and timeslots
    num_wave_numbers = stft_mic_signals.shape[1]
    num_timeslots = stft_mic_signals.shape[2]

    # Apply the beamforming weights to the microphone signals per wave number and timeslot and sum all microphone signals
    enhanced_signal_freq = np.zeros((num_wave_numbers, num_timeslots), dtype=np.complex128)
    for l in range(num_timeslots):
        for k in range(num_wave_numbers):
            for mic_idx in range(num_mics):
                enhanced_signal_freq[k][l] += np.multiply(beamforming_weights[k][l][mic_idx].conj().T, stft_mic_signals[mic_idx][k][l])

    # Calculate the inverse STFT of the enhanced signal in the frequency domain
    enhanced_signal = np.array(istft(enhanced_signal_freq, fs, nperseg=nperseg, noverlap=noverlap, freq_axis=0, time_axis=1))[1]
   
    return enhanced_signal

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

def calculate_delay_and_sum_weights(rtf):
    # Get the number of wave numbers and timeslots
    num_wave_numbers = rtf.shape[0]
    num_timeslots = rtf.shape[1]

    # Calculate the delay-and-sum beamforming weights
    delay_and_sum_beamformer = np.zeros(rtf.shape, dtype=np.complex64)
    for l in range(num_timeslots):
        for k in range(num_wave_numbers):
            delay_and_sum_beamformer[k][l] = rtf[k][l] / np.multiply(rtf[k][l].conj().T, rtf[k][l])
   
    return delay_and_sum_beamformer

def calculate_mvdr_weights(rtf, Rx):

    # Get the number of wave numbers and timeslots
    num_wave_numbers = rtf.shape[0]
    num_timeslots = rtf.shape[1]

    # Calculate the delay-and-sum beamforming weights
    mvdr_beamformer = np.zeros(rtf.shape, dtype=np.complex64)
    inverse_covariance_matrix = np.zeros(Rx.shape, dtype=np.complex64)
    for l in range(num_timeslots):
        for k in range(num_wave_numbers):
            # Add a small amount of noise to the diagonal of the covariance matrix
            Rx_reg = Rx[k][l] + np.eye(Rx[k][l].shape[0]) * 1e-6

            # Calculate the beamforming weights using the minimum variance distortionless response (MVDR) algorithm
            inverse_covariance_matrix[k][l] = np.linalg.inv(Rx_reg)
            
            mvdr_beamformer[k][l] = np.dot(inverse_covariance_matrix[k][l], rtf[k][l]) / (np.dot(np.dot(rtf[k][l].T, inverse_covariance_matrix[k][l]), rtf[k][l]))
   
    return mvdr_beamformer

def calculate_Multi_channel_Wiener_weigths(rtf, Rn, sig_var, MVDR_weigths):
    # Get the number of wave numbers and timeslots
    num_wave_numbers = rtf.shape[0]
    num_timeslots = rtf.shape[1]

    # Calculate the multi-channel Wiener beamforming weights
    mcw_beamformer = np.zeros(rtf.shape, dtype=np.complex64)
    inverse_covariance_matrix = np.zeros(Rn.shape, dtype=np.complex64)
    for l in range(num_timeslots):
        for k in range(num_wave_numbers):
            # Add a small amount of noise to the diagonal of the covariance matrix
            Rn_reg = Rn[k][l] + np.eye(Rn[k][l].shape[0]) * 1e-6

            # Calculate the beamforming weights using the multi-channel Wiener algorithm
            inverse_covariance_matrix[k][l] = np.linalg.inv(Rn_reg)

            # Single channel Wiener filter
            SC_Winer_w = sig_var[k][l] / (sig_var[k][l] + np.dot(np.dot(rtf[k][l].T, inverse_covariance_matrix[k][l]), rtf[k][l]))

            # Multi channel Wiener filter
            mcw_beamformer[k][l] = SC_Winer_w * MVDR_weigths[k][l]
    
    return mcw_beamformer