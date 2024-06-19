import numpy as np
from scipy.signal import stft, istft

# STFT specs
N_PER_SEG = 400
N_OVERLAP = 200
SAMPLE_FREQ = 16000
STFT_WINDOW = 'hann'
STFT_TIME = N_PER_SEG/SAMPLE_FREQ
STFT_OVERLAP = STFT_TIME/2


def apply_beamforming_weights(stft_mic_signals, beamforming_weights, fs=16000, nperseg=N_PER_SEG, noverlap=N_OVERLAP):
    # Get the number of microphones
    num_mics = stft_mic_signals.shape[0]

    # Get the number of wave numbers and timeslots
    num_wave_numbers = stft_mic_signals.shape[1]
    num_timeslots = stft_mic_signals.shape[2]

    # Apply the beamforming weights to the microphone signals per wave number and timeslot and sum all microphone signals
    enhanced_signal_freq = np.zeros((num_wave_numbers, num_timeslots), dtype=np.complex64)
    for l in range(num_timeslots):
        for k in range(num_wave_numbers):
            enhanced_signal_freq[k][l] = beamforming_weights[k][l][:].conj() @ stft_mic_signals[:,k,l]

    # Calculate the inverse STFT of the enhanced signal in the frequency domain
    enhanced_signal = np.array(istft(enhanced_signal_freq, fs, nperseg=nperseg, noverlap=noverlap, freq_axis=0, time_axis=1))[1]
   
    return enhanced_signal

def calculate_delay_and_sum_weights(rtf):
    # Get the number of wave numbers and timeslots
    num_wave_numbers = rtf.shape[0]
    num_timeslots = rtf.shape[1]

    # Calculate the delay-and-sum beamforming weights
    delay_and_sum_beamformer = np.zeros(rtf.shape, dtype=np.complex64)
    for l in range(num_timeslots):
        for k in range(num_wave_numbers):
            delay_and_sum_beamformer[k][l] = rtf[k][l] / (rtf[k][l] @ rtf[k][l].conj().T)
            # for mic_idx in range(rtf.shape[2]):
            #     # Calculate the beamforming weights using the delay-and-sum algorithm
            #     delay_and_sum_beamformer[k][l][mic_idx] = rtf[k][l][mic_idx] / np.dot(rtf[k][l][mic_idx].conj().T, rtf[k][l][mic_idx])
            #     # print("l: ", l, " k: ", k, " mic_idx: ", mic_idx)
            #     # print("np.dot(rtf[k][l][mic_idx].conj().T, rtf[k][l][mic_idx]): ", np.dot(rtf[k][l][mic_idx].conj().T, rtf[k][l][mic_idx]))
            #     # input("Press Enter to continue...")

    return delay_and_sum_beamformer

def calculate_mvdr_weights(rtf, Rn):

    # Get the number of wave numbers and timeslots
    num_wave_numbers = rtf.shape[0]
    num_timeslots = rtf.shape[1]

    # Calculate the delay-and-sum beamforming weights
    mvdr_beamformer = np.zeros(rtf.shape, dtype=np.complex64)
    inverse_covariance_matrix = np.zeros(Rn.shape, dtype=np.complex64)
    for l in range(num_timeslots):
        for k in range(num_wave_numbers):
            # Add a small amount of noise to the diagonal of the covariance matrix
            Rn_reg = Rn[k][l] + np.eye(Rn[k][l].shape[0]) * 1e-20 # Regularization term

            # Calculate the beamforming weights using the minimum variance distortionless response (MVDR) algorithm
            inverse_covariance_matrix[k][l] = np.linalg.inv(Rn_reg)

            #mvdr_beamformer[k][l] = np.dot(inverse_covariance_matrix[k][l], rtf[k][l]) / (np.dot(rtf[k][l].conj().T, inverse_covariance_matrix[k][l]) @ rtf[k][l]) # we use here becuase python is row major
            mvdr_beamformer[k][l] = (rtf[k][l] @ inverse_covariance_matrix[k][l].T) / ((rtf[k][l] @ inverse_covariance_matrix[k][l].T).conj() @ rtf[k][l])
            #mvdr_beamformer[k][l] = np.outer(inverse_covariance_matrix[k][l], rtf[k][l]) / ((rtf[k][l] @ inverse_covariance_matrix[k][l].T).conj() @ rtf[k][l])
   
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
            Rn_reg = Rn[k][l] + np.eye(Rn[k][l].shape[0]) * 1e-20

            # Calculate the beamforming weights using the multi-channel Wiener algorithm
            inverse_covariance_matrix[k][l] = np.linalg.inv(Rn_reg)

            # Single channel Wiener filter
            SC_Winer_w = sig_var[k][l] / (sig_var[k][l] + 1/((rtf[k][l] @ inverse_covariance_matrix[k][l].T).conj() @ rtf[k][l]))

            # Multi channel Wiener filter
            mcw_beamformer[k][l] = SC_Winer_w * MVDR_weigths[k][l]
    
    return mcw_beamformer