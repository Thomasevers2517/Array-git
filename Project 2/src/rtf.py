import numpy as np
from scipy.signal import stft, istft
from scipy.linalg import sqrtm
import time

# STFT specs
N_PER_SEG = 400
N_OVERLAP = 200
SAMPLE_FREQ = 16000
STFT_WINDOW = 'hann'
STFT_TIME = N_PER_SEG/SAMPLE_FREQ
STFT_OVERLAP = STFT_TIME/2

def determine_rtf_a_priori_CPSD(num_sources, num_mics, mic_signals, path_signals, ref_mic_idx=0, fs=16000, nperseg=N_PER_SEG, noverlap=N_OVERLAP, alpha=0.8):
    print("Determining rtf with a priori information")
    print("alpha: ", alpha)
    
    # Calculate the STFT of the path_signals from the reference microphone to the target source
    _, _, Zxx_ref = stft(path_signals[4][ref_mic_idx], fs, window=STFT_WINDOW, nperseg=nperseg, noverlap=noverlap)

    # Find the number of wave numbers and time samples
    num_wave_numbers = Zxx_ref.shape[0]
    num_time_samples = Zxx_ref.shape[1]

    # Initialize the rtf array
    rtf = np.zeros((num_wave_numbers, num_time_samples, num_mics), dtype=np.complex64)
    signal_variance = np.zeros((num_wave_numbers, num_time_samples, num_mics), dtype=np.complex64)

    # Initialize the STFT array for the microphone signals
    stft_mic_signals = np.array([np.zeros_like(Zxx_ref, dtype=np.complex64)] * num_mics)
    stft_mic_noise = np.array([np.zeros_like(Zxx_ref, dtype=np.complex64)] * num_mics)
    
    # Calculate the STFT for each microphone without the source signal
    for source_idx in range(num_sources-1): # Target is at the last source
        for mic_idx in range(num_mics):
            # Calculate the STFT of the path_signals from the this microphone to only noise
            _, _, Zxx_mic = stft(path_signals[source_idx][mic_idx], fs, window=STFT_WINDOW, nperseg=nperseg, noverlap=noverlap)
            stft_mic_noise[mic_idx] += Zxx_mic
    
    # Calculate the STFT for each microphone
    for mic_idx in range(num_mics):
        # Calculate the STFT of the path_signals from the this microphone
        _, _, Zxx_mic = stft(mic_signals[mic_idx], fs, window=STFT_WINDOW, nperseg=nperseg, noverlap=noverlap)
        stft_mic_signals[mic_idx] = Zxx_mic

    # Define the cross power spectal density matrix Rx and Rn
    Rx = np.zeros((num_wave_numbers, num_time_samples, num_mics, num_mics), dtype=np.complex64)
    Rn = np.zeros((num_wave_numbers, num_time_samples, num_mics, num_mics), dtype=np.complex64)
    Rs = np.zeros((num_wave_numbers, num_time_samples, num_mics, num_mics), dtype=np.complex64)

    # Rx and Rn per source
    Rx_per_source = np.zeros((num_wave_numbers, num_time_samples, num_sources, num_mics, num_mics), dtype=np.complex64)
    Rn_per_source = np.zeros((num_wave_numbers, num_time_samples, num_sources, num_mics, num_mics), dtype=np.complex64)

    # Calculate Cross Power Spectral Density Matrix Rx and Rn
    print("Calculating Rx and Rn... may take a while")
    start_time = time.time()
    for l in range(num_time_samples):
        for k in range(num_wave_numbers):
            if l == 0:
                Rx[k][l] = np.outer(stft_mic_signals[:, k, l], np.conj(stft_mic_signals[:, k, l]))
                Rn[k][l] = 1e-10 * np.eye(num_mics)
            else:
                # Calculate the cross power spectral density matrix Rx and Rn per source
                Rx[k][l] = Rx[k][l-1]*alpha + np.outer(stft_mic_signals[:, k, l], np.conj(stft_mic_signals[:, k, l]))*(1 - alpha)
                Rn[k][l] = Rn[k][l-1]*alpha + np.outer(stft_mic_noise[:, k, l], np.conj(stft_mic_noise[:, k, l]))*(1 - alpha)

            # And know subtract Rn from Rx to get Rs
            Rs[k][l] = Rx[k][l] - Rn[k][l]

            # Compute the EVD of the covariance matrix Rx and Rn
            w_s, V_s = np.linalg.eig(Rs[k][l])

            # Find the biggest eigenvalue of the covariance matrix Rx
            max_eigenvalue_idx = np.argmax(w_s)
            
            # Use the biggest eigenvalue to find the corresponding eigenvector
            max_eigenvalue_vector = V_s[:][max_eigenvalue_idx]

            # Now calculate the rtf from the principal eigenvector of Rx for one source
            rtf[k][l] = max_eigenvalue_vector / max_eigenvalue_vector[0]

            # Calculate the signal variance from the covariance matrix Rs
            signal_variance[k][l] = np.diag(Rs[k][l])

        # Print the progress every 10% of the total number of time samples
        if l % (num_time_samples // 10) == 0:
            print(f"Progress: {l / num_time_samples * 100:.0f}%")
    
    # Measure time taken to calculate Rs
    end_time = time.time()
    print(f"Time taken to calculate Rs: {end_time - start_time:.2f} seconds, for {num_wave_numbers * num_time_samples * 16 * num_sources} covariance and eigenvalue calculations")

    return rtf, Rx, Rn, signal_variance, stft_mic_signals

def estimate_rtf_prewhiten(num_mics, mic_signals, path_signals, ref_mic_idx=0, fs=16000, nperseg=N_PER_SEG, noverlap=N_OVERLAP, alpha=0.8, det_threshold=0.8, pre_cheat_time = 30):
    
    print("Determining rtf using pre-whiting method...")
    print("Pre-cheat time: ", pre_cheat_time*STFT_TIME, " seconds")
    print("detection threshold: ", det_threshold)
    print("alpha: ", alpha)
    
    # Calculate the STFT of the path_signals from the reference microphone to the target source
    _, _, Zxx_ref = stft(mic_signals[ref_mic_idx], fs, window=STFT_WINDOW, nperseg=nperseg, noverlap=noverlap)

    # Find the number of wave numbers and time samples
    num_wave_numbers = Zxx_ref.shape[0]
    num_time_samples = Zxx_ref.shape[1]

    # Initialize the rtf array
    rtf = np.zeros((num_wave_numbers, num_time_samples, num_mics), dtype=np.complex64)
    rtf = np.zeros((num_wave_numbers, num_time_samples, num_mics), dtype=np.complex64)
    signal_variance = np.zeros((num_wave_numbers, num_time_samples, num_mics), dtype=np.complex64)

    # Initialize the STFT array for the microphone signals
    stft_mic_signals = np.array([np.zeros_like(Zxx_ref, dtype=np.complex64)] * num_mics)
    stft_mic_noise = np.array([np.zeros_like(Zxx_ref, dtype=np.complex64)] * num_mics)
    
    # Calculate the STFT for each microphone without the source signal
    for source_idx in range(3):
        for mic_idx in range(num_mics):
            # Calculate the STFT of the path_signals from the this microphone to only noise
            _, _, Zxx_mic = stft(path_signals[source_idx][mic_idx], fs, window=STFT_WINDOW, nperseg=nperseg, noverlap=noverlap)
            stft_mic_noise[mic_idx] += Zxx_mic
    
    # Calculate the STFT for each microphone
    for mic_idx in range(num_mics):
        # Calculate the STFT of the path_signals from the this microphone
        _, _, Zxx_mic = stft(mic_signals[mic_idx], fs, window=STFT_WINDOW, nperseg=nperseg, noverlap=noverlap)
        stft_mic_signals[mic_idx] = Zxx_mic
    
    # Define the cross power spectal density matrix Rx and Rn
    Rx = np.zeros((num_wave_numbers, num_time_samples, num_mics, num_mics), dtype=np.complex64)
    Rn = np.zeros((num_wave_numbers, num_time_samples, num_mics, num_mics), dtype=np.complex64)
    xxH = np.zeros((num_mics, num_mics), dtype=np.complex64)
    
    # Calculate Cross Power Spectral Density Matrix Rx of the data
    print("Estimating Rx and Rn... may take a while")
    
    # Measure time taken to calculate Rs
    start_time = time.time()
    for l in range(num_time_samples):
        for k in range(num_wave_numbers):
            # Compute CPSD for each pair of microphones with a small pre-cheat time
            for mic_id1 in range(num_mics):
                for mic_id2 in range(num_mics):
                    xxH[mic_id1, mic_id2] = stft_mic_signals[mic_id1, k, l] * stft_mic_signals[mic_id2, k, l]
                    if l == 0:
                        Rx[k][l][mic_id1, mic_id2] = xxH[mic_id1, mic_id2]
                        Rn[k][l][mic_id1, mic_id2] = stft_mic_noise[mic_id1, k, l] * stft_mic_noise[mic_id2, k, l]
                    else:
                        Rx[k][l][mic_id1, mic_id2] = Rx[k][l-1][mic_id1, mic_id2]*alpha  + xxH[mic_id1, mic_id2]*(1 - alpha)
                        if l < pre_cheat_time:
                            # Pre-cheat the noise covariance matrix, to know what our target must be
                            Rn[k][l][mic_id1, mic_id2] = Rn[k][l-1][mic_id1, mic_id2]*alpha + (stft_mic_noise[mic_id1, k, l] * stft_mic_noise[mic_id2, k, l])*(1 - alpha)
            
            if l >= pre_cheat_time:
                # Compute the EVD of the covariance matrix Rx 
                w_x, V_x = np.linalg.eig(xxH)

                # Sort eigenvalues and eigenvectors in descending order
                sorted_indices = np.argsort(w_x)[::-1]
                eigenvalues = w_x[sorted_indices]
                eigenvectors = V_x[:, sorted_indices]
                eigenvalues = eigenvalues[:1]

                # and Rx[l-1]
                w_x, V_x = np.linalg.eig(Rx[k][l-1])

                # Sort eigenvalues and eigenvectors in descending order
                sorted_indices = np.argsort(w_x)[::-1]
                eigenvalues_prev = w_x[sorted_indices]
                eigenvectors_prev = V_x[:, sorted_indices]
                
                # Find the eigenvector that correlates most with our current measurement
                # Calculate the correlation coefficients
                correlation_coef = np.zeros(eigenvectors.shape[1])

                for i in range(eigenvectors.shape[1]):
                    # # Correlate the first eigenvector with each of the previous eigenvectors (real part)
                    # corr_real = np.corrcoef(eigenvectors[:, 0].real, eigenvectors_prev[:, i].real)[0, 1]
                    # # Correlate the first eigenvector with each of the previous eigenvectors (imaginary part)
                    # corr_imag = np.corrcoef(eigenvectors[:, 0].imag, eigenvectors_prev[:, i].imag)[0, 1]
                    # # Combine the correlations (average)
                    # correlation_coef[i] = (corr_real + corr_imag) / 2
                    correlation_coef[i] = np.abs(np.corrcoef(eigenvectors[:, 0], eigenvectors_prev[:, i])[0, 1])
                eigenvectors = eigenvectors[:, 0]
                eigenvectors_prev = eigenvectors_prev[:, np.argmax(correlation_coef)]

                # Normalize and dot the eigenvecotrs
                vector_dot_product = np.vdot(eigenvectors, eigenvectors_prev) / (np.linalg.norm(eigenvectors) * np.linalg.norm(eigenvectors_prev))
                corr_score = correlation_coef[np.argmax(correlation_coef)]

                # Decide on the detection threshold to update the noise space
                if corr_score > det_threshold and np.abs(vector_dot_product) > det_threshold: # find a way to see how accurate this is with the actually precense of the source
                    Rn[k][l] = Rn[k][l-1]*alpha + (eigenvectors * np.diag(eigenvalues) * eigenvectors.conj().T)*(1-alpha)
                else:
                    Rn[k][l] = Rn[k][l-1]

            # Regularization of Rn
            Rn[k][l] = Rn[k][l] + np.eye(Rx[k][l].shape[0]) * 1e-10 # TODO, is this needed?

            # Compute the hermitian square root of Rn^(1/2)
            Rn_sqrt = sqrtm(Rn[k][l])
            # print("Rn_sqrt: ", Rn_sqrt)
            # print("np.linalg.inv(Rn_sqrt): ", np.linalg.inv(Rn_sqrt))
            # print("stft_mic_signals[:, k, l].shape: ", stft_mic_signals[:, k, l].shape)
            # print("np.linalg.inv(Rn_sqrt).shape: ", np.linalg.inv(Rn_sqrt).shape)
            
            # Prewhiten the data
            x_tilde = stft_mic_signals[:, k, l] @ np.linalg.inv(Rn_sqrt)

            # Calculate the cross power spectral density matrix Rx_tilde
            Rx_tilde = np.zeros((num_mics, num_mics), dtype=np.complex64)
            for mic_id1 in range(num_mics):
                for mic_id2 in range(num_mics):
                    Rx_tilde[mic_id1, mic_id2] = x_tilde[mic_id1] * x_tilde[mic_id2].conj().T

            # Compute the EVD
            w_tilde, V_tilde = np.linalg.eig(Rx_tilde)
            
            # Sort eigenvalues and eigenvectors in descending order
            sorted_indices = np.argsort(w_tilde)[::-1]
            w_tilde = w_tilde[sorted_indices]
            V_tilde = V_tilde[:, sorted_indices]

            # Truncate M-r smallest eigenvalues
            w_tilde = w_tilde[:1]
            V_tilde = V_tilde[:, :1]
            w_tilde = np.diag(w_tilde)

            # Estimate the covariance matrix of the signal matirx Rs_tilde_hat
            Rs_tilde_hat = V_tilde @ w_tilde @ V_tilde.conj().T

            # De-whiten Rs_tilde_hat to obtain Rs_hat
            Rs_hat = Rn_sqrt @ Rs_tilde_hat @  Rn_sqrt

            # Compute the EVD of Rs_hat, or use the use Rn_sqrt U^tilde_1
            w_hat, V_hat = np.linalg.eig(Rs_hat)
            
             # Find the biggest eigenvalue of the covariance matrix Rx
            max_eigenvalue_idx = np.argmax(w_hat)
            
            # Use the biggest eigenvalue to find the corresponding eigenvector
            max_eigenvalue_vector = V_hat[:][max_eigenvalue_idx]

            # Now calculate the rtf from the principal eigenvector of Rx for one source
            rtf[k][l] = max_eigenvalue_vector
            # rtf[k][l] = V_hat[:][-1]
            
            # Create the relative transfer function from the first microphoneW
            #rtf[k][l] = rtf[k][l] / rtf[k][l][0]
            #print("rtf[k][l]: ", rtf[k][l])

            # # print("Rs_hat: ", Rs_hat)
            # # print("np.diag(Rs_hat): ", np.diag(Rs_hat))
            # Calculate the signal variance by sutracing Rn from Rx and Rn
            signal_variance[k][l] = np.diag(Rs_hat)
            # print(" signal_variance[k][l]: ",  signal_variance[k][l])
            # input("Enter to continue")

        # Print the progress every 10% of the total number of time samples
        if l % (num_time_samples // 10) == 0:
            print(f"Progress: {l / num_time_samples * 100:.0f}%")
    
    # Measure time taken to calculate Rs
    end_time = time.time()
    print(f"Time taken to calculate Rs: {end_time - start_time:.2f} seconds, for {num_wave_numbers * num_time_samples * 16} covariance and eigenvalue calculations")

    return rtf, Rx, Rn, signal_variance, stft_mic_signals

# Not used ####################################################################

def infer_geometry_ULA(impulse_responses, sample_rate=16000, speed_of_sound=343.0):
    num_sources = impulse_responses.shape[0]
    num_mics = impulse_responses.shape[1]
    geometry = np.zeros((num_sources + num_mics, 2))

    # Calculate microphone coordinates
    frequency = sample_rate//2  # Assume the frequency is the Nyquist frequency
    wavelength = speed_of_sound / frequency  # Calculate the wavelength of the sound
    for mic_idx in range(num_mics):
        mic_x = mic_idx * wavelength / 2  # Calculate the x-coordinate based on the microphone index and the wavelength
        mic_y = 0  # Assume the microphone is located at y=0
        geometry[num_sources + mic_idx] = [mic_x, mic_y]

    mic_x = 0
    mic_y = 0

    # Calculate source coordinates
    for source_idx in range(num_sources):
        for mic_idx in range(1, num_mics):  # Start from the second microphone
            impulse_response_1 = impulse_responses[source_idx, 0]
            impulse_response_2 = impulse_responses[source_idx, mic_idx]

            # Find the peak index and magnitude in the impulse responses
            peak_index_1 = np.argmax(impulse_response_1)
            peak_index_2 = np.argmax(impulse_response_2)
            peak_mag_1 = impulse_response_1[peak_index_1] #+ np.random.normal(0, 0.0001)
            peak_mag_2 = impulse_response_2[peak_index_2] #+ np.random.normal(0, 0.0001)
            if peak_mag_1 == peak_mag_2:
                peak_mag_2 = peak_mag_2 + 0.0001
            # print("peak_index_1: ", peak_index_1)
            # print("peak_index_2: ", peak_index_2)
            # print("peak_mag_1: ", peak_mag_1)
            # print("peak_mag_2: ", peak_mag_2)

            # Calculate the time delay and distance for each microphone and add a noise term
            time_delay_1 = peak_index_1 / sample_rate + np.random.normal(0, 0.00001)
            time_delay_2 = peak_index_2 / sample_rate + np.random.normal(0, 0.00001)
            if time_delay_1 == time_delay_2:
                time_delay_2 = time_delay_2 + 0.0001
            # print("time_delay_1: ", time_delay_1)
            # print("time_delay_2: ", time_delay_2)

            # Calculate the distance traveled by sound for each microphone
            distance_1 = time_delay_1 * speed_of_sound
            distance_2 = time_delay_2 * speed_of_sound
            # print("distance_1: ", distance_1)
            # print("distance_2: ", distance_2)

            # Calculate the difference in time delay and distance between the two microphones
            difference_time_delay = time_delay_2 - time_delay_1 # Calculate the time delay difference
            difference_distance = difference_time_delay * speed_of_sound  # Calculate the difference in distance
            # print("difference_time_delay: ", difference_time_delay)
            # print("difference_distance: ", difference_distance)
            
            # Use trigonometry to calculate the x and y coordinates
            angle_mic_1 = np.arccos(((wavelength/2)**2 + peak_mag_1**2 - peak_mag_2**2) / (2 * (wavelength/2) * peak_mag_1))
            angle_mic_2 = np.arccos(((wavelength/2)**2 + peak_mag_2**2 - peak_mag_1**2) / (2 * (wavelength/2) * peak_mag_2))
            angle_mic = np.arcsin(difference_distance/(wavelength/2))
            # print("angle_mic_1: ", angle_mic_1)
            # print("angle_mic_2: ", angle_mic_2)
            # print("angle_mic: ", angle_mic)

            # Calculate the x and y coordinates of the source
            source_x = distance_1 * np.cos(angle_mic_1) + mic_idx * wavelength / 2
            source_y = distance_1 * np.sin(angle_mic_1)
            geometry[source_idx] = [source_x, source_y]
            # print("geometry[source_idx]: ", geometry[source_idx])

    return geometry