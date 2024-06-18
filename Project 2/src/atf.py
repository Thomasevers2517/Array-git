import numpy as np
from scipy.signal import stft
import matplotlib.pyplot as plt
import time

N_PER_SEG = 512
N_OVERLAP = 256

def determine_atf_a_priori(num_sources, num_mics, path_signals, ref_mic_idx=0, fs=16000, nperseg=N_PER_SEG, noverlap=N_OVERLAP):
    print("Determining ATF with a priori information")
    # Calculate the STFT of the path_signals from the reference microphone to the target source
    _, _, Zxx_ref = stft(path_signals[4][ref_mic_idx], fs, nperseg=nperseg, noverlap=noverlap)

    # Find the number of wave numbers and time samples
    num_wave_numbers = Zxx_ref.shape[0]
    num_time_samples = Zxx_ref.shape[1]

    # Initialize the ATF array
    atf = np.zeros((num_wave_numbers, num_time_samples, num_mics), dtype=np.complex64)

    # # plot the magnitude of the STFT
    # plt.figure()
    # plt.pcolormesh(np.abs(Zxx_ref))
    # plt.title(f'Magnitude of STFT of Microphone {ref_mic_idx}')
    # plt.colorbar()
    # plt.show()

    # Initialize the STFT array for the path signals
    stft_path_signals = np.array([[np.zeros_like(Zxx_ref, dtype=np.complex64)] * num_mics] * num_sources)
    stft_mic_signals = np.array([np.zeros_like(Zxx_ref, dtype=np.complex64)] * num_mics)

    # Calculate the STFT for each microphone
    for mic_idx in range(num_mics):
        for source_idx in range(num_sources):
            # Calculate the STFT of the path_signals from the this microphone to the target source
            _, _, Zxx_mic = stft(path_signals[source_idx][mic_idx], fs, nperseg=nperseg, noverlap=noverlap)

            # Calculate the RTF
            stft_path_signals[source_idx][mic_idx] = Zxx_mic
        # Calculate the STFT of the microphone signals as the sum of the STFT of the path signals
        stft_mic_signals[mic_idx] = np.sum(stft_path_signals[:, mic_idx], axis=0)

    # Calculate the variance of the microphone signals from source 4
    signal_variance = np.var(stft_path_signals[4], axis=0)

    # # Initialize the covariance matrix Rs
    # Rs = np.zeros((num_wave_numbers, num_time_samples, num_mics, num_mics), dtype=np.complex64)

    # # Calculate Cross Power Spectral Density Matrix Rs
    # print("Calculating Rs... may take a while")
    # # Measure time taken to calculate Rs
    # start_time = time.time()
    # for k in range(num_wave_numbers):
    #     for l in range(num_time_samples):
    #         # Calculate the covariance matrix Rs for one source
    #         Rs[k, l] = np.cov(stft_path_signals[4][:, k, l])

    #         # Compute the EVD
    #         w, V = np.linalg.eig(Rs[k][l])

    #         # Now calculate the ATF from the principal eigenvector of Rs for one source
    #         atf[k][l] = V[:][-1]
        
    #     # Print the progress every 10% of the total number of wave numbers and time samples
    #     if k % (num_wave_numbers // 10) == 0:
    #         print(f"Progress: {k / num_wave_numbers * 100:.0f}%")
    # # Measure time taken to calculate Rs
    # end_time = time.time()
    # print(f"Time taken to calculate Rs: {end_time - start_time:.2f} seconds, for {num_wave_numbers * num_time_samples * 16} covariance and eigenvalue calculations")

    # Define the cross power spectal density matrix Rx and Rn
    Rx = np.zeros((num_wave_numbers, num_time_samples, num_mics, num_mics), dtype=np.complex64)
    Rn = np.zeros((num_wave_numbers, num_time_samples, num_mics, num_mics), dtype=np.complex64)

    # Rx and Rn per source
    Rx_per_source = np.zeros((num_wave_numbers, num_time_samples, num_sources, num_mics, num_mics), dtype=np.complex64)
    Rn_per_source = np.zeros((num_wave_numbers, num_time_samples, num_sources, num_mics, num_mics), dtype=np.complex64)

    # Set the alpha value (filter lag) TODO: create a larger filter size than 1 for smoother filtering
    alpha = 0.99

    # Calculate Cross Power Spectral Density Matrix Rx and Rn
    print("Calculating Rx and Rn... may take a while")
    # Measure time taken to calculate Rs
    start_time = time.time()
    for k in range(num_wave_numbers):
        for source_idx in range(num_sources):
            # Calculate the covariance matrix Rx for one source
            
            # Calculate the cross power spectral density matrix Rn
            # We know that our target source is at index 4
            if source_idx == 4:
                Rn_per_source[k][0][source_idx] = 0
            else:
                Rn_per_source[k][0][source_idx] = np.cov(stft_path_signals[source_idx][:, k, 0])
        
        # Convert the correleation matrix per source to a matrix of the sum of the correlation matrices
        Rx[k][0] = np.sum(Rx_per_source[k][0][:,], axis=0)
        Rn[k][0] = np.sum(Rn_per_source[k][0][:,], axis=0)

        # Compute the EVD of the covariance matrix Rx and Rn
        w_x, V_x = np.linalg.eig(Rx[k][0])
        w_n, V_n = np.linalg.eig(Rn[k][0])
        # print("V_x: ", V_x)
        # print("w_x: ", w_x)
        
        # Find the biggest eigenvalue of the covariance matrix Rx
        max_eigenvalue = np.max(w_x)
        max_eigenvalue_idx = np.argmax(w_x)
        # print("max_eigenvalue: ", max_eigenvalue)
        
        # Use the biggest eigenvalue to find the corresponding eigenvector
        max_eigenvalue_vector = V_x[:][max_eigenvalue_idx]

        # print("max_eigenvalue_vector: ", max_eigenvalue_vector)
        # print("V_x[:][-1]: ",  V_x[:][-1])
        # input("Press Enter to continue...")

        # Now calculate the ATF from the principal eigenvector of Rx for one source
        atf[k][0] = max_eigenvalue_vector
        # atf[k][0] = V_x[:][-1]
        
    for k in range(num_wave_numbers):
        for l in range(num_time_samples):
            for source_idx in range(num_sources):
                Rx_per_source[k][l][source_idx] = Rx_per_source[k][l-1][source_idx]*alpha + np.cov(stft_path_signals[source_idx][:, k, l])*(1 - alpha)

                # Calculate the cross power spectral density matrix Rn
                # We know that our target source is at index 4
                if source_idx == 4:
                    Rn_per_source[k][l][source_idx] = Rn_per_source[k][l-1][source_idx]
                else:
                    Rn_per_source[k][l][source_idx] = Rn_per_source[k][l-1][source_idx]*alpha + np.cov(stft_path_signals[source_idx][:, k, l])*(1 - alpha)

            # Convert the correleation matrix per source to a matrix of the sum of the correlation matrices
            Rx[k][l] = np.sum(Rx_per_source[k][0][:,], axis=0)
            Rn[k][l] = np.sum(Rn_per_source[k][0][:,], axis=0)

            # Compute the EVD of the covariance matrix Rx and Rn
            w_x, V_x = np.linalg.eig(Rx[k][l])
            w_n, V_n = np.linalg.eig(Rn[k][l])

            # Find the biggest eigenvalue of the covariance matrix Rx
            max_eigenvalue = np.max(w_x)
            max_eigenvalue_idx = np.argmax(w_x)
            # print("max_eigenvalue: ", max_eigenvalue)
            
            # Use the biggest eigenvalue to find the corresponding eigenvector
            max_eigenvalue_vector = V_x[:][max_eigenvalue_idx]

            # Now calculate the ATF from the principal eigenvector of Rx for one source
            atf[k][l] = max_eigenvalue_vector # TODO: check if this is the correct way to calculate the ATF, with eigenvector is needed?
            #atf[k][l] = V_x[:][-1]

        # Print the progress every 10% of the total number of wave numbers and time samples
        if k % (num_wave_numbers // 10) == 0:
            print(f"Progress: {k / num_wave_numbers * 100:.0f}%")
    
    # Measure time taken to calculate Rs
    end_time = time.time()
    print(f"Time taken to calculate Rs: {end_time - start_time:.2f} seconds, for {num_wave_numbers * num_time_samples * 16 * num_sources} covariance and eigenvalue calculations")

    # # Check if Rx = Rs + Rn
    # if np.allclose(Rx, Rs + Rn):
    #     print("Rx = Rs + Rn")
    # else:
    #     print("RX is not the same as Rs + Rn??? Rx != Rs + Rn")

    return atf, Rx, Rn, signal_variance, stft_mic_signals

def estimate_rtf_prewhiten(num_mics, microphone_signals, ref_mic_idx=0, fs=16000, nperseg=N_PER_SEG, noverlap=N_OVERLAP):
    
    print("Determining RTF using pre-whiting method...")
    # Calculate the STFT of the path_signals from the reference microphone to the target source
    _, _, Zxx_ref = stft(microphone_signals[ref_mic_idx], fs, nperseg=nperseg, noverlap=noverlap)

    # Find the number of wave numbers and time samples
    num_wave_numbers = Zxx_ref.shape[0]
    num_time_samples = Zxx_ref.shape[1]

    # Initialize the ATF array
    rtf = np.zeros((num_wave_numbers, num_time_samples, num_mics), dtype=np.complex64)
    signal_variance = np.zeros((num_wave_numbers, num_time_samples, num_mics), dtype=np.complex64)

    # Initialize the STFT array for the microphone signals
    stft_mic_signals = np.array([np.zeros_like(Zxx_ref, dtype=np.complex64)] * num_mics)

    # Calculate the STFT for each microphone
    for mic_idx in range(num_mics):
        # Calculate the STFT of the path_signals from the this microphone to the target source
        _, _, Zxx_mic = stft(microphone_signals[mic_idx], fs, nperseg=nperseg, noverlap=noverlap)

        # Calculate the RTF
        stft_mic_signals[mic_idx] = Zxx_mic
    
    # Define the cross power spectal density matrix Rx and Rn
    Rx = np.zeros((num_wave_numbers, num_time_samples, num_mics, num_mics), dtype=np.complex64)
    Rn = np.zeros((num_wave_numbers, num_time_samples, num_mics, num_mics), dtype=np.complex64)

    # Calculate Cross Power Spectral Density Matrix Rx of the data
    print("Calculating Rx... may take a while")
    
    # Measure time taken to calculate Rs
    start_time = time.time()
    for k in range(num_wave_numbers):
        for l in range(num_time_samples):
            # Compute Rx for the current time sample
            Rx[k][l] = np.cov(stft_mic_signals[:, k, l])

            # Compute the rank of the covariance matrix Rx
            rank = np.linalg.matrix_rank(Rx[k][l])
            # print(f"Rank of Rx: {rank}")

            # Compute the EVD of the covariance matrix Rx
            w_x, V_x = np.linalg.eig(Rx[k][l])
            # print("w_x: ", w_x)
            # print("V_x: ", V_x)

            # Eigenvalue thresholding using MDL criterion
            num_snapshots = 1
            eigenvalues = np.sort(w_x)[::-1] + 1e-42  # Sort in descending order and regularize
            # print("eigenvalues: ", eigenvalues)

            N = len(eigenvalues)
            mdl = []

            for k in range(1, N):
                # print("k:", k)
                likelihood = np.sum(np.log(eigenvalues[k:])) / (N - k)
                penalty = (k * (2 * N - k)) / num_snapshots
                mdl_value = -(N - k) * num_snapshots * np.log(likelihood) + penalty * np.log(num_snapshots)
                mdl.append(mdl_value)
            
            num_sources = np.argmin(mdl) + 1
            # print("num_of_sources: ", num_sources) 

            """
            Estimate the noise covariance matrix Rn.
            Args:
                Rx: Covariance matrix of the observed signal.
                num_sources: Number of sources.
            Returns:
                Rn: Estimated noise covariance matrix.
            """
            eigenvalues, eigenvectors = np.linalg.eigh(Rx[k][l])
            eigenvalues = np.sort(eigenvalues)[::-1]  # Sort in descending order
            noise_eigenvalues = eigenvalues[num_sources:]
            noise_eigenvectors = eigenvectors[:, num_sources:]
            # print("noise_eigenvalues: ", noise_eigenvalues)
            # print("noise_eigenvectors: ", noise_eigenvectors)

            Rn[k][l] = noise_eigenvectors @ np.diag(noise_eigenvalues) @ noise_eigenvectors.T
            
            # Regularization of Rn
            Rn[k][l] = Rn[k][l] + np.eye(Rx[k][l].shape[0]) * 1e-6
            # print("Rn: ", Rn[k][l])
        
            # Partition Rx
            # Rx_hat = V_x[:][-(num_mics-rank):]
            # Rn_hat = V_x[:][:-(num_mics-rank)]
            # print("Rx_hat: ", Rx_hat)
            # print("Rn_hat: ", Rn_hat)
            # input("Press Enter to continue...")

            # Rx_hat = V_x[:][-(num_mics-rank):]

            # Retrieve the Rx and Rn matrices from the subspaces
            # Rx[k][l] = Rx_hat
            #Rn[k][l] = Rn_hat

            # Compute the hermitian square root of Rn^(1/2)
            Rn_sqrt = np.sqrt(Rn[k][l])

            # Prewritten the data
            x_tilde = np.multiply(np.linalg.inv(Rn_sqrt), stft_mic_signals[:, k, l])
            Rx_tilde = np.multiply(x_tilde, x_tilde.conj().T)

            # Compute the EVD
            w_tilde, V_tilde = np.linalg.eig(Rx_tilde)

            # Compute the rank of the covariance matrix Rx
            rank = np.linalg.matrix_rank(Rx[k][l])
            # print(f"Rank of Rx: {rank}")
            
            # Truncate M-r smallest eigenvalues and reduce the remaining ones
            # M = 2
            # w_tilde = w_tilde[-(M - rank):]
            # V_tilde = V_tilde[:][-(M - rank):]
            # print("w_tilde: ", w_tilde)
            # print("V_tilde: ", V_tilde)
            
            # Estimate the covariance matrix of the signal matirx Rs_tilde_hat
            Rs_tilde_hat = np.zeros_like(Rx_tilde, dtype=np.complex64)
            Rs_tilde_hat += V_tilde @ w_tilde @ V_tilde.conj().T

            # De-whiten Rs_tilde_hat to obtain Rs_hat
            Rs_hat = np.linalg.inv(Rn_sqrt) @ Rs_tilde_hat @ np.linalg.inv(Rn_sqrt)

            # Compute the EVD of Rs_hat
            w_hat, V_hat = np.linalg.eig(Rs_hat)

            # Truncate the num_mics - r smallest eigenvalues and reduce the remaining ones
            w_hat = w_hat[-num_mics:]
            V_hat = V_hat[:][-num_mics:]

            # Find the biggest eigenvalue of the covariance matrix Rx
            max_eigenvalue = np.max(w_hat)
            max_eigenvalue_idx = np.argmax(w_hat)
            # print("max_eigenvalue: ", max_eigenvalue)
            
            # Use the biggest eigenvalue to find the corresponding eigenvector
            max_eigenvalue_vector = V_hat[:][max_eigenvalue_idx]

            # Now calculate the ATF from the principal eigenvector of Rx for one source
            rtf[k][l] = max_eigenvalue_vector # TODO: check if this is the correct way to calculate the ATF, with eigenvector is needed?
            #rtf[k][l] = V_hat[:][-1]

            # print("Rs_hat: ", Rs_hat)
            # print("np.diag(Rs_hat): ", np.diag(Rs_hat))

            signal_variance[k][l] = np.diag(Rs_hat)

        # Print the progress every 10% of the total number of wave numbers and time samples
        if k % (num_wave_numbers // 10) == 0:
            print(f"Progress: {k / num_wave_numbers * 100:.0f}%")
    
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