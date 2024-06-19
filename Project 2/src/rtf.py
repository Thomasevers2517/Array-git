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

def determine_rtf_a_priori_CPSD(mic_signals, path_signals, ref_mic_idx=0, fs=16000, nperseg=N_PER_SEG, noverlap=N_OVERLAP, alpha=0.2, speak_detection_threshold=1e-6):
    print("Determining rtf with a priori information")
    print("alpha: ", alpha)
    
    # Calculate the STFT of the path_signals from the reference microphone to the target source
    _, _, Zxx_ref = stft(path_signals[4][ref_mic_idx], fs, window=STFT_WINDOW, nperseg=nperseg, noverlap=noverlap)

    # Find the number of wave numbers and time samples
    num_wave_numbers = Zxx_ref.shape[0]
    num_time_samples = Zxx_ref.shape[1]

    num_sources = len(path_signals)
    num_mics= len(path_signals[0])
    print("Number of sources: ", num_sources)
    print("Number of microphones: ", num_mics)

    # Initialize the rtf array
    rtf = np.zeros((num_wave_numbers, num_time_samples, num_mics), dtype=np.complex64)
    signal_variance = np.zeros((num_wave_numbers, num_time_samples, num_mics), dtype=np.complex64)

    # Initialize the STFT array for the microphone signals
    stft_mic_signals = np.array([np.zeros_like(Zxx_ref, dtype=np.complex64)] * num_mics)
    stft_mic_noise = np.array([np.zeros_like(Zxx_ref, dtype=np.complex64)] * num_mics)
    stft_mic_source = np.array([np.zeros_like(Zxx_ref, dtype=np.complex64)] * num_mics)

    # Calculate the STFT for each microphone
    for mic_idx in range(num_mics):
        # Calculate the STFT of the path_signals from the this microphone
        _, _, Zxx_mic = stft(mic_signals[mic_idx], fs, window=STFT_WINDOW, nperseg=nperseg, noverlap=noverlap)
        stft_mic_signals[mic_idx] = Zxx_mic

    # Calculate the STFT for each microphone without the source signal
    for mic_idx in range(num_mics):
        for source_idx in range(num_sources-1): # Target is at the last source
            # Calculate the STFT of the path_signals from the this microphone to only noise
            _, _, Zxx_mic = stft(path_signals[source_idx][mic_idx], fs, window=STFT_WINDOW, nperseg=nperseg, noverlap=noverlap)
            stft_mic_noise[mic_idx] += Zxx_mic

    # Calculate the STFT for each microphone ONLY with the source signal
    for mic_idx in range(num_mics):
        # Calculate the STFT of the path_signals from the this microphone to only noise
        _, _, Zxx_mic = stft(path_signals[4][mic_idx], fs, window=STFT_WINDOW, nperseg=nperseg, noverlap=noverlap)
        stft_mic_source[mic_idx] = Zxx_mic

    # Define the cross power spectal density matrix Rx and Rn
    Rx = np.zeros((num_wave_numbers, num_time_samples, num_mics, num_mics), dtype=np.complex64)
    Rn = np.zeros((num_wave_numbers, num_time_samples, num_mics, num_mics), dtype=np.complex64)
    Rs = np.zeros((num_wave_numbers, num_time_samples, num_mics, num_mics), dtype=np.complex64)
    Rs_est = np.zeros((num_wave_numbers, num_time_samples, num_mics, num_mics), dtype=np.complex64)

    xxH = np.zeros((num_mics, num_mics), dtype=np.complex64)
    nnH = np.zeros((num_mics, num_mics), dtype=np.complex64)
    ssH = np.zeros((num_mics, num_mics), dtype=np.complex64)

    # Calculate Cross Power Spectral Density Matrix Rx and Rn
    print("Calculating Rx and Rn... may take a while")
    start_time = time.time()
    for l in range(num_time_samples):
        # Voice detector
        #if np.abs(stft_mic_source[0][:, l].sum()) > speak_detection_threshold:
        #     # print("Detected speech at time sample: ", l)
        #     Rn[k][l] = Rn[k][l-1]
        # else: 
        #     Rn[k][l] = Rn[k][l-1]*alpha + nnH*(1 - alpha)

        for k in range(num_wave_numbers):
            # Compute CPSD for each pair of microphones with a small pre-cheat time
            xxH = np.outer(stft_mic_signals[:, k, l], np.conj(stft_mic_signals[:, k, l]))
            nnH = np.outer(stft_mic_noise[:, k, l], np.conj(stft_mic_noise[:, k, l]))
            ssH = np.outer(stft_mic_source[:, k, l], np.conj(stft_mic_source[:, k, l]))
            if l == 0:
                Rx[k][l] = xxH
                Rn[k][l] = 0
                Rs[k][l] = ssH
            else:
                # Calculate the cross power spectral density matrix Rx and Rn per source
                Rx[k][l] = Rx[k][l-1]*alpha + xxH*(1 - alpha)
                Rn[k][l] = Rn[k][l-1]*alpha + nnH*(1 - alpha)
                Rs[k][l] = ssH

            # Regularization of Rn
            Rn[k][l] = Rn[k][l] + np.eye(Rn[k][l].shape[0]) * 1e-10 # Regularization term
            
            # And know subtract Rn from Rx to get Rs
            Rs_est[k][l] = Rx[k][l] - Rn[k][l]

            # Compute the EVD of the covariance matrix Rx and Rn
            w_s, V_s = np.linalg.eig(Rs_est[k][l])

            # Sort eigenvalues and eigenvectors in descending order
            sorted_indices = np.argsort(w_s)[::-1]
            eigenvectors = V_s[:, sorted_indices]
            max_eigenvalue_vector = eigenvectors[:, 0]
            
            # Calculate the signal variance from the Rs_est matrix
            signal_variance[k][l] = np.trace(Rs_est[k][l]) / num_mics

            # Now calculate the rtf from the principal eigenvector of Rx for one source
            rtf[k][l] = max_eigenvalue_vector
            if(rtf[k][l][0] == 0):
                rtf[k][l][0] = rtf[k][l][0] + 1e-10
            rtf[k][l] = rtf[k][l] / rtf[k][l][0]

        # Print the progress every 10% of the total number of time samples
        if l % (num_time_samples // 10) == 0:
            print(f"Progress: {l / num_time_samples * 100:.0f}%")
    
    # Measure time taken to calculate Rs
    end_time = time.time()
    print(f"Time taken to calculate Rs: {end_time - start_time:.2f} seconds, for {num_wave_numbers * num_time_samples * 16 * num_sources} covariance and eigenvalue calculations")

    return rtf, Rn, signal_variance, stft_mic_signals

def estimate_rtf_prewhiten(mic_signals, path_signals, ref_mic_idx=0, fs=16000, nperseg=N_PER_SEG, noverlap=N_OVERLAP, alpha=0.99, det_threshold=0.05, pre_cheat_time=50, speak_detection_threshold=1e-6):
    
    print("Determining rtf using pre-whiting method...")
    print("Pre-cheat time: ", pre_cheat_time*STFT_TIME, " seconds")
    print("detection threshold: ", det_threshold)
    print("alpha: ", alpha)
    
    num_sources = len(path_signals)
    num_mics= len(path_signals[0])
    print("Number of sources: ", num_sources)
    print("Number of microphones: ", num_mics)

    # Calculate the STFT of the path_signals from the reference microphone to the target source
    _, _, Zxx_ref = stft(mic_signals[ref_mic_idx], fs, window=STFT_WINDOW, nperseg=nperseg, noverlap=noverlap)

    # Find the number of wave numbers and time samples
    num_wave_numbers = Zxx_ref.shape[0]
    num_time_samples = Zxx_ref.shape[1]

    # Initialize the rtf array
    rtf = np.zeros((num_wave_numbers, num_time_samples, num_mics), dtype=np.complex64)
    signal_variance = np.zeros((num_wave_numbers, num_time_samples, num_mics), dtype=np.complex64)

    # Initialize the STFT array for the microphone signals
    stft_mic_signals = np.array([np.zeros_like(Zxx_ref, dtype=np.complex64)] * num_mics)
    stft_mic_noise = np.array([np.zeros_like(Zxx_ref, dtype=np.complex64)] * num_mics)
    stft_mic_source = np.array([np.zeros_like(Zxx_ref, dtype=np.complex64)] * num_mics)

    # Calculate the STFT for each microphone
    for mic_idx in range(num_mics):
        # Calculate the STFT of the path_signals from the this microphone
        _, _, Zxx_mic = stft(mic_signals[mic_idx], fs, window=STFT_WINDOW, nperseg=nperseg, noverlap=noverlap)
        stft_mic_signals[mic_idx] = Zxx_mic
    
    # Calculate the STFT for each microphone without the source signal
    for mic_idx in range(num_mics):
        for source_idx in range(num_sources - 1):
            # Calculate the STFT of the path_signals from the this microphone to only noise
            _, _, Zxx_mic = stft(path_signals[source_idx][mic_idx], fs, window=STFT_WINDOW, nperseg=nperseg, noverlap=noverlap)
            stft_mic_noise[mic_idx] += Zxx_mic
    
    # Calculate the STFT for each microphone ONLY with the source signal
    for mic_idx in range(num_mics):
        # Calculate the STFT of the path_signals from the this microphone to only noise
        _, _, Zxx_mic = stft(path_signals[4][mic_idx], fs, window=STFT_WINDOW, nperseg=nperseg, noverlap=noverlap)
        stft_mic_source[mic_idx] = Zxx_mic
    
    # Define the cross power spectal density matrix Rx and Rn
    Rx = np.zeros((num_wave_numbers, num_time_samples, num_mics, num_mics), dtype=np.complex64)
    Rn = np.zeros((num_wave_numbers, num_time_samples, num_mics, num_mics), dtype=np.complex64)
    Rs = np.zeros((num_wave_numbers, num_time_samples, num_mics, num_mics), dtype=np.complex64)
    xxH = np.zeros((num_mics, num_mics), dtype=np.complex64)
    nnH = np.zeros((num_mics, num_mics), dtype=np.complex64)
    
    # Calculate Cross Power Spectral Density Matrix Rx of the data
    print("Estimating Rx and Rn... may take a while")
    
    source_present_flag = False
    source_present_begin = 0
    source_present_count = 0
    source_detection_begin = 0
    source_det_count = 0
    source_det_FA = 0
    source_det_score = 0

    # Measure time taken to calculate Rs
    start_time = time.time()
    for l in range(num_time_samples):
        if np.abs(stft_mic_source[:, :, l].sum()) > speak_detection_threshold:
            source_present_flag = True
            if(source_present_begin == 0):
                source_present_begin = l
        else:
            source_present_flag = False
        for k in range(num_wave_numbers):
            # Compute CPSD for each pair of microphones with a small pre-cheat time
            xxH = np.outer(stft_mic_signals[:, k, l], np.conj(stft_mic_signals[:, k, l]))
            if l == 0:
                Rx[k][l] = xxH
                Rn[k][l] = 0
            else:
                if l < pre_cheat_time:
                    nnH = np.outer(stft_mic_noise[:, k, l], np.conj(stft_mic_noise[:, k, l]))
                    ssH = np.outer(stft_mic_source[:, k, l], np.conj(stft_mic_source[:, k, l]))
                    # Pre-cheat the noise covariance matrix, to know what our target must be
                    # Check if the source is present
                    #if np.abs(stft_mic_source[:, k, l].sum()) > speak_detection_threshold:
                    Rn[k][l] = Rn[k][l-1]*alpha + nnH*(1 - alpha)
                    Rs[k][l] = ssH
                    Rx[k][l] = Rx[k][l-1]*alpha + xxH*(1 - alpha)

            # Decide on the detection threshold to update the noise space
            if l >= pre_cheat_time:
                # Compute the EVD of the covariance matrix Rx 
                w_x, V_x = np.linalg.eig(xxH)

                # Sort eigenvalues and eigenvectors in descending order
                sorted_indices = np.argsort(w_x)[::-1]
                eigenvectors = V_x[:, sorted_indices]
                max_eigenvectors = eigenvectors[:, 0]

                # # and Rx[l-1] to find the eigenvector that correlates most with our current measurement
                # w_x, V_x = np.linalg.eig(Rx[k][l])

                # # Sort eigenvalues and eigenvectors in descending order
                # sorted_indices = np.argsort(w_x)[::-1]
                # eigenvectors_prev = V_x[:, sorted_indices]

                # # Find the eigenvector that correlates most with our current measurement
                # correlation_coef = np.zeros(eigenvectors.shape[1])

                # for i in range(eigenvectors_prev.shape[1]):
                #     correlation_coef[i] = np.abs(np.corrcoef(max_eigenvectors, eigenvectors_prev[:, i])[0, 1]) # TODO, is this the correct way to correlate complex numbers?
                # best_fitting_eigenvectors_prev = eigenvectors_prev[:, np.argmax(correlation_coef)]

                # # Normalize and dot the eigenvecotrs
                # vector_dot_product = np.vdot(max_eigenvectors, best_fitting_eigenvectors_prev) / (np.linalg.norm(max_eigenvectors) * np.linalg.norm(best_fitting_eigenvectors_prev))
                # corr_score = correlation_coef[np.argmax(correlation_coef)]

                # if corr_score < det_threshold and np.abs(vector_dot_product) < det_threshold: # find a way to see how accurate this is with the actual precense of the source
                #     #Rx[k][l] = Rx[k][l-1]*alpha + xxH*(1 - alpha)
                #     Rn[k][l] = Rn[k][l-1]
                # else:
                #     #Rx[k][l] = Rx[k][l-1]
                #     Rn[k][l] = Rn[k][l-1]*alpha + xxH*(1-alpha)
                
                Rs[k][l] = Rx[k][l-1] - Rn[k][l-1]
                w_s, V_s = np.linalg.eig(Rs[k][l])

                # Sort eigenvalues and eigenvectors in descending order
                sorted_indices = np.argsort(w_s)[::-1]
                eigenvectors_prev = V_s[:, sorted_indices]
                max_eigenvectors_prev = eigenvectors_prev[:, 0]

                # Decide on the detection threshold to update the noise space
                corr_score = np.dot(np.abs(max_eigenvectors), np.abs(max_eigenvectors_prev)) / (np.linalg.norm(max_eigenvectors) * np.linalg.norm(max_eigenvectors_prev))
                
                # Gather statistics on the source detection
                if corr_score > det_threshold:
                    source_det_count += 1
                                
                    if source_detection_begin == 0:
                        source_detection_begin = l
                    if source_present_flag:
                        source_present_count += 1
                        source_det_score += 1
                    else:
                        source_det_FA += 1
                
                if corr_score > det_threshold:
                    Rx[k][l] = Rx[k][l-1]*alpha + xxH*(1 - alpha)
                    Rn[k][l] = Rn[k][l-1]
                else:
                    Rx[k][l] = Rx[k][l-1]
                    Rn[k][l] = Rn[k][l-1]*alpha + xxH*(1 - alpha)

            # Regularization of Rn
            Rn[k][l] = Rn[k][l] + np.eye(Rn[k][l].shape[0]) * 1e-10 # Regularization term

            # Compute the hermitian square root of Rn^(1/2)
            Rn_sqrt = sqrtm(Rn[k][l])

            # Prewhiten the data
            x_tilde = stft_mic_signals[:, k, l] @ np.linalg.inv(Rn_sqrt)

            # Calculate the cross power spectral density matrix Rx_tilde
            Rx_tilde = np.zeros((num_mics, num_mics), dtype=np.complex64)
            Rx_tilde = np.outer(x_tilde, x_tilde.conj())

            # Compute the EVD
            w_tilde, V_tilde = np.linalg.eig(Rx_tilde)
            
            # Sort eigenvalues and eigenvectors in descending order
            sorted_indices = np.argsort(w_tilde)[::-1]
            w_tilde = w_tilde[sorted_indices]
            V_tilde = V_tilde[:, sorted_indices]

            # Truncate M-r smallest eigenvalues
            w_tilde = w_tilde[:1]
            V_tilde = V_tilde[:, 0]
            
            # Calculate the signal variance from the EVD of the Estimated Rs
            Rs_tilde_hat = np.outer(V_tilde * w_tilde, V_tilde.conj())

            # De-whiten Rs_tilde_hat to obtain Rs_hat
            Rs_hat = Rn_sqrt @ Rs_tilde_hat @ Rn_sqrt

            # Compute the EVD of Rs_hat, or use the use Rn_sqrt U^tilde_1
            w_hat, V_hat = np.linalg.eig(Rs_hat)
            
            # Sort eigenvalues and eigenvectors in descending order
            sorted_indices = np.argsort(w_hat)[::-1]
            eigenvectors = V_hat[:, sorted_indices]
            max_eigenvectors = eigenvectors[:, 0]

            # Calculate the signal variance from the Rs_est matrix
            signal_variance[k][l] = np.trace(Rs_hat) / num_mics

            # Now calculate the rtf from the principal eigenvector of Rx for one source
            rtf[k][l] = max_eigenvectors
            if(rtf[k][l][0] == 0):
                rtf[k][l][0] = rtf[k][l][0] + 1e-10
            rtf[k][l] = rtf[k][l] / rtf[k][l][0]

        # Print the progress every 10% of the total number of time samples
        if l % (num_time_samples // 10) == 0:
            print(f"Progress: {l / num_time_samples * 100:.0f}%")
    
    # Measure time taken to calculate Rs
    end_time = time.time()
    print(f"Time taken to calculate Rs: {end_time - start_time:.2f} seconds, for {num_wave_numbers * num_time_samples * 16} covariance and eigenvalue calculations")

    
    print("source_present_begin l : ", source_present_begin)
    print("source_detection_begin l : ", source_detection_begin)
    print("source_present_count (incl every wave number per time sample): ", source_present_count)
    print("source detection count (incl every wave number per time sample): ", source_det_count)
    print("source_det_score: ", source_det_score)
    print("Detection_score: ", source_det_score/source_present_count)
    print("source_det_FA: ", source_det_FA)

    return rtf, Rn, signal_variance, stft_mic_signals

def estimate_rtf_GEVD(mic_signals, path_signals, ref_mic_idx=0, fs=16000, nperseg=N_PER_SEG, noverlap=N_OVERLAP, alpha=0.8, det_threshold=0.9, pre_cheat_time=50, speak_detection_threshold=1e-6):
    print("Determining rtf using GEVD")
    print("Pre-cheat time: ", pre_cheat_time*STFT_TIME, " seconds")
    print("detection threshold: ", det_threshold)
    print("alpha: ", alpha)
    
    num_sources = len(path_signals)
    num_mics= len(path_signals[0])
    print("Number of sources: ", num_sources)
    print("Number of microphones: ", num_mics)

    # Calculate the STFT of the path_signals from the reference microphone to the target source
    _, _, Zxx_ref = stft(mic_signals[ref_mic_idx], fs, window=STFT_WINDOW, nperseg=nperseg, noverlap=noverlap)

    # Find the number of wave numbers and time samples
    num_wave_numbers = Zxx_ref.shape[0]
    num_time_samples = Zxx_ref.shape[1]

    # Initialize the rtf array
    rtf = np.zeros((num_wave_numbers, num_time_samples, num_mics), dtype=np.complex64)
    signal_variance = np.zeros((num_wave_numbers, num_time_samples, num_mics), dtype=np.complex64)

    # Initialize the STFT array for the microphone signals
    stft_mic_signals = np.array([np.zeros_like(Zxx_ref, dtype=np.complex64)] * num_mics)
    stft_mic_noise = np.array([np.zeros_like(Zxx_ref, dtype=np.complex64)] * num_mics)
    stft_mic_source = np.array([np.zeros_like(Zxx_ref, dtype=np.complex64)] * num_mics)

    # Calculate the STFT for each microphone
    for mic_idx in range(num_mics):
        # Calculate the STFT of the path_signals from the this microphone
        _, _, Zxx_mic = stft(mic_signals[mic_idx], fs, window=STFT_WINDOW, nperseg=nperseg, noverlap=noverlap)
        stft_mic_signals[mic_idx] = Zxx_mic
    
    # Calculate the STFT for each microphone without the source signal
    for mic_idx in range(num_mics):
        for source_idx in range(num_sources - 1):
            # Calculate the STFT of the path_signals from the this microphone to only noise
            _, _, Zxx_mic = stft(path_signals[source_idx][mic_idx], fs, window=STFT_WINDOW, nperseg=nperseg, noverlap=noverlap)
            stft_mic_noise[mic_idx] += Zxx_mic
    
    # Calculate the STFT for each microphone ONLY with the source signal
    for mic_idx in range(num_mics):
        # Calculate the STFT of the path_signals from the this microphone to only noise
        _, _, Zxx_mic = stft(path_signals[4][mic_idx], fs, window=STFT_WINDOW, nperseg=nperseg, noverlap=noverlap)
        stft_mic_source[mic_idx] = Zxx_mic
    
    # Define the cross power spectal density matrix Rx and Rn
    Rx = np.zeros((num_wave_numbers, num_time_samples, num_mics, num_mics), dtype=np.complex64)
    Rn = np.zeros((num_wave_numbers, num_time_samples, num_mics, num_mics), dtype=np.complex64)
    Rs = np.zeros((num_wave_numbers, num_time_samples, num_mics, num_mics), dtype=np.complex64)
    xxH = np.zeros((num_mics, num_mics), dtype=np.complex64)
    nnH = np.zeros((num_mics, num_mics), dtype=np.complex64)
    
    # Calculate Cross Power Spectral Density Matrix Rx of the data
    print("Estimating Rx and Rn... may take a while")
    
    # Measure time taken to calculate Rs
    start_time = time.time()
    for l in range(num_time_samples):
        for k in range(num_wave_numbers):
            # Compute CPSD for each pair of microphones with a small pre-cheat time
            xxH = np.outer(stft_mic_signals[:, k, l], np.conj(stft_mic_signals[:, k, l]))
            nnH = np.outer(stft_mic_noise[:, k, l], np.conj(stft_mic_noise[:, k, l]))
            if l == 0:
                Rx[k][l] = xxH
                Rn[k][l] = 0 #TODO: is this correct?
            else:
                if l < pre_cheat_time:
                    # Pre-cheat the noise covariance matrix, to know what our target must be
                    # Check if the source is present
                    #if np.abs(stft_mic_source[:, k, l].sum()) > speak_detection_threshold:
                    Rn[k][l] = Rn[k][l-1]*alpha + nnH*(1 - alpha)
                    Rx[k][l] = Rx[k][l-1]*alpha + xxH*(1 - alpha)

            # Decide on the detection threshold to update the noise space
            if l >= pre_cheat_time:
                # Compute the EVD of the covariance matrix Rx 
                w_x, V_x = np.linalg.eig(xxH)

                # Sort eigenvalues and eigenvectors in descending order
                sorted_indices = np.argsort(w_x)[::-1]
                eigenvectors = V_x[:, sorted_indices]
                max_eigenvectors = eigenvectors[:, 0]

                # and Rx[l-1] to find the eigenvector that correlates most with our current measurement
                w_x, V_x = np.linalg.eig(Rx[k][l-1])

                # Sort eigenvalues and eigenvectors in descending order
                sorted_indices = np.argsort(w_x)[::-1]
                eigenvectors_prev = V_x[:, sorted_indices]

                # Find the eigenvector that correlates most with our current measurement
                correlation_coef = np.zeros(eigenvectors.shape[1])

                for i in range(eigenvectors_prev.shape[1]):
                    correlation_coef[i] = np.abs(np.corrcoef(max_eigenvectors, eigenvectors_prev[:, i])[0, 1]) # TODO, is this the correct way to correlate complex numbers?
                best_fitting_eigenvectors_prev = eigenvectors_prev[:, np.argmax(correlation_coef)]

                # Normalize and dot the eigenvecotrs
                vector_dot_product = np.vdot(max_eigenvectors, best_fitting_eigenvectors_prev) / (np.linalg.norm(max_eigenvectors) * np.linalg.norm(best_fitting_eigenvectors_prev))
                corr_score = correlation_coef[np.argmax(correlation_coef)]

                if corr_score > det_threshold and np.abs(vector_dot_product) > det_threshold: # find a way to see how accurate this is with the actual precense of the source
                    Rx[k][l] = Rx[k][l-1]*alpha + xxH*(1 - alpha)
                    Rn[k][l] = Rn[k][l-1]
                else:
                    Rx[k][l] = Rx[k][l-1]
                    Rn[k][l] = Rn[k][l-1]*alpha + xxH*(1-alpha)
                
                # Rs_prev = Rx[k][l-1] - Rn[k][l-1]
                # w_s, V_s = np.linalg.eigh(Rs_prev)

                # # Sort eigenvalues and eigenvectors in descending order
                # sorted_indices = np.argsort(w_s)[::-1]
                # eigenvectors_prev = V_s[:, sorted_indices]
                # max_eigenvectors_prev = eigenvectors_prev[:, 0]

                # # Decide on the detection threshold to update the noise space
                # corr_score = np.dot(np.abs(max_eigenvectors), np.abs(max_eigenvectors_prev)) / (np.linalg.norm(max_eigenvectors) * np.linalg.norm(max_eigenvectors_prev))
                # if corr_score > det_threshold:
                #     Rx[k][l] = Rx[k][l-1]*alpha + xxH*(1 - alpha)
                #     Rn[k][l] = Rn[k][l-1]
                # else:
                #     Rx[k][l] = Rx[k][l-1]
                #     Rn[k][l] = Rn[k][l-1]*alpha + xxH*(1 - alpha)

            # Regularization of Rn
            Rn[k][l] = Rn[k][l] + np.eye(Rn[k][l].shape[0]) * 1e-10 # Regularization term

            # Determine Rs from Rx and Rn
            Rs[k][l] = Rx[k][l] - Rn[k][l]

            # Apply GEVD
            w, V = np.linalg.eigh(Rs[k][l])
            sorted_indices = np.argsort(w)[::-1]
            eigenvectors = V[:, sorted_indices]
            max_eigenvectors = eigenvectors[:, 0]

            # Calculate the signal variance from the Rs_est matrix
            signal_variance[k][l] = np.trace(Rs[k][l]) / num_mics

            # Now calculate the rtf from the principal eigenvector of Rx for one source
            rtf[k][l] = max_eigenvectors
            if(rtf[k][l][0] == 0):
                rtf[k][l][0] = rtf[k][l][0] + 1e-10
            rtf[k][l] = rtf[k][l] / rtf[k][l][0]

        # Print the progress every 10% of the total number of time samples
        if l % (num_time_samples // 10) == 0:
            print(f"Progress: {l / num_time_samples * 100:.0f}%")
    
    # Measure time taken to calculate Rs
    end_time = time.time()
    print(f"Time taken to calculate Rs: {end_time - start_time:.2f} seconds, for {num_wave_numbers * num_time_samples * 16} covariance and eigenvalue calculations")

    return rtf, Rn, signal_variance, stft_mic_signals
    
# Not used ####################################################################
def estimate_rtf_prewhiten_working(mic_signals, path_signals, ref_mic_idx=0, fs=16000, nperseg=N_PER_SEG, noverlap=N_OVERLAP, alpha=0.99, det_threshold=0.05, pre_cheat_time=50, speak_detection_threshold=1e-6):
    
    print("Determining rtf using pre-whiting method...")
    print("Pre-cheat time: ", pre_cheat_time*STFT_TIME, " seconds")
    print("detection threshold: ", det_threshold)
    print("alpha: ", alpha)
    
    num_sources = len(path_signals)
    num_mics= len(path_signals[0])
    print("Number of sources: ", num_sources)
    print("Number of microphones: ", num_mics)

    # Calculate the STFT of the path_signals from the reference microphone to the target source
    _, _, Zxx_ref = stft(mic_signals[ref_mic_idx], fs, window=STFT_WINDOW, nperseg=nperseg, noverlap=noverlap)

    # Find the number of wave numbers and time samples
    num_wave_numbers = Zxx_ref.shape[0]
    num_time_samples = Zxx_ref.shape[1]

    # Initialize the rtf array
    rtf = np.zeros((num_wave_numbers, num_time_samples, num_mics), dtype=np.complex64)
    signal_variance = np.zeros((num_wave_numbers, num_time_samples, num_mics), dtype=np.complex64)

    # Initialize the STFT array for the microphone signals
    stft_mic_signals = np.array([np.zeros_like(Zxx_ref, dtype=np.complex64)] * num_mics)
    stft_mic_noise = np.array([np.zeros_like(Zxx_ref, dtype=np.complex64)] * num_mics)
    stft_mic_source = np.array([np.zeros_like(Zxx_ref, dtype=np.complex64)] * num_mics)

    # Calculate the STFT for each microphone
    for mic_idx in range(num_mics):
        # Calculate the STFT of the path_signals from the this microphone
        _, _, Zxx_mic = stft(mic_signals[mic_idx], fs, window=STFT_WINDOW, nperseg=nperseg, noverlap=noverlap)
        stft_mic_signals[mic_idx] = Zxx_mic
    
    # Calculate the STFT for each microphone without the source signal
    for mic_idx in range(num_mics):
        for source_idx in range(num_sources - 1):
            # Calculate the STFT of the path_signals from the this microphone to only noise
            _, _, Zxx_mic = stft(path_signals[source_idx][mic_idx], fs, window=STFT_WINDOW, nperseg=nperseg, noverlap=noverlap)
            stft_mic_noise[mic_idx] += Zxx_mic
    
    # Calculate the STFT for each microphone ONLY with the source signal
    for mic_idx in range(num_mics):
        # Calculate the STFT of the path_signals from the this microphone to only noise
        _, _, Zxx_mic = stft(path_signals[4][mic_idx], fs, window=STFT_WINDOW, nperseg=nperseg, noverlap=noverlap)
        stft_mic_source[mic_idx] = Zxx_mic
    
    # Define the cross power spectal density matrix Rx and Rn
    Rx = np.zeros((num_wave_numbers, num_time_samples, num_mics, num_mics), dtype=np.complex64)
    Rn = np.zeros((num_wave_numbers, num_time_samples, num_mics, num_mics), dtype=np.complex64)
    Rs = np.zeros((num_wave_numbers, num_time_samples, num_mics, num_mics), dtype=np.complex64)
    xxH = np.zeros((num_mics, num_mics), dtype=np.complex64)
    nnH = np.zeros((num_mics, num_mics), dtype=np.complex64)
    
    # Calculate Cross Power Spectral Density Matrix Rx of the data
    print("Estimating Rx and Rn... may take a while")
    
    source_present_flag = False
    source_present_begin = 0
    source_present_count = 0
    source_detection_begin = 0
    source_det_count = 0
    source_det_FA = 0
    source_det_score = 0

    # Measure time taken to calculate Rs
    start_time = time.time()
    for l in range(num_time_samples):
        if np.abs(stft_mic_source[:, :, l].sum()) > speak_detection_threshold:
            source_present_flag = True
            if(source_present_begin == 0):
                source_present_begin = l
        else:
            source_present_flag = False
        for k in range(num_wave_numbers):
            # Compute CPSD for each pair of microphones with a small pre-cheat time
            xxH = np.outer(stft_mic_signals[:, k, l], np.conj(stft_mic_signals[:, k, l]))
            if l == 0:
                Rx[k][l] = xxH
                Rn[k][l] = 0
            else:
                if l < pre_cheat_time:
                    nnH = np.outer(stft_mic_noise[:, k, l], np.conj(stft_mic_noise[:, k, l]))
                    ssH = np.outer(stft_mic_source[:, k, l], np.conj(stft_mic_source[:, k, l]))
                    # Pre-cheat the noise covariance matrix, to know what our target must be
                    # Check if the source is present
                    #if np.abs(stft_mic_source[:, k, l].sum()) > speak_detection_threshold:
                    Rn[k][l] = Rn[k][l-1]*alpha + nnH*(1 - alpha)
                    Rs[k][l] = ssH
                    Rx[k][l] = Rx[k][l-1]*alpha + xxH*(1 - alpha)

            # Decide on the detection threshold to update the noise space
            if l >= pre_cheat_time:
                # Compute the EVD of the covariance matrix Rx 
                w_x, V_x = np.linalg.eig(xxH)

                # Sort eigenvalues and eigenvectors in descending order
                sorted_indices = np.argsort(w_x)[::-1]
                eigenvectors = V_x[:, sorted_indices]
                max_eigenvectors = eigenvectors[:, 0]

                # # and Rx[l-1] to find the eigenvector that correlates most with our current measurement
                # w_x, V_x = np.linalg.eig(Rx[k][l])

                # # Sort eigenvalues and eigenvectors in descending order
                # sorted_indices = np.argsort(w_x)[::-1]
                # eigenvectors_prev = V_x[:, sorted_indices]

                # # Find the eigenvector that correlates most with our current measurement
                # correlation_coef = np.zeros(eigenvectors.shape[1])

                # for i in range(eigenvectors_prev.shape[1]):
                #     correlation_coef[i] = np.abs(np.corrcoef(max_eigenvectors, eigenvectors_prev[:, i])[0, 1]) # TODO, is this the correct way to correlate complex numbers?
                # best_fitting_eigenvectors_prev = eigenvectors_prev[:, np.argmax(correlation_coef)]

                # # Normalize and dot the eigenvecotrs
                # vector_dot_product = np.vdot(max_eigenvectors, best_fitting_eigenvectors_prev) / (np.linalg.norm(max_eigenvectors) * np.linalg.norm(best_fitting_eigenvectors_prev))
                # corr_score = correlation_coef[np.argmax(correlation_coef)]

                # if corr_score < det_threshold and np.abs(vector_dot_product) < det_threshold: # find a way to see how accurate this is with the actual precense of the source
                #     #Rx[k][l] = Rx[k][l-1]*alpha + xxH*(1 - alpha)
                #     Rn[k][l] = Rn[k][l-1]
                # else:
                #     #Rx[k][l] = Rx[k][l-1]
                #     Rn[k][l] = Rn[k][l-1]*alpha + xxH*(1-alpha)
                
                Rs[k][l] = Rx[k][l-1] - Rn[k][l-1]
                w_s, V_s = np.linalg.eig(Rs[k][l])

                # Sort eigenvalues and eigenvectors in descending order
                sorted_indices = np.argsort(w_s)[::-1]
                eigenvectors_prev = V_s[:, sorted_indices]
                max_eigenvectors_prev = eigenvectors_prev[:, 0]

                # Decide on the detection threshold to update the noise space
                corr_score = np.dot(np.abs(max_eigenvectors), np.abs(max_eigenvectors_prev)) / (np.linalg.norm(max_eigenvectors) * np.linalg.norm(max_eigenvectors_prev))
                
                # Gather statistics on the source detection
                if corr_score > det_threshold:
                    source_det_count += 1
                                
                    if source_detection_begin == 0:
                        source_detection_begin = l
                    if source_present_flag:
                        source_present_count += 1
                        source_det_score += 1
                    else:
                        source_det_FA += 1
                
                if corr_score > det_threshold:
                    Rx[k][l] = Rx[k][l-1]*alpha + xxH*(1 - alpha)
                    Rn[k][l] = Rn[k][l-1]
                else:
                    Rx[k][l] = Rx[k][l-1]
                    Rn[k][l] = Rn[k][l-1]*alpha + xxH*(1 - alpha)

            # Regularization of Rn
            Rn[k][l] = Rn[k][l] + np.eye(Rn[k][l].shape[0]) * 1e-10 # Regularization term

            # Compute the hermitian square root of Rn^(1/2)
            Rn_sqrt = sqrtm(Rn[k][l])

            # Prewhiten the data
            x_tilde = stft_mic_signals[:, k, l] @ np.linalg.inv(Rn_sqrt)

            # Calculate the cross power spectral density matrix Rx_tilde
            Rx_tilde = np.zeros((num_mics, num_mics), dtype=np.complex64)
            Rx_tilde = np.outer(x_tilde, x_tilde.conj())

            # Compute the EVD
            w_tilde, V_tilde = np.linalg.eig(Rx_tilde)
            
            # Sort eigenvalues and eigenvectors in descending order
            sorted_indices = np.argsort(w_tilde)[::-1]
            w_tilde = w_tilde[sorted_indices]
            V_tilde = V_tilde[:, sorted_indices]

            # Truncate M-r smallest eigenvalues
            w_tilde = w_tilde[:1]
            V_tilde = V_tilde[:, 0]
            
            # Calculate the signal variance from the EVD of the Estimated Rs
            Rs_tilde_hat = np.outer(V_tilde * w_tilde, V_tilde.conj())

            # De-whiten Rs_tilde_hat to obtain Rs_hat
            Rs_hat = Rn_sqrt @ Rs_tilde_hat @ Rn_sqrt

            # Compute the EVD of Rs_hat, or use the use Rn_sqrt U^tilde_1
            w_hat, V_hat = np.linalg.eig(Rs_hat)
            
            # Sort eigenvalues and eigenvectors in descending order
            sorted_indices = np.argsort(w_hat)[::-1]
            eigenvectors = V_hat[:, sorted_indices]
            max_eigenvectors = eigenvectors[:, 0]

            # Calculate the signal variance from the Rs_est matrix
            signal_variance[k][l] = np.trace(Rs_hat) / num_mics

            # Now calculate the rtf from the principal eigenvector of Rx for one source
            rtf[k][l] = max_eigenvectors
            if(rtf[k][l][0] == 0):
                rtf[k][l][0] = rtf[k][l][0] + 1e-10
            rtf[k][l] = rtf[k][l] / rtf[k][l][0]

        # Print the progress every 10% of the total number of time samples
        if l % (num_time_samples // 10) == 0:
            print(f"Progress: {l / num_time_samples * 100:.0f}%")
    
    # Measure time taken to calculate Rs
    end_time = time.time()
    print(f"Time taken to calculate Rs: {end_time - start_time:.2f} seconds, for {num_wave_numbers * num_time_samples * 16} covariance and eigenvalue calculations")

    
    print("source_present_begin l : ", source_present_begin)
    print("source_detection_begin l : ", source_detection_begin)
    print("source_present_count (incl every wave number per time sample): ", source_present_count)
    print("source detection count (incl every wave number per time sample): ", source_det_count)
    print("source_det_score: ", source_det_score)
    print("Detection_score: ", source_det_score/source_present_count)
    print("source_det_FA: ", source_det_FA)

    return rtf, Rn, signal_variance, stft_mic_signals

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

            # Calculate the time delay and distance for each microphone and add a noise term
            time_delay_1 = peak_index_1 / sample_rate + np.random.normal(0, 0.00001)
            time_delay_2 = peak_index_2 / sample_rate + np.random.normal(0, 0.00001)
            if time_delay_1 == time_delay_2:
                time_delay_2 = time_delay_2 + 0.0001

            # Calculate the distance traveled by sound for each microphone
            distance_1 = time_delay_1 * speed_of_sound
            distance_2 = time_delay_2 * speed_of_sound

            # Calculate the difference in time delay and distance between the two microphones
            difference_time_delay = time_delay_2 - time_delay_1 # Calculate the time delay difference
            difference_distance = difference_time_delay * speed_of_sound  # Calculate the difference in distance
            
            # Use trigonometry to calculate the x and y coordinates
            angle_mic_1 = np.arccos(((wavelength/2)**2 + peak_mag_1**2 - peak_mag_2**2) / (2 * (wavelength/2) * peak_mag_1))
            angle_mic_2 = np.arccos(((wavelength/2)**2 + peak_mag_2**2 - peak_mag_1**2) / (2 * (wavelength/2) * peak_mag_2))
            angle_mic = np.arcsin(difference_distance/(wavelength/2))

            # Calculate the x and y coordinates of the source
            source_x = distance_1 * np.cos(angle_mic_1) + mic_idx * wavelength / 2
            source_y = distance_1 * np.sin(angle_mic_1)
            geometry[source_idx] = [source_x, source_y]

    return geometry