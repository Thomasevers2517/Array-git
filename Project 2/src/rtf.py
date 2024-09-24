import numpy as np
from scipy.signal import stft
from scipy.linalg import sqrtm
from scipy.linalg import eigh
import time
import matplotlib.pyplot as plt

# STFT specs
N_PER_SEG = 400
N_OVERLAP = 200
SAMPLE_FREQ = 16000
STFT_WINDOW = 'hann'
STFT_TIME = N_PER_SEG/SAMPLE_FREQ
STFT_OVERLAP = STFT_TIME/2

num_sources = 5
num_mics = 4

def audio_speech_det(stft_mic_signals, stft_mic_source, plot=False, audio_threshold=13, speech_threshold=5e-3):

    mic_power = np.sum(np.abs(stft_mic_signals[0])**2, axis=0)
    t = np.linspace(0, mic_power.shape[0], mic_power.shape[0])

    percentile_ratio = 0.1
    percentile = np.percentile(mic_power, percentile_ratio, axis=0)
    audio_threshold = percentile 
    # Detect speech presence based on the power threshold
    audio_presence = mic_power > audio_threshold

    # Optionally, visualize the speech detection
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(t, mic_power, label='Mic total Power')
        plt.axhline(y=audio_threshold, color='r', linestyle='--', label=f'percentile {100-percentile_ratio} audio threshold')
        plt.fill_between(t, 0, mic_power, where=audio_presence, facecolor='green', alpha=0.4, label='Audio detected')
        plt.title('Speech Detection based on STFT Power')
        plt.xlabel('Time [s]')
        plt.ylabel('Power')
        plt.legend()
        plt.show()

    # Find and plot the speech threshold per band
    speech_threshold_per_band = np.ones(stft_mic_signals.shape[1])
    for k in range(stft_mic_signals.shape[1]):
        mic_source_power = np.abs(stft_mic_source[0][k])**2

        avg = np.mean(mic_source_power)
        percentile_ratio = 95
        percentile = np.percentile(mic_source_power, percentile_ratio, axis=0)
        per_avg_comb = (avg + percentile) / 2
        speech_threshold_per_band[k] = percentile
        speech_presence = mic_source_power > speech_threshold_per_band[k]

        # Optionally, visualize the speech detection
        if plot:
            figure = plt.figure(figsize=(9, 10))
            plt.plot(t, mic_source_power, label='Mic source total Power')
            # plt.axhline(y=avg, color='r', linestyle='--', label='avg power')
            # plt.axhline(y=percentile, color='b', linestyle='--', label=f'percentile {100-percentile_ratio} power')
            # plt.axhline(y=per_avg_comb, color='g', linestyle='--', label=f'average percentile {100-percentile_ratio} combined power')
            plt.axhline(y=speech_threshold_per_band[k], color='r', linestyle='--', label=f'percentile {100-percentile_ratio} speech threshold')
            plt.fill_between(t, 0, mic_source_power, where=speech_presence, facecolor='green', alpha=0.4, label='Speech detected')
            plt.title('Speech Detection based on STFT Power')
            plt.xlabel('Time [s]')
            plt.ylabel('Power')
            plt.legend()
            figure.canvas.manager.window.move(0,0)
            plt.show()

    return audio_threshold, speech_threshold_per_band


def determine_rtf_a_priori(mic_signals, noise_signal, source_signal, ref_mic_idx=0, fs=16000, nperseg=N_PER_SEG, noverlap=N_OVERLAP, alpha=0.5, estimate_Rs=True):
    print("Determining rtf with a priori information")
    print("alpha: ", alpha)
    
    # Calculate the STFT from the target source to the reference microphone as a template
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
        _, _, Zxx_mic = stft(mic_signals[mic_idx], fs, window=STFT_WINDOW, nperseg=nperseg, noverlap=noverlap)
        stft_mic_signals[mic_idx] = Zxx_mic
    
        _, _, Zxx_mic = stft(noise_signal[mic_idx], fs, window=STFT_WINDOW, nperseg=nperseg, noverlap=noverlap)
        stft_mic_noise[mic_idx] = Zxx_mic
    
        _, _, Zxx_mic = stft(source_signal[mic_idx], fs, window=STFT_WINDOW, nperseg=nperseg, noverlap=noverlap)
        stft_mic_source[mic_idx] = Zxx_mic

    # Define the cross power spectal density matrix Rx and Rn
    Rx = np.zeros((num_wave_numbers, num_time_samples, num_mics, num_mics), dtype=np.complex64)
    Rn = np.zeros((num_wave_numbers, num_time_samples, num_mics, num_mics), dtype=np.complex64)
    Rs = np.zeros((num_wave_numbers, num_time_samples, num_mics, num_mics), dtype=np.complex64)
    Rs_est = np.zeros((num_wave_numbers, num_time_samples, num_mics, num_mics), dtype=np.complex64)

    xxH = np.zeros((num_mics, num_mics), dtype=np.complex64)
    nnH = np.zeros((num_mics, num_mics), dtype=np.complex64)
    ssH = np.zeros((num_mics, num_mics), dtype=np.complex64)

    # Plot the audio signals from the source and from all sound to the microphones
    speech_det_thres_per_band = np.zeros(num_wave_numbers)
    audio_det_thres, speech_det_thres_per_band = audio_speech_det(stft_mic_signals, stft_mic_source, plot=False)

    # Measure powers per time frame
    mic_power = np.sum(np.abs(stft_mic_signals[0][:])**2, axis=0)
    mic_source_power = np.abs(stft_mic_source[0][:])**2

    # Calculate Cross Power Spectral Density Matrix Rx and Rn
    print("Calculating Rx and Rn... may take a while")
    start_time = time.time()
    k=0
    for l in range(num_time_samples):
        # Audio detector on refernce microphone
        if mic_power[l] < audio_det_thres: # Detect non zero volume audio, happens almost never with sources close to the microphones
            rtf[:,l,:] = 1
            Rx[k][l] = 0
            Rn[k][l] = 0
            Rs[k][l] = 0
            signal_variance[k][l] = np.trace(Rs[k][l])/num_mics
            continue
            
        for k in range(num_wave_numbers):
            # Compute CPSD for each pair of microphones with a small pre-cheat time
            xxH = np.outer(stft_mic_signals[:, k, l], np.conj(stft_mic_signals[:, k, l]))
            nnH = np.outer(stft_mic_noise[:, k, l], np.conj(stft_mic_noise[:, k, l]))
            ssH = np.outer(stft_mic_source[:, k, l], np.conj(stft_mic_source[:, k, l]))
            if l == 0:
                Rx[k][l] = xxH
                Rn[k][l] = 0
                Rs[k][l] = ssH # perfect estimate
            else:
                if mic_source_power[k][l] > speech_det_thres_per_band[k]:
                    Rx[k][l] = Rx[k][l-1]*alpha + xxH*(1 - alpha)
                    Rn[k][l] = Rn[k][l-1]
                else:
                    Rx[k][l] = Rx[k][l-1]
                    Rn[k][l] = Rn[k][l-1]*alpha + nnH*(1 - alpha)
                Rs[k][l] = ssH # perfect estimate

            # Regularization of Rn
            Rn[k][l] = Rn[k][l] + np.eye(Rn[k][l].shape[0]) * 1e-10 # Regularization term
            
            # And know subtract Rn from Rx to get Rs
            Rs_est[k][l] = Rx[k][l] - Rn[k][l]
        
            # Compute the EVD of the covariance matrix Rx and Rn
            if estimate_Rs:
                w_s, V_s = np.linalg.eig(Rs_est[k][l])
            else:
                w_s, V_s = np.linalg.eig(Rs[k][l])

            # Sort eigenvalues and eigenvectors in descending order
            sorted_indices = np.argsort(w_s)[::-1]
            eigenvectors = V_s[:, sorted_indices]
            max_eigenvalue_vector = eigenvectors[:, 0]
            
            # Calculate the signal variance from the Rs_est matrix
            if estimate_Rs:
                signal_variance[k][l] = np.trace(Rs_est[k][l]) / num_mics
            else:
                signal_variance[k][l] = np.trace(Rs[k][l]) / num_mics

            # Now calculate the rtf from the principal eigenvector of Rx for one source
            rtf[k][l] = max_eigenvalue_vector
            rtf[k][l] = rtf[k][l] / rtf[k][l][0]

        # Print the progress every 10% of the total number of time samples
        if l % (num_time_samples // 10) == 0:
            print(f"Progress: {l / num_time_samples * 100:.0f}%")
    
    # Measure time taken to calculate Rs
    end_time = time.time()
    print(f"Time taken to calculate Rs: {end_time - start_time:.2f} seconds, for {num_wave_numbers * num_time_samples * 16 * num_sources} covariance and eigenvalue calculations")
    return rtf, Rn, signal_variance, stft_mic_signals

def estimate_rtf_Rs(mic_signals, noise_signal, source_signal, use_GEVD=True, ref_mic_idx=0, fs=16000, nperseg=N_PER_SEG, noverlap=N_OVERLAP, cheat_alpha=0.8, alpha=0.1, det_threshold=0.1, cheat_time=60, eig_val_thr_ratio=200):
    
    print("Estimating rtf...")
    if use_GEVD:
        print("Using GEVD")
    else:
        print("Using prewhitening")
    print("Pre-cheat time: ", cheat_time*STFT_TIME, " seconds")
    print("detection threshold: ", det_threshold)
    print("cheat alpha: ", cheat_alpha)
    print("alpha: ", alpha)
    print("eig_val_thr_ratio: ", eig_val_thr_ratio)

    # Calculate the STFT from the target source to the reference microphone as a template
    _, _, Zxx_ref = stft(mic_signals[ref_mic_idx], fs, window=STFT_WINDOW, nperseg=nperseg, noverlap=noverlap)

    # Find the number of wave numbers and time samples
    num_wave_numbers = Zxx_ref.shape[0]
    num_time_samples = Zxx_ref.shape[1]
    print("Number of wave numbers: ", num_wave_numbers)
    print("Number of time samples: ", num_time_samples)

    # Initialize the rtf array
    rtf = np.zeros((num_wave_numbers, num_time_samples, num_mics), dtype=np.complex64)
    signal_variance = np.zeros((num_wave_numbers, num_time_samples, num_mics), dtype=np.complex64)

    # Initialize the STFT array for the microphone signals
    stft_mic_signals = np.array([np.zeros_like(Zxx_ref, dtype=np.complex64)] * num_mics)
    stft_mic_noise = np.array([np.zeros_like(Zxx_ref, dtype=np.complex64)] * num_mics)
    stft_mic_source = np.array([np.zeros_like(Zxx_ref, dtype=np.complex64)] * num_mics)

    # Calculate the STFT for each microphone
    for mic_idx in range(num_mics):
        _, _, Zxx_mic = stft(mic_signals[mic_idx], fs, window=STFT_WINDOW, nperseg=nperseg, noverlap=noverlap)
        stft_mic_signals[mic_idx] = Zxx_mic

        _, _, Zxx_mic = stft(noise_signal[mic_idx], fs, window=STFT_WINDOW, nperseg=nperseg, noverlap=noverlap)
        stft_mic_noise[mic_idx] = Zxx_mic

        _, _, Zxx_mic = stft(source_signal[mic_idx], fs, window=STFT_WINDOW, nperseg=nperseg, noverlap=noverlap)
        stft_mic_source[mic_idx] = Zxx_mic
    
    # Define the cross power spectal density matrix Rx and Rn
    Rx = np.zeros((num_wave_numbers, num_time_samples, num_mics, num_mics), dtype=np.complex64)
    Rn = np.zeros((num_wave_numbers, num_time_samples, num_mics, num_mics), dtype=np.complex64)
    Rs = np.zeros((num_wave_numbers, num_time_samples, num_mics, num_mics), dtype=np.complex64)
    Rs_est = np.zeros((num_wave_numbers, num_time_samples, num_mics, num_mics), dtype=np.complex64)
    
    xxH = np.zeros((num_mics, num_mics), dtype=np.complex64)
    nnH = np.zeros((num_mics, num_mics), dtype=np.complex64)
    ssH = np.zeros((num_mics, num_mics), dtype=np.complex64)
    
    # Initialize the source detection variables
    source_present_flag = False
    source_present_begin = 0
    source_present_count = 0
    source_not_present_count = 0

    source_detection_begin = 0
    source_det_count = 0
    source_FA_count = 0
    source_MD_count = 0
    source_no_det_count = 0

    avg_correlation_present_score = 0
    avg_correlation_not_present_score = 0
    avg_eig_ratio_present = 0
    avg_eig_ratio_not_present = 0

    xxH_eig_src_present = []
    Rs_eig_src_present = []
    xxH_eig_src_not_present = []
    Rs_eig_src_not_present = []

    # Plot the audio signals from the source and from all sound to the microphones
    speech_det_thres_per_band = np.zeros(num_wave_numbers)
    audio_det_thres, speech_det_thres_per_band = audio_speech_det(stft_mic_signals, stft_mic_source, plot=False)

    # Measure powers per time frame
    mic_power = np.sum(np.abs(stft_mic_signals[0][:])**2, axis=0)
    mic_source_power = np.abs(stft_mic_source[0][:])**2

    # Calculate Cross Power Spectral Density Matrix Rx of the data
    print("Estimating Rx and Rn... may take a while")

    start_time = time.time()
    k = 0
    for l in range(num_time_samples):
        if mic_power[l] < audio_det_thres: # Detect non zero volume audio, happens almost never with sources close to the microphones
            rtf[:,l,:] = 1
            Rx[k][l] = 0
            Rn[k][l] = 0
            Rs[k][l] = 0
            signal_variance[k][l] = np.trace(Rs[k][l])/num_mics
            continue

        for k in range(num_wave_numbers):
            # Check if the source is present using a voice detector
            if mic_source_power[k][l] > speech_det_thres_per_band[k]:
                source_present_flag = True

                # Increment the source present count for stastics
                if l >= cheat_time:
                    source_present_count += 1
                
                # Measure time frame when the source is first present strong enough
                if(source_present_begin == 0):
                    source_present_begin = l
            else:
                if l >= cheat_time:
                    source_not_present_count += 1
                source_present_flag = False

            # Compute CPSD for each pair of microphones with a small pre-cheat time
            xxH = np.outer(stft_mic_signals[:, k, l], np.conj(stft_mic_signals[:, k, l]))
            nnH = np.outer(stft_mic_noise[:, k, l], np.conj(stft_mic_noise[:, k, l]))
            ssH = np.outer(stft_mic_source[:, k, l], np.conj(stft_mic_source[:, k, l]))
            Rs[k][l] = ssH # perfect estimate
            if l < cheat_time:
                if mic_source_power[k][l] > speech_det_thres_per_band[k]:
                    Rx[k][l] = Rx[k][l-1]*cheat_alpha + xxH*(1 - cheat_alpha)
                    Rn[k][l] = Rn[k][l-1]
                else:
                    Rx[k][l] = Rx[k][l-1]
                    Rn[k][l] = Rn[k][l-1]*cheat_alpha + nnH*(1 - cheat_alpha)
            else:
                # Compute the EVD of the covariance matrix xxH - Rn
                #w_x, V_x = np.linalg.eig(xxH - nnH)
                w_x, V_x = np.linalg.eig(xxH - Rn[k][l-1])

                # Sort eigenvalues and eigenvectors in descending order
                sorted_indices = np.argsort(w_x)[::-1]
                eigenvalues = w_x[sorted_indices]
                max_eigenvalue = eigenvalues[0]
                noise_eigenvalue = eigenvalues[1]
                eigenvectors = V_x[:, sorted_indices]
                max_eigenvector = eigenvectors[:, 0]

                # Compute Rs from the previous Rs and Rn
                Rs[k][l] = Rx[k][l-1] - Rn[k][l-1] #uncommenting this line is cheating
                w_s, V_s = np.linalg.eig(Rs[k][l])

                # Sort eigenvalues and eigenvectors in descending order
                sorted_indices = np.argsort(w_s)[::-1]
                eigenvalues_prev = w_s[sorted_indices]
                max_eigenvalue_prev = eigenvalues_prev[0]
                noise_eigenvalue_prev = eigenvalues_prev[1]
                eigenvectors_prev = V_s[:, sorted_indices]
                max_eigenvector_prev = eigenvectors_prev[:, 0]

                # Decide on the detection threshold to update the noise space
                corr_score = np.abs(np.vdot(max_eigenvector, max_eigenvector_prev)) / (np.linalg.norm(max_eigenvector) * np.linalg.norm(max_eigenvector_prev))
                
                # Gather the information about the eigenvalues of the estimated source and measurement
                if source_present_flag:
                    #print("max_eigenvalue: ", max_eigenvalue, "\t\tnoise_eigenvalue: ", noise_eigenvalue)
                    xxH_eig_src_present.append(w_x)
                    Rs_eig_src_present.append(w_s)
                    
                    # Update the average correlation score
                    avg_correlation_present_score += corr_score
                    avg_eig_ratio_present += np.abs(max_eigenvalue / noise_eigenvalue)

                    # print("source present ")
                    # print("corr_score: ", corr_score)
                    # print("===xxH - nnH===")
                    # print("np.abs(max_eigenvalue) / np.abs(noise_eigenvalue): ", np.abs(max_eigenvalue) / np.abs(noise_eigenvalue))
                    # print("max_eigenvalue: ", max_eigenvalue, "\t\tnoise_eigenvalue: ", noise_eigenvalue)
                    # print("eigenvalues ", eigenvalues)
                    # print("===ssH===")
                    # print("np.abs(max_eigenvalue_prev) / np.abs(noise_eigenvalue_prev): ", np.abs(max_eigenvalue_prev) / np.abs(noise_eigenvalue_prev))
                    # print("max_eigenvalue_prev: ", max_eigenvalue_prev, "\t\tnoise_eigenvalue_prev: ", noise_eigenvalue_prev)
                    # print("eigenvalues_prev ", eigenvalues_prev)
                    # input("Press Enter to continue...")
                    # print("=====================================")

                else:
                    xxH_eig_src_not_present.append(w_x)
                    Rs_eig_src_not_present.append(w_s)

                    # Update the average correlation score
                    avg_correlation_not_present_score += corr_score
                    avg_eig_ratio_not_present += np.abs(max_eigenvalue / noise_eigenvalue)
                    
                    # # print("source not present ")
                    # print("corr_score: ", corr_score)
                    # print("===xxH - nnH===")
                    # print("np.abs(max_eigenvalue) / np.abs(noise_eigenvalue): ", np.abs(max_eigenvalue) / np.abs(noise_eigenvalue))
                    # print("max_eigenvalue: ", max_eigenvalue, "\t\tnoise_eigenvalue: ", noise_eigenvalue)
                    # print("eigenvalues ", eigenvalues)
                    # print("===ssH===")
                    # print("np.abs(max_eigenvalue_prev) / np.abs(noise_eigenvalue_prev): ", np.abs(max_eigenvalue_prev) / np.abs(noise_eigenvalue_prev))
                    # print("max_eigenvalue_prev: ", max_eigenvalue_prev, "\t\tnoise_eigenvalue_prev: ", noise_eigenvalue_prev)
                    # print("eigenvalues_prev ", eigenvalues_prev)
                    # input("Press Enter to continue...")
                    # print("=====================================")

                # Gather statistics on the source detection
                if corr_score > det_threshold and max_eigenvalue > np.abs(noise_eigenvalue)*eig_val_thr_ratio:
                    # Check when the detection algorithm starts detecting the source
                    if source_detection_begin == 0:
                        source_detection_begin = l

                    # Check for false alarms and succefull detects
                    if source_present_flag:
                        source_det_count += 1
                    else:
                        source_FA_count += 1
                else:
                    if source_present_flag:
                        source_MD_count += 1
                    else:
                        source_no_det_count += 1

                # Update the noise and signal space
                if corr_score > det_threshold and max_eigenvalue > np.abs(noise_eigenvalue)*eig_val_thr_ratio:
                    Rx[k][l] = Rx[k][l-1]*alpha + xxH*(1 - alpha)
                    Rn[k][l] = Rn[k][l-1]
                else:
                    Rx[k][l] = Rx[k][l-1]
                    Rn[k][l] = Rn[k][l-1]*alpha + xxH*(1 - alpha)

            # Regularization of Rn
            # Rn[k][l] = Rn[k][l] + np.eye(Rn[k][l].shape[0]) * 1e-10 # Regularization term
            
            if use_GEVD :
                Rs_est[k][l] = Rx[k][l] - Rn[k][l]
                w_s, V_s = np.linalg.eigh(Rs_est[k][l])

                # Sort eigenvalues and eigenvectors in descending order and truncate the M-r smallest eigenvalues
                sorted_indices = np.argsort(w_s)[::-1]
                eigenvalues = w_s[sorted_indices]
                eigenvectors = V_s[:, sorted_indices]
                max_eigenvalue = eigenvalues[0]
                max_eigenvector = eigenvectors[:, 0]
                 
                #Rs[k][l] = np.dot(max_eigenvector, np.dot(max_eigenvalue, max_eigenvector.conj().T))

                # Calculate the signal variance from the Rs_est matrix
                signal_variance[k][l] = np.trace(Rs_est[k][l]) / num_mics

                # Now calculate the rtf from the principal eigenvector of Rs_hat for one source
                rtf[k][l] = max_eigenvector

            else:
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
                max_eigenvector = eigenvectors[:, 0]

                # Calculate the signal variance from the Rs_est matrix
                signal_variance[k][l] = np.trace(Rs_hat) / num_mics

                # Now calculate the rtf from the principal eigenvector of Rs_hat for one source
                rtf[k][l] = max_eigenvector

            rtf[k][l] = rtf[k][l] / rtf[k][l][0]

        # Print the progress every 10% of the total number of time samples
        if l % (num_time_samples // 10) == 0:
            print(f"Progress: {l / num_time_samples * 100:.0f}%")
    
    # Measure time taken to calculate Rs
    end_time = time.time()
    print(f"Time taken to calculate Rs: {end_time - start_time:.2f} seconds, for {num_wave_numbers * num_time_samples * 16} covariance and eigenvalue calculations")

    # Print the average correlation and eigenvalue statistics
    print("avg_correlation_present_score: ", avg_correlation_present_score/source_present_count)
    print("avg_correlation_not_present_score: ", avg_correlation_not_present_score/source_not_present_count)
    print("avg_eig_ratio_present: ", avg_eig_ratio_present/source_present_count)
    print("avg_eig_ratio_not_present: ", avg_eig_ratio_not_present/source_not_present_count)

    # Print the source detection statistics
    print("source_present_begin: ", source_present_begin)
    print("source_detection_begin: ", source_detection_begin)

    print("source_present_count: ", source_present_count)
    print("source_not_present_count: ", source_not_present_count)

    print("source_det_count: ", source_det_count)
    print("source_no_det_count: ", source_no_det_count)
    print("source_FA_count: ", source_FA_count)
    print("source_MD_count: ", source_MD_count)

    print("Detection score: ", np.round(source_det_count/source_present_count,2))
    print("No detection score: ", np.round(source_no_det_count/source_not_present_count ,2))
    print("Missed detection score: ", np.round(source_MD_count/source_present_count, 2))
    print("False alarm score: ", np.round(source_FA_count/source_not_present_count, 2))
    
    xxH_eig_src_present = np.array(xxH_eig_src_present)
    Rs_eig_src_present = np.array(Rs_eig_src_present)
    xxH_eig_src_not_present = np.array(xxH_eig_src_not_present)
    Rs_eig_src_not_present = np.array(Rs_eig_src_not_present)

    xxH_eig_src_present_mean = np.mean(xxH_eig_src_present, axis=0)
    Rs_eig_src_present_mean = np.mean(Rs_eig_src_present, axis=0)
    xxH_eig_src_not_present_mean = np.mean(xxH_eig_src_not_present, axis=0)
    Rs_eig_src_not_present_mean = np.mean(Rs_eig_src_not_present, axis=0)

    xxH_eig_src_present_var  = np.var(xxH_eig_src_present, axis=0)
    Rs_eig_src_present_var = np.var(Rs_eig_src_present, axis=0)
    xxH_eig_src_not_present_var = np.var(xxH_eig_src_not_present, axis=0)
    Rs_eig_src_not_present_var = np.var(Rs_eig_src_not_present, axis=0)

    #print the mean and variance of the eigenvalues of the source and noise space
    print("xxH_eig_src_present_mean: ", xxH_eig_src_present_mean)
    print("Rs_eig_src_present_mean: ", Rs_eig_src_present_mean)
    print("xxH_eig_src_not_present_mean: ", xxH_eig_src_not_present_mean)
    print("Rs_eig_src_not_present_mean: ", Rs_eig_src_not_present_mean)

    print("xxH_eig_src_present_var: ", xxH_eig_src_present_var)
    print("Rs_eig_src_present_var: ", Rs_eig_src_present_var)
    print("xxH_eig_src_not_present_var: ", xxH_eig_src_not_present_var)
    print("Rs_eig_src_not_present_var: ", Rs_eig_src_not_present_var)

    # Plot the mean and variance of the eigenvalues of the source and noise space in two subplots
    # plt.figure(figsize=(10, 6))
    # plt.subplot(1, 2, 1)
    # plt.plot(xxH_eig_src_present_mean, label='xxH-Rn Source present')
    # plt.plot(Rs_eig_src_present_mean, label='Rs Source presentr')
    # plt.plot(xxH_eig_src_not_present_mean, label='xxH-Rn Source not present')
    # plt.plot(Rs_eig_src_not_present_mean, label='Rs Source not present')
    # plt.title('Mean of the eigenvalues of the source and noise space')
    # plt.xlabel('Sorted Eigenvalue Index')
    # plt.ylabel('Eigenvalue mean')
    # plt.legend()

    # plt.subplot(1, 2, 2)
    # plt.plot(xxH_eig_src_present_var, label='xxH-Rn Source present')
    # plt.plot(Rs_eig_src_present_var, label='Rs Source presentr')
    # plt.plot(xxH_eig_src_not_present_var, label='xxH-Rn Source not present')
    # plt.plot(Rs_eig_src_not_present_var, label='Rs Source not present')
    # plt.title('Variance of the eigenvalues of the source and noise space')
    # plt.xlabel('Sorted Eigenvalue Index')
    # plt.ylabel('Eigenvalue variance')
    # plt.legend()
    # plt.show()

    return rtf, Rn, signal_variance, stft_mic_signals

# Not used ############################################################################################################
def infer_geometry_ULA(impulse_responses, sample_rate=16000, speed_of_sound=343.0):
    geometry = np.zeros((num_sources + num_mics, 2))

    # Calculate microphone coordinates
    frequency = sample_rate//2  # Assume the frequency is the Nyquist frequency
    wavelength = speed_of_sound / frequency  # Calculate the wavelength of the sound
    print("Inter-microphone spacing: ", wavelength/2)
    for mic_idx in range(num_mics):
        mic_x = mic_idx * wavelength / 2  # Calculate the x-coordinate based on the microphone index and the wavelength
        mic_y = 0  # Assume the microphone is located at y=0
        geometry[num_sources + mic_idx] = [mic_x, mic_y]

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
                peak_mag_2 = peak_mag_2 + 0.00001

            # Calculate the time delay and distance for each microphone and add a noise term
            time_delay_1 = peak_index_1 / sample_rate + np.random.normal(0, 0.000001)
            time_delay_2 = peak_index_2 / sample_rate + np.random.normal(0, 0.000001)
            if time_delay_1 == time_delay_2:
                time_delay_2 = time_delay_2 + 0.00001

            # Calculate the distance traveled by sound for each microphone
            distance_1 = time_delay_1 * speed_of_sound
            distance_2 = time_delay_2 * speed_of_sound

            # Calculate the difference in time delay and distance between the two microphones
            difference_time_delay = time_delay_2 - time_delay_1 # Calculate the time delay difference
            difference_distance = difference_time_delay * speed_of_sound  # Calculate the difference in distance
            print("Difference in distance: ", difference_distance)
            print("Difference in time delay: ", difference_time_delay)
            # Use trigonometry to calculate the x and y coordinates
            angle_mic_1 = np.arccos(((wavelength/2)**2 + peak_mag_1**2 - peak_mag_2**2) / (2 * (wavelength/2) * peak_mag_1))
            #angle_mic_1 = np.arccos(((wavelength/2)**2 + peak_mag_2**2 - peak_mag_1**2) / (2 * (wavelength/2) * peak_mag_2))
            
            # Calculate the x and y coordinates of the source
            source_x = distance_1 * np.cos(angle_mic_1) + mic_idx * wavelength / 2
            source_y = distance_1 * np.sin(angle_mic_1)
            geometry[source_idx] = [source_x, source_y]

    return geometry