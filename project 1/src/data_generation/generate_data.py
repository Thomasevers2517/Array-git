import numpy as np
def generate_data(M=3, N=500, distance=0.2, theta_degrees=[30, 50], f = [0.005, 0.002], snr_db=[100, 100], Fs = 1000000):
    """Generates data for the given parameters
    Parameters
    ----------
        M : int
        Number of antennas
        N : int
        Number of samples
        delta : float
        Time delay between antennas
        theta_degrees : list
        List of angles of arrival of sources between -180 and 180 degrees
        f : list
        List of normalized frequencies of sources, should be between 0 and 1
        snr_db : list
        List of signal to noise ratios for sources in dB
        Returns
        -------
        X : numpy.ndarray
        Matrix of data generated
        S : numpy.ndarray
        Matrix of sources
        TAU : numpy.ndarray
        Matrix of time delays
        NOISE : numpy.ndarray
        Matrix of noise
        """
    theta = np.deg2rad(theta_degrees)
    print("theta: ", theta)
    snr = np.power(10, np.array(snr_db)/10)
    assert M > 0, "Number of antennas should be greater than 0"
    assert N > 0, "Number of samples should be greater than 0"
    assert distance > 0, "Time delay between antennas should be greater than 0"
    assert len(theta) > 0, "List of angles of arrival of sources should be greater than 0"
    assert len(f) ==len(theta), "List of frequencies of sources should be equal to the number of sources"
    assert len(snr) == len(theta), "List of signal to noise ratios for sources should be equal to the number of sources"
    assert all(isinstance(i, int) for i in [M, N]), "Number of antennas and number of samples should be integers"
    assert all(theta[i] >= -np.pi and theta[i] <= np.pi for i in range(len(theta))), "Angles of arrival of sources should be between -pi and pi"
    
    
    print("Generating data with the following parameters:")
    print("num antennas (M): ", M)
    print("num samples (N): ", N)
    print("num sources (d): ", len(theta))
    print("distance: ", distance)
    print("theta: ", theta)
    print ("f: ", f)

    d= len(theta)
    c = 343
    theta = np.array(theta)
    f = np.array(f)
    X = np.zeros((M, N), dtype=complex)
    TAU =  np.zeros((M, d, N), dtype=complex) #antennas, sources, time
    print("ANGLE: ", np.sin(np.deg2rad(theta)))
    TAU[0, :, :] = np.repeat(np.array( np.exp(1j* (2*np.pi * distance * f * Fs /c * np.sin(theta)))), N).reshape(d, N)
    S = np.zeros((d, N), dtype=complex)
    
    for m in range(1,M):
        TAU[m, :, :] = TAU[m-1, :, :] * TAU[0, :, :]
    
    # Create a 3D array to hold the noise
    NOISE = np.zeros((M, d, N), dtype=complex)

    # Generate the noise for each source
    for i in range(d):
        NOISE[:, i, :] = np.random.normal(0, 1/(np.sqrt(snr[i])), (M, N))
    
    TAU = TAU + NOISE
    for i in range(d):
        S[i, :] = np.exp(1j *2 * np.pi * f[i] * np.arange(N))
        
    for N in range(N):
        X[:, N] = TAU[:, :, N] @ S[:, N] 
    print("data generated successfully")

    return X, S, TAU, NOISE