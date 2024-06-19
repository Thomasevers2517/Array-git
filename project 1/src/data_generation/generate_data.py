import numpy as np
def generate_data(M=3, N=500, theta_degrees=[30, 50], f = [0.005, 0.002], snr_db=[100, 100], delta = 0.5):
    """Generates data for the given parameters
    Parameters
    ----------
        M : int
        Number of antennas
        N : int
        Number of samples
        delta : float
        Time delay between antennas
        theta_rad_degrees : list
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
    theta_rad = np.deg2rad(theta_degrees)
    print("theta_rad: ", theta_rad)
    snr = np.power(10, np.array(snr_db)/10)
    assert M > 0, "Number of antennas should be greater than 0"
    assert N > 0, "Number of samples should be greater than 0"
    assert len(theta_rad) > 0, "List of angles of arrival of sources should be greater than 0"
    assert len(f) ==len(theta_rad), "List of frequencies of sources should be equal to the number of sources"
    assert len(snr) == len(theta_rad), "List of signal to noise ratios for sources should be equal to the number of sources"
    assert all(isinstance(i, int) for i in [M, N]), "Number of antennas and number of samples should be integers"
    assert all(theta_rad[i] >= -np.pi and theta_rad[i] <= np.pi for i in range(len(theta_rad))), "Angles of arrival of sources should be between -pi and pi"
    
    
    print("Generating data with the following parameters:")
    print("num antennas (M): ", M)
    print("num samples (N): ", N)
    print("num sources (d): ", len(theta_rad))
    print("snr: ", snr)
    print("delta: ", delta)
    print("theta_rad: ", theta_rad)
    print ("f: ", f)

    d= len(theta_rad)
    c = 343


    X = np.zeros((M, N), dtype=complex)
    A =  np.zeros((M, d), dtype=complex) #antennas, sources, time
    print("ANGLE: ", np.sin(theta_rad))
    
    phi = np.array( np.exp(1j* (2*np.pi * delta  *  np.sin(theta_rad))))
    A[ 0, :] = np.ones(d)
    for m in range(1,M):
        A[m, :] = A[m-1, :] * phi
    
    # Create a 3D array to hold the noise
    NOISE = np.zeros((d, N), dtype=complex)

    # Generate the noise for each source
    for i in range(d):
        NOISE[i,:] = np.random.normal(0, 1/(np.sqrt(snr[i])), (N))
    
    
    S = np.ones((d, N), dtype=complex)
    S[0, :] = 1/np.sqrt(2) + 1j/np.sqrt(2)
    # S[:, int(N/2):] = 1/np.sqrt(2) + 1j/np.sqrt(2)
    S_withcar = np.zeros((d, N), dtype=complex)
    for i in range(d):
        S_withcar[i, :] = np.exp(1j *2 * np.pi * f[i] * np.arange(N)) * (S[i, :]) + NOISE[i, :]
        
    
    X = A @ S_withcar
    print("data generated successfully")

    return X, S_withcar, A, NOISE