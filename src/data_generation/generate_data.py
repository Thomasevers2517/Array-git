import numpy as np
def generate_data(M, N, delta, theta, f, snr):
    """Generates data for the given parameters
    Parameters
    ----------
        M : int
        Number of antennas
        N : int
        Number of samples
        delta : float
        Time delay between antennas
        theta : list
        List of angles of arrival of sources
        f : list
        List of frequencies of sources
        snr : list
        List of signal to noise ratios for sources
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
    print("Generating data with the following parameters:")
    print("num antennas (M): ", M)
    print("num samples (N): ", N)
    print("num sources (d): ", len(theta))
    print("delta: ", delta)
    print("theta: ", theta)
    print ("f: ", f)

    d= len(theta)
    c = 343
    theta = np.array(theta)
    f = np.array(f)
    X = np.zeros((M, N), dtype=complex)
    TAU =  np.zeros((M, d, N), dtype=complex) #antennas, sources, time
    TAU[0, :, :] = np.repeat(np.array( np.exp(1j* (2*np.pi*f) * (delta/c)*np.sin(theta))), N).reshape(d, N)
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