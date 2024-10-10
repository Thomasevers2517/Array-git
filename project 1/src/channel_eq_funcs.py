import numpy as np
def gendata_conv(s, P, N, sigma):
    """
    Generate the sequence x by sampling the received signal at rate 1/P
    """
    # Define the channel h(t)
    def h(t):
        if 0 <= t < 0.25:
            return 1
        elif 0.25 <= t < 0.5:
            return -1
        elif 0.5 <= t < 0.75:
            return 1
        elif 0.75 <= t < 1:
            return -1
        else:
            return 0
    
    # Generate the received signal x(t)
    t_vals = np.arange(0, N, 1/P)
    x = np.zeros(N*P, dtype=complex)
    for i, t in enumerate(t_vals):
        for k in range(N):
            if 0 <= t - k < 1:
                x[i] += h(t - k) * s[k]
        x[i] += (np.random.normal(0, sigma) + 1j * np.random.normal(0, sigma))/np.sqrt(2)
    
    return x

def make_X(x, P, N):
    X = np.zeros((2*P, N-1), dtype=complex)
    for i in range(2*P):
        for j in range(N-1):
            X[i, j] = x[i+j*P]
    return X