import numpy as np
import scipy as sp

def generate_data_khatri_rao(M=3, N=500, theta_degrees=[30, 50], f = [0.005, 0.002], snr_db=[100, 100], m=5, P = 60, delta = 0.5): #delta =0.2
    d = len(theta_degrees)
    
    f = np.array(f)
    theta = np.array(np.deg2rad(theta_degrees))
    snr_db = np.array(snr_db)
    snr = 10**(snr_db/10)
    assert all(f < P/2) and all(f > -P/2), "f*Fs should be less than P/2 and greater than 0"
    
    phi = np.exp(1j*2*np.pi*f/P)
    
    A_theta = np.zeros((M, d), dtype=complex) #ant by src
    B = np.zeros((d, d), dtype=complex) #src by time
 
    S = np.zeros((d, N), dtype=complex) #src by time
    
    PHI = np.zeros((d, d), dtype=complex) #ant by src
    
    F_phi = np.zeros((d, m), dtype=complex) #src by time
    F_p = np.zeros((d, N), dtype=complex) #src by time

    F_phi[:,0] = np.ones(d)
    F_p[:,0] = np.ones(d)
    for i in range(N-1):
        F_p[:,i+1] = phi**P * F_p[:,i]
        
    for i in range(m-1):
        F_phi[:,i+1] = phi * F_phi[:,i]
    F_phi= F_phi.T
    
    A_theta[0, :] = np.array( np.exp(1j* (2*np.pi * delta  * np.sin(theta))))
    for i in range(1,M):
        A_theta[i, :] = A_theta[0,:] * A_theta[i-1, :]
    
    B = np.eye(d)
    
    S = np.ones((d, N))
    
    noise = np.random.randn(d, N) + 1j*np.random.randn(d, N)
    noise = (noise/np.abs(noise))

    noise[0, :] = noise[0, :] / np.sqrt(snr[0])
    noise[1, :] = noise[1, :] / np.sqrt(snr[1])
    
    S = S + noise

    for i in range(d):
        PHI[i, i] = phi[i]
    
    X = sp.linalg.khatri_rao(F_phi, A_theta) @ B @ np.multiply(F_p, S)
    print("X: ", X.shape)
    print("np.multiply(F_p, S): ", np.multiply(F_p, S).shape)
    print("sp.linalg.khatri_rao(F_p, S): ", sp.linalg.khatri_rao(F_p, S).shape)
    X = sp.linalg.khatri_rao(F_phi, A_theta) @ B @ sp.linalg.khatri_rao(F_p, S)
    print("X (F_p o S): ", X.shape)
    return X, S
