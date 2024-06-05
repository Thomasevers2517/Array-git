import numpy as np
import scipy as sp
def generate_data_khatri_rao(M=3, N=500, theta_degrees=[30, 50], f = [0.005, 0.002], snr_db=[100, 100]):
    d = len(theta_degrees)
    P = 60
    f = np.array(f)
    theta = np.array(np.deg2rad(theta_degrees))
    snr_db = np.array(snr_db)
    delta = 0.2
    m = 5
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
    

    for i in range(d):
        PHI[i, i] = phi[i]
    
    print("F_phi: ", F_phi.shape)
    print("A_theta: ", A_theta.shape)
    print("sp.linalg.khatri_rao(F_phi, A_theta)", sp.linalg.khatri_rao(F_phi, A_theta).shape)
    print("B: ", B.shape)
    print("F_p: ", F_p.shape)
    print("S: ", S.shape)
    
    X = sp.linalg.khatri_rao(F_phi, A_theta) @ B @ np.multiply(F_p, S)
 
    
    print("X: ", X.shape)
    return X
