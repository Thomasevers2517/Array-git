import numpy as np
def espirit(X, d, delta):
    print("X: ", X.shape)
    M = X.shape[0]
    X_esp = X[:-1]
    print("X_esp: ", X_esp.shape)
    Y_esp = X[1:]
    print("Y_esp: ", Y_esp.shape)
    
    Z = np.concatenate((X_esp, Y_esp), axis=0)
    print("Z: ", Z.shape)
    U_z, S_z, V_z = np.linalg.svd(Z)
    print("U_z: ", U_z.shape) 
    print("S_z: ", S_z.shape)
    print("S_z: ", S_z)
    print("V_z: ", V_z.shape) 
    U_z = U_z[:, :d]
    U_x = U_z[:M-1]
    U_y = U_z[M-1:]
    print("U_z: ", U_z.shape) 
    print("U_x: ", U_x.shape)
    print("U_y: ", U_y.shape) 
    U_x_inv = np.linalg.pinv(U_x)
    U_x_inv_U_y = U_x_inv @ U_y
    print("U_x_inv: ", U_x_inv.shape)
    print("U_x_inv_U_y: ", U_x_inv_U_y.shape)
    
    phis, T = np.linalg.eig(U_x_inv_U_y)
    print("T:", T)
    print("T shape: ", T.shape)
    print("phis: ", phis)
    angles = np.angle(phis)
    print("angles/(2*np.pi*delta): ", angles/(2*np.pi*delta))
    thetas = np.rad2deg(np.arcsin(angles/(2*np.pi*delta)))
    thetas.sort()
    print("thetas: ", thetas)
    
    return thetas

def espirit_freq(X, d, m, Fs):
    print("X: ", X.shape)
    N = X.shape[1]
    M = X.shape[0]
    m = int(np.floor(N/3)) # this is recommended on the slides and book, but why is it good
    #m=3
    # print("X[:,0+1]: ", X[:,0+1])
    # print("sum(X[:,0+1],1): ", sum(X[:,0+1],1))
    # input("Press Enter to continue...")
    print("m: ", m)
    Z = np.zeros((m, N-m), dtype=complex)
    for i in range(m):
        for j in range(N-m):
            Z[i,j] = X[0,j+i]

    # X_esp = X[:-1]
    # print("X_esp: ", X_esp.shape)
    # Y_esp = X[1:]
    # print("Y_esp: ", Y_esp.shape)
    
    # Z = np.concatenate((X_esp, Y_esp), axis=0)
    print("Z: ", Z.shape)
    U_z, S_z, V_z = np.linalg.svd(Z)
    print("U_z: ", U_z.shape)
    print("S_z: ", S_z.shape)
    print("V_z: ", V_z.shape)
    print("S_z: ", S_z)
    U_z = U_z[:, :d] #times 2 because the negative frequencies are also included
    print("U_z: ", U_z.shape)
    U_x = U_z[:m-1]
    U_y = U_z[1:m]
    print("U_x: ", U_x.shape)
    print("U_y: ", U_y.shape)
    U_x_inv = np.linalg.pinv(U_x)
    print("U_x_inv: ", U_x_inv.shape)
    U_x_inv_U_y = U_x_inv @ U_y
    
    phis, T = np.linalg.eig(U_x_inv_U_y)
    print("T: ", T.shape)
    print("phis: ", phis)
    angles = np.angle(phis)
    print("angles", angles)
    freqs = np.abs(angles*Fs)
    
    # freqs = freqs[freqs>0]
    # freqs.sort()
    # freqs = np.unique(freqs)
    norm_freqs = freqs/(Fs*2*np.pi)
    print("freqs: ", freqs)
    print("norm_freqs: ", norm_freqs)

    return norm_freqs
