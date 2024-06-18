import os
import sys

from src.data_generation.generate_data_khatri_rao import generate_data_khatri_rao
from src.espirit.joint_diagonalisation import joint_diag
import numpy as np

import scipy.io

def joint(X, d, m, delta, P):
    print("X: ", X.shape)
    M = int(X.shape[0]/m)
    U, S,V = np.linalg.svd(X)
    U = U[:, :d]
    S = S[:d]
    V = V[:d, :]

    U_phi_x = np.zeros((M*(m-1), d), dtype=complex)
    U_phi_y = np.zeros((M*(m-1), d), dtype=complex)

    U_theta_x = np.zeros(((M-1)*m, d), dtype=complex)
    U_theta_y = np.zeros(((M-1)*m, d), dtype=complex)

    for i in range(m-1):
        U_phi_x[i*M:(i+1)*M, :] = U[i*M:(i+1)*M, :]
        U_phi_y[i*M:(i+1)*M, :] = U[(i+1)*M:(i+2)*M, :]
        
    for i in range(m):    
        for j in range(M-1):
            U_theta_x[i*(M-1)+j, :] = U[i*M+j, :]
            U_theta_y[i*(M-1)+j, :] = U[i*M+j+1, :]


    U_phi_x_inv = np.linalg.pinv(U_phi_x)
    U_theta_x_inv = np.linalg.pinv(U_theta_x)




    M_mat_phi = U_phi_x_inv @ U_phi_y
    M_mat_theta = U_theta_x_inv @ U_theta_y


    M_mat = np.zeros((d, 2*d), dtype=complex)

    M_mat[:, :d] = M_mat_phi
    M_mat[:, d:] = M_mat_theta
        

    scipy.io.savemat('data.mat', {'M': M_mat})

    V, D = joint_diag(M_mat, 1e-10)


    phi = V @ M_mat_phi @ np.linalg.inv(V)
    theta = V @ M_mat_theta @ np.linalg.inv(V)
    # phi = V @ D[:,:d] @ np.linalg.inv(V)
    # theta = V @ D[:,d:] @ np.linalg.inv(V)


    phi_rads = np.angle(phi)    
    theta_rads = np.angle(theta)

    phi_rads = np.eye(d) *  phi_rads
    theta_rads = np.eye(d) *  theta_rads

    pred_freqs = phi_rads * P/(2*np.pi)
    pred_angles = np.arcsin(theta_rads/(2*np.pi*delta)) * 180/np.pi

    pred_freqs = np.diag(pred_freqs)
    pred_angles = np.diag(pred_angles)
    
    return list(pred_freqs), list(pred_angles)

