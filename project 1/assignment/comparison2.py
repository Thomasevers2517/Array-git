import config_assignment
from src.data_generation.generate_data_khatri_rao import generate_data_khatri_rao
from src.data_generation.generate_data import generate_data
from src.espirit.joint_diagonalisation import joint_diag
from src.espirit.joint_estimation import joint
from src.espirit.espirit import espirit
from src.espirit.espirit import espirit_freq
import numpy as np
from matplotlib import pyplot as plt
d = 2
M = 3
N = 20
m = 3
P = 2
delta = 0.5

freqs = np.array([0.1, 0.12])
angles = np.array([-20, 30])

snr = 100

X_normal, S_with_car, A, _ = generate_data(M=M, N=N, theta_degrees=angles, f=freqs, snr_db=[snr,snr], delta=delta)

if not np.array_equal(X_normal, A @ S_with_car):
    raise ValueError("X_normal is not equal to A @ S_with_car")
    
def bf_angle(X, d, delta):
    pred_angle = espirit(X_normal, d, delta)
    print("Predicted angles: ", pred_angle)
    pred_angle.sort()
    print("Predicted angles: ", pred_angle)
    #input("enter")

    phi = np.array( np.exp(1j* (2*np.pi * delta  *  np.sin(np.deg2rad(pred_angle)))))
    #phi = np.exp(1j* (2*np.pi * delta  *  np.sin(np.deg2rad(pred_angle))))
    # print("phi: ", phi)
    # input("enter")
    A = np.zeros((M, d), dtype=complex) 
    A[ 0, :] = np.ones(d)
    for m in range(1,M):
        A[m, :] = A[m-1, :] * phi
        
    W_angle = A @ np.linalg.pinv(A.conj().T @ A)
    return W_angle

def bf_freq(X, d, m, N):
    pred_freq = espirit_freq(X, d, m, 1)
    print("Predicted freq: ", pred_freq)
    pred_freq.sort()
    print("Predicted freq: ", pred_freq)
    #input("enter")

    S_est_freq= np.zeros((d, N), dtype=complex)
    S_est_freq[:, 0] = np.ones(d)
    for i in range(1,N):
        S_est_freq[:, i] = S_est_freq[:, i-1] * np.exp(1j*2*np.pi*pred_freq)

    A_est_freq = np.zeros((M, d), dtype=complex)
    A_est_freq = X_normal @ np.linalg.pinv(S_est_freq)

    for i in range(d):
        A_est_freq[:, i] = A_est_freq[:, i] / A_est_freq[0, i]

    W_freq = A_est_freq @ np.linalg.inv(A_est_freq.conj().T @ A_est_freq)
    return W_freq

W_angle = bf_angle(X_normal, d, delta)
S_est_angle =  W_angle.conj().T @ X_normal
W_freq = bf_freq(X_normal, d, m, N)
print("Here")
S_est_freq2 = W_freq.conj().T @ X_normal

plt.figure()
plt.plot(S_with_car[0, :], marker = "o", label='Source 1')
plt.plot(S_with_car[1, :], marker = "o", label='Source 2')
plt.plot(S_est_angle[0, :], marker = "+", markersize="10", label='Estimated Source 1')
plt.plot(S_est_angle[1, :], marker = "+", markersize="10", label='Estimated Source 2')
plt.plot(S_est_freq2[0, :], label='Estimated Source 1 Freq')
plt.plot(S_est_freq2[1, :], label='Estimated Source 2 Freq')
plt.ylabel('Real part of the signal')
plt.xlabel('Time')
plt.title('Source and Estimated Source Signals')
plt.legend()
plt.show()
