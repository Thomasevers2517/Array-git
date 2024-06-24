
import config_assignment
from src.data_generation.generate_data_khatri_rao import generate_data_khatri_rao
from src.data_generation.generate_data import generate_data
from src.espirit.joint_diagonalisation import joint_diag
from src.espirit.joint_estimation import joint
from src.espirit.espirit import espirit
from src.espirit.espirit import espirit_freq
import numpy as np
from matplotlib import pyplot as plt
d=2
M =3
N = 20
m =10
P =100
delta = 0.5

freqs = np.array([0.1, 0.12])
angles = np.array([-20, 30])

snr =10


X_normal, S_with_car, A, Noise = generate_data(M=M, N=N, theta_degrees=angles, f=freqs, snr_db=[snr,snr], delta=delta)

if not np.array_equal(X_normal, A @ S_with_car):
    raise ValueError("X_normal is not equal to A @ S_with_car")
    
 
def bf_angle(X, d, delta):
    pred_angle = espirit(X_normal, d, delta)
    print("Predicted angles: ", pred_angle)

    phi = np.array( np.exp(1j* (2*np.pi * delta  *  np.sin(np.deg2rad(pred_angle)))))
    A_est_angle = np.zeros((M, d), dtype=complex) 
    A_est_angle[ 0, :] = np.ones(d)
    for m in range(1,M):
        A_est_angle[m, :] = A_est_angle[m-1, :] * phi
        
    W_angle = A_est_angle @ np.linalg.pinv(A_est_angle.conj().T @ A_est_angle)
    return W_angle

def bf_freq(X, d, m, N):
    pred_freq = espirit_freq(X, d, 1)

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
S_est_freq2 = W_freq.conj().T @ X_normal


def a(theta_degrees):
    a = np.zeros((M), dtype=complex)
    a[0] = 1
    phi = np.exp(1j * 2 * np.pi * delta * np.sin(np.deg2rad(theta_degrees)))
    for m in range(1,M):
        a[m] = a[m-1] * phi
    return a
def plot_beam_pattern(W, label):
    y = []
    for theta in range(-90, 90):
        result =np.dot(W.conj().T,  a(theta))
        y.append(abs(result))
    plt.plot(range(-90, 90), y, label=label)
    
    
    

plt.figure()

plot_beam_pattern(W_angle[:,0], 'Angle of arrival est 1')
plot_beam_pattern(W_angle[:,1], 'Angle of arrival est 2')
plot_beam_pattern(W_freq[:,0], 'Freq of arrival est 1')
plot_beam_pattern(W_freq[:,1], 'Freq of arrival est 2')
est_angles = espirit(X_normal, d, delta)
for i in range(d):
    plt.axvline(est_angles[i], 0, color='r', linestyle='--')
plt.xlabel('Angle')
plt.ylabel('Magnitude')
plt.title('Beam Pattern')
plt.legend()
plt.show()

