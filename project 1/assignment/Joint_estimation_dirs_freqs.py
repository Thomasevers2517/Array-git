import config_assignment
from src.data_generation.generate_data_khatri_rao import generate_data_khatri_rao
from src.espirit.joint_diagonalisation import joint_diag
from src.espirit.joint_estimation import joint
import numpy as np
d=2
M =7
m =10
P =100
delta = 0.2

freqs = np.array([0.0101, 0.01])
angles = np.array([30, 65])
X,_ = generate_data_khatri_rao(M=M, N=500, theta_degrees=angles, f=freqs, snr_db=[100, 100], m=m, P=P, delta=delta)
pred_freqs, pred_angles = joint(X, d, m, delta, P)

print("pred_freqs: \n", pred_freqs)
print("pred_angles: \n", pred_angles)