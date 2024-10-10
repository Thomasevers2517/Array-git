import config_assignment
from src.data_generation.generate_data_khatri_rao import generate_data_khatri_rao
from src.espirit.joint_diagonalisation import joint_diag
from src.espirit.joint_estimation import joint
import numpy as np
d = 2
M = 3
N = 20
m = 3
P = 2
delta = 0.5
thetas = [-20, 30]
freqs = [0.1, 0.12]
SNRs = [100, 100]

X,_,_ = generate_data_khatri_rao(M=M, N=N, theta_degrees=thetas, f=freqs, snr_db=SNRs, m=m, P=P, delta=delta)
pred_freqs, pred_angles = joint(X, d, m, delta, P)

print("Actual freqs: \n", freqs)
print("pred_freqs: \n", np.sort(pred_freqs))
print("Actual angles: \n", thetas)
print("pred_angles: \n", np.sort(pred_angles))
