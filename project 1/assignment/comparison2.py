
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
delta = 0.2

freqs = np.array([0.1, 0.12])
angles = np.array([-20, 30])

snr =1000


X_normal, _, _, _ = generate_data(M=M, N=N, theta_degrees=angles, f=freqs, snr_db=[snr,snr], delta=delta)

pred_freq = espirit_freq(X_normal, d, 1)
pred_angle = espirit(X_normal, d, delta)


