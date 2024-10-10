import config_assignment
import numpy as np
import matplotlib.pyplot as plt
from src.espirit.espirit import espirit
from src.data_generation.generate_data import generate_data
d = 2
M = 3
N = 20
m = 3
delta = 0.5
thetas = [-20, 30]
freqs = [0.1, 0.12]
SNRs = [100, 100]

X,_,_,_ = generate_data(M=M, N=N, theta_degrees=thetas, f=freqs, snr_db=SNRs, delta=delta)

thetas_estimated = espirit(X, d, delta)

print("Estimated thetas: ", thetas_estimated)
print("Actual thetas: ", thetas)
print("MSE: ", np.mean(np.sort(thetas)-np.sort(thetas_estimated))**2)