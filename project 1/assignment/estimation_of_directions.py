import config_assignment
import numpy as np
import matplotlib.pyplot as plt
from src.espirit.espirit import espirit
from src.data_generation.generate_data import generate_data
d=2
delta =0.5
norm_freq = 0.02
c= 343
thetas = [30, 85]

X,_,_,_ = generate_data(M=10, N=500, theta_degrees=thetas, f = [norm_freq*1.01, norm_freq], snr_db=[1000, 1000], delta=delta)

thetas_estimated = espirit(X, d, delta)
print("MSE: ", np.mean((np.sort(thetas)-np.sort(thetas_estimated))**2))