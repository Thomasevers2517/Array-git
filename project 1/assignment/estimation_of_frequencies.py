#1) It is rank 1

import config_assignment
import numpy as np
import matplotlib.pyplot as plt
from src.espirit.espirit import espirit_freq
from src.data_generation.generate_data import generate_data
d=2
delta =0.5
norm_freq = 0.02

X,_,_,_ = generate_data(M=3, N=500, theta_degrees=[30, 85], f = [norm_freq*1.8, norm_freq], snr_db=[1000, 1000], delta=delta)

pred_freqs = espirit_freq(X, d, 1)
print("pred_freqs: ", pred_freqs)
print("actual freqs: ", [norm_freq*1.8, norm_freq])

