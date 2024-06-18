import config_test
import numpy as np
import matplotlib.pyplot as plt
from src.espirit.espirit import espirit
from src.data_generation.generate_data import generate_data
d=2
delta =0.5
norm_freq = 0.02
c= 343
Fs = 1e6
wave_length = c/(norm_freq*Fs)

X,_,_,_ = generate_data(M=11, N=500,theta_degrees=[30, 85], f = [norm_freq*1.01, norm_freq], snr_db=[1000, 1000], delta=delta)

test = espirit(X, d, delta)