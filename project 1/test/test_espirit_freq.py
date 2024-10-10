import config_test
import numpy as np
import matplotlib.pyplot as plt
from src.espirit.espirit import espirit_freq
from src.data_generation.generate_data import generate_data
d=2
delta =0.5
freqs = [0.1, 0.12]
m = 3

X,_,_,_ = generate_data(M=3, N=20, theta_degrees=[-20, 30], f = freqs, snr_db=[2000, 2000], delta=delta)

test = espirit_freq(X, d, m, 1)
print(test)