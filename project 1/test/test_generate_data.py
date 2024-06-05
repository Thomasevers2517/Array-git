import config_test

from src.data_generation.generate_data import generate_data
import numpy as np

d=2
delta =0.1
norm_freq = 0.05
c= 343
Fs = 1e6
wave_length = c/(norm_freq*Fs)

X, S, TAU, NOISE = generate_data(M=4, N=15, distance=delta*wave_length, theta_degrees=[70, 30], f = [norm_freq, norm_freq], snr_db=[1000, 1000], Fs = Fs)


import matplotlib.pyplot as plt
plt.figure()
plt.plot(S[0, :], label='Source 1')
plt.plot(S[1, :], label='Source 2')
plt.plot(X[0, :], label='Antenna 1 Signal')
plt.plot(X[1, :], label='Antenna 2 Signal')
plt.plot(X[2, :], label='Antenna 3 Signal')
plt.plot(X[3, :], label='Antenna 4 Signal')

plt.ylabel('Amplitude')
plt.xlabel('Time')
plt.title('Source and Antenna Signals')
plt.legend()
plt.show()

plt.figure()
plt.plot(np.angle(TAU[0, 0, :]), label='Antenna 1, Source 1')
plt.plot(np.angle(TAU[1, 0, :]), label='Antenna 2, Source 1')
plt.plot(np.angle(TAU[2, 0, :]), label='Antenna 3, Source 1')
plt.xlabel('Time')
plt.ylabel('Phase')
plt.title('Phase of TAU')
plt.legend()
plt.show()
