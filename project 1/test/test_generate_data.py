import config_test

from src.data_generation.generate_data import generate_data
import numpy as np

d=2
delta =0.5
freqs = np.array([0.1, 0.12])
angles = np.array([-20, 30])

X, S, A, NOISE = generate_data(M=4, N=105, theta_degrees= angles, f =freqs, snr_db=[20, 20], delta = delta)


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
plt.plot(np.angle(A[0, :]), label='Antenna 1, Source 1')
plt.plot(np.angle(A[1, :]), label='Antenna 2, Source 1')
plt.plot(np.angle(A[2, :]), label='Antenna 3, Source 1')
plt.xlabel('Source')
plt.ylabel('Phase')
plt.title('Phase of TAU')
plt.legend()
plt.show()
