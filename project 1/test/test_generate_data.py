import config_test

from src.data_generation.generate_data import generate_data
import numpy as np

X, S, TAU, NOISE = generate_data(3, 500, 100000, theta=[0.5], f=[0.005], snr=[20000])

print("X: ", X)
import matplotlib.pyplot as plt
plt.figure()
plt.plot(S[0, :], label='Source 1')
plt.plot(X[0, :], label='Antenna 1 Signal')
plt.plot(X[1, :], label='Antenna 2 Signal')
plt.plot(X[2, :], label='Antenna 3 Signal')
plt.ylabel('Amplitude')
plt.xlabel('Time')
plt.title('Source and Antenna Signals')
plt.legend()
plt.show()

plt.figure()
plt.plot(np.angle(TAU[0, 0, :]), label='Antenna 1')
plt.plot(np.angle(TAU[1, 0, :]), label='Antenna 2')
plt.plot(np.angle(TAU[2, 0, :]), label='Antenna 3')
plt.xlabel('Time')
plt.ylabel('Phase')
plt.title('Phase of TAU')
plt.legend()
plt.show()
