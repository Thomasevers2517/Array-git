import matplotlib.pyplot as plt
import config_assignment
from src.data_generation.generate_data import generate_data
import numpy as np

list_num_samples = [20, 40]
list_num_antennas = [5, 10]
list_theta = [[-20, 30], [-20, -25]]
list_f = [[0.1, 0.3], [0.1, 0.105]]
snr = [20, 20]
colors = plt.cm.viridis(np.linspace(0, 1, len(list_num_samples)*len(list_num_antennas)*len(list_theta)*len(list_f)))
index =0
for num_sample in list_num_samples:
    for num_antenna in list_num_antennas:
        for theta in list_theta:
            for f in list_f:
                X,_,_,_ = generate_data(num_antenna, num_sample, 0.5, theta, f, snr)
                X_singular_values = np.linalg.svd(X, compute_uv=False)
                plt.scatter(np.ones(len(X_singular_values))*index, X_singular_values, color='blue', label=f"N={num_sample}, M={num_antenna}, theta={theta}, f={f}")
                index += 1
plt.legend()      
plt.show()