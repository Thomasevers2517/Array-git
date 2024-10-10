import matplotlib.pyplot as plt
import config_assignment
from src.data_generation.generate_data import generate_data
import numpy as np

list_num_samples = [20, 40]
list_num_antennas = [5, 10]
list_theta = [[-20, 30], [-20, -25]]
list_f = [[0.1, 0.3], [0.1, 0.105]]
snr = [20, 20]
colors = ['b', 'r', 'c', 'm' ]
markers = ['o', 'v', 's', 'D']
svd_index = [1,2,3,4,5,6,7,8,9,10]
color_labels = [f'N = {list_num_samples[0]}   M = {list_num_antennas[0]}', f'N = {list_num_samples[1]}   M = {list_num_antennas[0]}', f'N = {list_num_samples[0]}   M = {list_num_antennas[1]}', f'N = {list_num_samples[1]}   M = {list_num_antennas[1]}']
marker_labels = [f'f = {list_f[0]}   theta = {list_theta[0]}', f'f = {list_f[1]}   theta = {list_theta[0]}', f'f = {list_f[0]}   theta = {list_theta[1]}', f'f = {list_f[1]}   theta = {list_theta[1]}']
index =0
for s,num_sample in enumerate(list_num_samples):
    for a,num_antenna in enumerate(list_num_antennas):
        for i,theta in enumerate(list_theta):
            for j,f in enumerate(list_f):
                X,_,_,_ = generate_data(num_antenna, num_sample, theta, f, snr, delta=0.5)
                X_singular_values = np.linalg.svd(X, compute_uv=False)
                
                plt.scatter(np.ones(len(X_singular_values))*index, X_singular_values, color=colors[s*2 +a], marker = markers[2*i+j] , label=f"__nolegend__", )
                index += 1

# Creating custom legend handles
color_handles = [plt.Line2D([0], [0], color=color, lw=4) for color in colors]
marker_handles = [plt.Line2D([0], [0], marker=marker, color='w', markerfacecolor='k', markersize=10) for marker in markers]

# Adding custom legends
first_legend = plt.legend(color_handles, color_labels, loc='center left', title="Colors")
plt.gca().add_artist(first_legend)  # Add the first legend back
plt.xlabel('Index')
plt.ylabel('Singular Values')
plt.yscale("log")
plt.legend(marker_handles, marker_labels, loc='center right', title="Markers", )
plt.show()
