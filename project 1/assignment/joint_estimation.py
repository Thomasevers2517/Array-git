import config_assignment
from src.data_generation.generate_data_khatri_rao import generate_data_khatri_rao
from src.joint_diagonalisation import joint_diag
import numpy as np

X = generate_data_khatri_rao(3, 500, [30, 50], [0.005, 0.002], [100, 100])

V, D = joint_diag(np.array(X), 1e-10)
print(V)