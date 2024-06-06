import config_assignment
from src.data_generation.generate_data_khatri_rao import generate_data_khatri_rao
from src.joint_diagonalisation import joint_diag
import numpy as np
d=2
M =7
m =5
X = generate_data_khatri_rao(M, 500, [30, 50], [0.05, 0.002], [100, 100], m=m)
import scipy.io


print(X.shape)
U, S,V = np.linalg.svd(X)
U = U[:, :d]
S = S[:d]
V = V[:d, :]
print("U: ", U.shape)
print("S: ", S)
print("V: ", V.shape)

U_x = U[:M, :]
U_x_inv = np.linalg.pinv(U_x)
print("U_x: ", U_x.shape)
print("U_x_inv: ", U_x_inv.shape)

M_mat = np.zeros((d, d*(m)), dtype=complex)
for i in range(m-1):
    M_mat[ :, i*d:(i+1)*d] =U_x_inv @ U[(i)*M:(i+1)*M, :] 
   
print("M_mat: ", M_mat.shape)

scipy.io.savemat('data.mat', {'M': M_mat})

V, D = joint_diag(M_mat, 1e-10)
print("V: ", V.shape)
print("M_mat: ", M_mat.shape)
for i in range(m-1):
    result = V @ M_mat[:,i*d:(1+i)*d] @ np.linalg.inv(V)
    print(np.angle(result))
