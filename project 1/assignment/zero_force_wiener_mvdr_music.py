import numpy as np
import config_assignment
from src.channel_eq_funcs import  gendata_conv, make_X
import matplotlib.pyplot as plt
N = 500
P = 4
m = 2
sigma = 0.5

if N%2 != 0:
    raise ValueError("N must be even")
if P%2 != 0:
    raise ValueError("P must be even")

QPSK = np.array([1+1j, -1+1j, -1-1j, 1-1j])/np.sqrt(2)
s = np.zeros(N, dtype=complex)
for i in range(N):
    s[i] = QPSK[np.random.randint(4)]
print("shape of s", s.shape)    
S = np.zeros((m, N-1), dtype=complex)
for i in range(m):
    S[m-1-i, :] = s[m-1-i : N-m-i+2]
if not np.array_equal(S[0,1:], S[1, :-1]):
    raise ValueError("S is not equal to s")

h= np.zeros(P)
for i in range(P):
    if i<P/4:
        h[i] = 1
    elif i<P/2:
        h[i] = -1
    elif i<3*P/4:
        h[i] = 1
    else:
        h[i] = -1
H = np.zeros((m*P,m), dtype=complex)
for i in range(m):
        H[i*P:(i+1)*P, m-1-i] = h



X = H @ S
X = X + (np.random.normal(0, sigma, X.shape) + 1j*np.random.normal(0, sigma, X.shape))/np.sqrt(2)
show_plots = False
if show_plots:
    plt.figure()
    for i in range(N-3):
        plt.plot(np.real(X[:, i]), label=f'Real part, sample {i}')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.title('Received signal')
    plt.legend()
    plt.show()

delay = 1

e = np.zeros((m), dtype=complex)
e[delay] = 1
h_col =  H @ e

w_zf = H @ np.linalg.pinv(H.conj().T @ H) @ e
w_wiener = np.linalg.pinv(H @ H.conj().T + sigma**2 * np.eye(m*P)) @ h_col

# Calculate MVDR beamformer
Rx = X @ X.conj().T / N
Rn = np.eye(2) * sigma**2 #np.eye(P/m) * sigma**2
Rn_inv = np.linalg.pinv(Rn)
R_inv = np.linalg.pinv(Rx)
print("shape of R_inv", R_inv.shape)
print("shape of H", H.shape)
print("shape of Rn_inv", Rn_inv.shape)
print("shape of e", e.shape)
w_mvdr = R_inv @ H @ Rn_inv @ e / P

# # Calculate MUSIC beamformer
# Rx = X @ X.conj().T / N
# _, V = np.linalg.eig(Rx)
# Vn = V[:, m:]
# Hn = np.kron(np.eye(P), Vn)
# Hn = Hn.conj().T

# w_music = np.linalg.pinv(Hn @ Hn.conj().T) @ e

print("shape of X", X.shape)
print("shape of S", S.shape)
print("shape of H", H.shape)
print("shape of h_col", h_col.shape)
print("shape of w_zf", w_zf.shape)
print("shape of w_wiener", w_wiener.shape)
print("shape of w_mvdr", w_mvdr.shape)
#print("shape of w_music", w_music.shape)

s_est_zf = w_zf @ X
s_est_wiener = w_wiener @ X
s_est_mvdr = w_mvdr @ X
#s_est_music = w_music @ X
print("shape of s_est_zf", s_est_zf.shape)
print("shape of s_est_wiener", s_est_wiener.shape)
print("shape of s_est_mvdr", s_est_mvdr.shape)
#print("shape of s_est_music", s_est_music.shape)

print(f"Error ZF: {np.linalg.norm(s_est_zf - s[delay:+delay+ N-1])}")  
print(f"Error Wiener: {np.linalg.norm(s_est_wiener - s[delay:delay+N-1])}")
print(f"Error MVDR: {np.linalg.norm(s_est_mvdr - s[delay:delay+N-1])}")
#print(f"Error MUSIC: {np.linalg.norm(s_est_music - s[delay:delay+N-1])}")

plt.figure()
plt.scatter(np.real(s_est_zf), np.imag(s_est_zf), label='ZF')
plt.scatter(np.real(s_est_wiener), np.imag(s_est_wiener), label='Wiener')
plt.scatter(np.real(s_est_mvdr), np.imag(s_est_mvdr), label='MVDR')
#plt.scatter(np.real(s_est_music), np.imag(s_est_music), label='MUSIC')
plt.scatter(np.real(s), np.imag(s), label='s')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.title('Complex Symbols retrieved by ZF, Wiener, MVDR and MUSIC filters')
plt.legend()
plt.show()

