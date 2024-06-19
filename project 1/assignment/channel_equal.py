import numpy as np
import matplotlib.pyplot as plt
import config_assignment
from src.channel_eq_funcs import  gendata_conv, make_X
N = 500
P = 4
sigma = 0.0

QPSK = np.array([1+1j, -1+1j, -1-1j, 1-1j])/np.sqrt(2)
s = np.zeros(N, dtype=complex)
for i in range(N):
    s[i] = QPSK[np.random.randint(4)]
    
x = gendata_conv(s, P, N, sigma)
plt.figure()
plt.plot(np.real(x), label='Real part')
plt.plot(np.imag(x), label='Imaginary part')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.title('Received signal')
plt.legend()
plt.show()


X = make_X(x, P, N)
# Print the rank of X
print(f"Rank of X: {np.linalg.matrix_rank(X)}")

# Doubling P does not increase the rank seeing as the rows are linearly dependent.  
# -1 +1j and 1 -1j, passed through the filter are linearly dependent, same as 1 +1j and -1 -1j.