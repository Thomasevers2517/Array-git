import numpy as np
from scipy.linalg import eig

def joint_diag(A, jthresh=1.0e-8):
    """
    Joint approximate diagonalization of n (complex) matrices of size m*m stored in the
    m*nm matrix A by minimization of a joint diagonality criterion
    
    Parameters:
    A : numpy.ndarray
        The m*nm matrix A is the concatenation of n matrices with size m by m.
        We denote A = [ A1 A2 .... An ]
    jthresh : float, optional
        Threshold for the algorithm (default is 1.0e-8).
        
    Returns:
    V : numpy.ndarray
        An m*m unitary matrix.
    D : numpy.ndarray
        Collection of diagonal matrices if A1, ..., An are exactly jointly unitarily diagonalizable.
    """
    m, nm = A.shape

    # Better declare the variables used in the loop
    B = np.array([[1, 0, 0], [0, 1, 1], [0, -1j, 1j]])
    Bt = B.T
    g = np.zeros((3, nm), dtype=complex)
    G = np.zeros((2, 2), dtype=complex)
    vcp = np.zeros((3, 3), dtype=complex)
    D = np.zeros((3, 3), dtype=complex)
    la = np.zeros(3)
    angles = np.zeros(3)
    pair = np.zeros(2, dtype=int)

    # Init
    V = np.eye(m, dtype=complex)
    encore = True

    while encore:
        encore = False
        for p in range(m - 1):
            Ip = np.arange(p, nm, m)
            for q in range(p + 1, m):
                Iq = np.arange(q, nm, m)

                # Computing the Givens angles
                g = np.vstack((A[p, Ip] - A[q, Iq], A[p, Iq], A[q, Ip]))
                D, vcp = eig(np.real(B @ (g @ g.conj().T) @ Bt), right=True)
                # print("vcp: ", vcp)
                # print("D: ", D)
                diag_D = np.diag(D)
                la = np.sort(D)
                K = np.argsort(D)
                # print("la: ", la)
                # print("K: ", K)
                
                angles = vcp[:, K[2]]
                if angles[0] < 0:
                    angles = -angles
                c = np.sqrt(0.5 + angles[0] / 2)
                s = 0.5 * (angles[1] - 1j * angles[2]) / c

                if np.abs(s) > jthresh:  # updates matrices A and V by a Givens rotation
                    encore = True
                    pair = [p, q]
                    G = np.array([[c, -np.conj(s)], [s, c]])
                    V[:, pair] = V[:, pair] @ G
                    A[pair, :] = G.conj().T @ A[pair, :]
                    A[:, np.hstack((Ip, Iq))] = np.hstack((c * A[:, Ip] + s * A[:, Iq], -np.conj(s) * A[:, Ip] + c * A[:, Iq]))

    D = A

    return V, D

# Example usage
if __name__ == "__main__":
    # Example matrices A1, A2, ... concatenated in A
    m = 3
    n = 2
    A1 = np.random.rand(m, m) + 1j * np.random.rand(m, m)
    A2 = np.random.rand(m, m) + 1j * np.random.rand(m, m)
    A = np.hstack((A1, A2))
    
    V, D = joint_diag(A)
    print("V:", V)
    print("D:", D)
