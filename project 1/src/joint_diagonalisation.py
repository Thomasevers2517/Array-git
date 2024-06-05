import numpy as np
from scipy.linalg import eig
def joint_diag(A, jthresh=1.0e-8):
    """
    Joint approximate diagonalization of n (complex) matrices of size m*m stored in the
    m*nm matrix A by minimization of a joint diagonality criterion

    Parameters:
    A (numpy.ndarray): the m*nm matrix A is the concatenation of n matrices with size m by m.
    jthresh (float): threshold, an optional small number (default = 1.0e-8).

    Returns:
    tuple: V is an m*m unitary matrix. D is the joint diagonalized matrix.
    """

    m, nm = A.shape
    V = np.eye(m)
    encore = True

    # Initialize constants
    B = np.array([[1, 0, 0], [0, 1, 1], [0, -1j, 1j]])
    Bt = B.T

    while encore:
        encore = False
        for p in range(m-1):
            Ip = np.arange(p, nm, m)
            for q in range(p+1, m):
                Iq = np.arange(q, nm, m)

                g = np.vstack([A[p, Ip] - A[q, Iq], A[p, Iq], A[q, Ip]])
                vcp, D = eig(np.real(np.dot(np.dot(B, np.dot(g, g.T)), Bt)))
                print("vcp", vcp)   
                print("D", D)
                la = np.sort(np.diag(D))
                K = np.argsort(np.diag(D))
                angles = vcp[:, K[-1]]
                
                if angles[0] < 0:
                    angles = -angles

                c = np.sqrt(0.5 + angles[0] / 2)
                s = 0.5 * (angles[1] - 1j * angles[2]) / c

                if abs(s) > jthresh:
                    encore = True
                    pair = [p, q]
                    G = np.array([[c, -np.conj(s)], [s, c]])
                    V[:, pair] = np.dot(V[:, pair], G)
                    A[pair, :] = np.dot(G.T, A[pair, :])
                    A[:, np.r_[Ip, Iq]] = np.hstack([c * A[:, Ip] + s * A[:, Iq], -np.conj(s) * A[:, Ip] + c * A[:, Iq]])

    return V, A
