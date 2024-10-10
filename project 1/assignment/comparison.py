
import config_assignment
from src.data_generation.generate_data_khatri_rao import generate_data_khatri_rao
from src.data_generation.generate_data import generate_data
from src.espirit.joint_diagonalisation import joint_diag
from src.espirit.joint_estimation import joint
from src.espirit.espirit import espirit
from src.espirit.espirit import espirit_freq
import numpy as np
from matplotlib import pyplot as plt

# Old config of submitted assignment
# d=2
# M =3
# N = 20
# m =10
# P =100
# delta = 0.2
# freqs = np.array([0.1, 0.12])
# thetas = np.array([-20, 30])
d = 2
M = 3
N = 20
m = 6
P = 2
delta = 0.5
thetas = [-20, 30]
freqs = [0.1, 0.12]
#SNRs = [100, 100]

SNRs = [i*4 for i in range(0, 16) ]
SNRs.append(100)
print("SNRS:", SNRs)

predictions = {}
predictions['joint_freq'] = {}
predictions['joint_angle'] = {}
predictions['angle'] = {}
predictions['freq'] = {}
num =150

for snr in SNRs:
    predictions['angle'][snr] = np.zeros((num, d))
    predictions['freq'][snr] = np.zeros((num, d))
    predictions['joint_freq'][snr] = np.zeros((num, d))
    predictions['joint_angle'][snr] = np.zeros((num, d))
    for i in range(num):
        X_rao,_,_ = generate_data_khatri_rao(M=M, N=N, theta_degrees=thetas, f=freqs, snr_db=[snr,snr], m=m, P=P, delta=delta)

        X_normal, _, _, _ = generate_data(M=M, N=N, theta_degrees=thetas, f=freqs, snr_db=[snr,snr], delta=delta)

        joint_pred_freqs, joint_pred_angles = joint(X_rao, d, m, delta, P)
        
        pred_freq = espirit_freq(X_normal, d, m, 1)
        pred_angle = espirit(X_normal, d, delta)
        
        joint_pred_freqs.sort()
        joint_pred_angles.sort()
        pred_freq.sort()
        pred_angle.sort()
    
        predictions['joint_freq'][snr][i] = np.array(joint_pred_freqs)
        predictions['joint_angle'][snr][i] = np.array(joint_pred_angles)    
        predictions['angle'][snr][i] = np.array(pred_angle)
        predictions['freq'][snr][i] = np.array(pred_freq)

 
print(predictions['joint_freq'][8].shape) 
print(predictions['joint_angle'][8].shape)
print(predictions['angle'][8].shape)
print(predictions['freq'][8].shape)
print(predictions['angle'][snr][:,:])
  
plt.figure()

plt.subplot(4,2,1)
plt.scatter(SNRs, [np.mean(predictions['joint_freq'][snr][:,1]) for snr in SNRs], label='Joint freq mean')
plt.scatter(SNRs, [np.mean(predictions['joint_freq'][snr][:,0]) for snr in SNRs], label='joint freq mean')
plt.yscale('log')
plt.xlabel('SNR')
plt.legend()
plt.title('Joint freq mean')


plt.subplot(4,2,2)
plt.scatter(SNRs, [np.var(predictions['joint_freq'][snr][:,1]) for snr in SNRs], label='Joint freq variance')
plt.scatter(SNRs, [np.var(predictions['joint_freq'][snr][:,0]) for snr in SNRs], label='joint freq variance')
plt.yscale('log')
plt.xlabel('SNR')
plt.legend()
plt.title('Joint freq variance')


plt.subplot(4,2,3)
plt.scatter(SNRs, [np.mean(predictions['joint_angle'][snr][:,1]) for snr in SNRs], label='Joint angle mean')
plt.scatter(SNRs, [np.mean(predictions['joint_angle'][snr][:,0]) for snr in SNRs], label='joint angle mean')
plt.xlabel('SNR')
plt.legend()
plt.title('Joint angle mean')


plt.subplot(4,2,4)
plt.scatter(SNRs, [np.var(predictions['joint_angle'][snr][:,1]) for snr in SNRs], label='Joint angle variance')
plt.scatter(SNRs, [np.var(predictions['joint_angle'][snr][:,0]) for snr in SNRs], label='joint angle variance')
plt.yscale('log')
plt.xlabel('SNR')
plt.legend()
plt.title('Joint angle variance')


plt.subplot(4,2,5)
plt.scatter(SNRs, [np.mean(predictions['freq'][snr][:,1]) for snr in SNRs], label='freq mean')
plt.scatter(SNRs, [np.mean(predictions['freq'][snr][:,0]) for snr in SNRs], label='freq mean')
plt.yscale('log')
plt.xlabel('SNR')
plt.legend()
plt.title('freq mean')


plt.subplot(4,2,6)
plt.scatter(SNRs, [np.var(predictions['freq'][snr][:,1]) for snr in SNRs], label='freq variance')
plt.scatter(SNRs, [np.var(predictions['freq'][snr][:,0]) for snr in SNRs], label='freq variance')
plt.yscale('log')
plt.xlabel('SNR')
plt.legend()
plt.title('freq variance')

plt.subplot(4,2,7)
plt.scatter(SNRs, [np.mean(predictions['angle'][snr][:,1]) for snr in SNRs], label='angle mean')
plt.scatter(SNRs, [np.mean(predictions['angle'][snr][:,0]) for snr in SNRs], label='angle mean')
# plt.yscale('log')
plt.xlabel('SNR')
plt.legend()
plt.title('angle mean')


plt.subplot(4,2,8)
plt.scatter(SNRs, [np.var(predictions['angle'][snr][:,1]) for snr in SNRs], label='angle variance')
plt.scatter(SNRs, [np.var(predictions['angle'][snr][:,0]) for snr in SNRs], label='angle variance')
plt.yscale('log')
plt.xlabel('SNR')
plt.legend()
plt.title('angle variance')

plt.show()






    