import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# Generate a sample signal at 0.5 seconds with two frequencies
fs = 16000  # Sample rate
t = np.linspace(0.5, 1, (fs//2), endpoint=False)  # 1 second of audio
print("t: ", t)
frequency = 440  # A4 note
x = np.zeros(fs)
x[8000:] = 0.5 * np.sin(2 * np.pi * frequency * t)  # A4 note
frequency = 880  # A5 note
x[8000:] += 0.5 * np.sin(2 * np.pi * frequency * t)  # A5 note

# Add some noise
noise = np.random.normal(0, 0.1, x.shape)
x = x + noise

# Perform STFT
f, t, Zxx = signal.stft(x, fs, nperseg=256)

# Compute the power for each time frame
power = np.sum(np.abs(Zxx)**2, axis=0)

# Plot the power over time
plt.figure(figsize=(10, 6))
plt.plot(t, power)
plt.title('Power of STFT over time')
plt.xlabel('Time [s]')
plt.ylabel('Power')
plt.show()

# Determine a threshold for speech detection (this is an example, the actual threshold may vary)
threshold = 0.01

# Detect speech presence based on the power threshold
speech_presence = power > threshold

# Print the result
print("Speech detected at time indices:", np.where(speech_presence)[0])

# Optionally, visualize the speech detection
plt.figure(figsize=(10, 6))
plt.plot(t, power, label='Power')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.fill_between(t, 0, power, where=speech_presence, facecolor='green', alpha=0.5, label='Speech detected')
plt.title('Speech Detection based on STFT Power')
plt.xlabel('Time [s]')
plt.ylabel('Power')
plt.legend()
plt.show()
