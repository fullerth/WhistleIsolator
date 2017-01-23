import numpy as np

import librosa
import librosa.display

import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')

audio_path = 'gameplay.wav'

###### Mel Spectrum ######

y, sr = librosa.load(audio_path, sr=None)

# # Let's make and display a mel-scaled power (energy-squared) spectrogram
# S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

# # Convert to log scale (dB). We'll use the peak power as reference.
# log_S = librosa.logamplitude(S, ref_power=np.max)

# # Make a new figure
# plt.figure(figsize=(12,4))

# # Display the spectrogram on a mel scale
# # sample rate and hop length parameters are used to render the time axis
# librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

# # Put a descriptive title on the plot
# plt.title('mel power spectrogram')

# # draw a color bar
# plt.colorbar(format='%+02.0f dB')

# # Make the figure layout compact
# plt.tight_layout()

# plt.show()


###### Harmonic Percussive Spectral Split ######
y_harmonic, y_percussive = librosa.effects.hpss(y)

# What do the spectrograms look like?
# Let's make and display a mel-scaled power (energy-squared) spectrogram
S_harmonic   = librosa.feature.melspectrogram(y_harmonic, sr=sr)
S_percussive = librosa.feature.melspectrogram(y_percussive, sr=sr)
S_subtractive = librosa.feature.melspectrogram(y_harmonic - y_percussive, sr=sr)



# Convert to log scale (dB). We'll use the peak power as reference.
log_Sh = librosa.logamplitude(S_harmonic, ref_power=np.max)
log_Sp = librosa.logamplitude(S_percussive, ref_power=np.max)
log_Ss = librosa.logamplitude(S_subtractive, ref_power=np.max)

# Make a new figure
plt.figure(figsize=(12,9))

plt.subplot(2,1,1)
# Display the spectrogram on a mel scale
librosa.display.specshow(log_Sh, sr=sr, y_axis='mel')

# Put a descriptive title on the plot
plt.title('mel power spectrogram (Harmonic)')

# draw a color bar
plt.colorbar(format='%+02.0f dB')

plt.subplot(2,1,2)
librosa.display.specshow(log_Sp, sr=sr, x_axis='time', y_axis='mel')

# Put a descriptive title on the plot
plt.title('mel power spectrogram (Percussive)')

# draw a color bar
plt.colorbar(format='%+02.0f dB')

plt.subplot(3,1,3)
librosa.display.specshow(log_Ss, sr=sr, x_axis='time', y_axis='mel')

# Put a descriptive title on the plot
plt.title('mel power spectrogram (Subtraction)')

# draw a color bar
plt.colorbar(format='%+02.0f dB')

# Make the figure layout compact
# plt.tight_layout()

plt.show()

