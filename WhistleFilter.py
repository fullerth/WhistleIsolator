from scipy.signal import butter, lfilter
from scipy.io.wavfile import read as wavfile_read, write as wavfile_write
from scipy.signal import correlate


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.rcParams['agg.path.chunksize'] = 10000

    fs, data = wavfile_read('gameplay.wav')    
    filtered_fs, filtered_data = wavfile_read('filtered_output.wav')
    whistle_fs, whistle_data = wavfile_read('clipped_whistle.wav')

    #plt.plot(Time, data)
    #plt.plot(Time, filtered_data)
    #plt.plot(whistle_data)

    #plt.show()

    print("fs: {0}".format(fs))

    # Desired cutoff frequencies (in Hz).
    lowcut = 1800.0
    highcut = 22000.0

    y = butter_bandpass_filter(data, lowcut, highcut, fs, order=6)

    wavfile_write('filtered_output.wav', fs, y)

    correlated_filtered = correlate(y, whistle_data) 

    Time = np.linspace(0, correlated_filtered.shape[0]/fs, 
        num = correlated_filtered.shape[0])

    plt.plot(Time, correlated_filtered)
    plt.show()


