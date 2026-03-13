import numpy as np
from scipy.signal import spectrogram


def iq_to_spectrogram(iq_signal):

    I = np.real(iq_signal)
    Q = np.imag(iq_signal)

    complex_signal = I + 1j*Q

    f,t,Sxx = spectrogram(
        complex_signal,
        nperseg=64,
        noverlap=32
    )

    Sxx = np.abs(Sxx)

    Sxx = np.log(Sxx + 1e-6)

    return Sxx
