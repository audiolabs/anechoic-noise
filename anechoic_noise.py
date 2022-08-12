import os

import numpy as np
import librosa as lr
import soundfile as sf
import matplotlib.pyplot as plt

def normalize(sig: np.ndarray, peak_db: float) -> np.ndarray:
    """
    Normalize input signal to specified peak amplitude

    :param np.ndarray sig: Single-channel input signal
    :param float fs: Sample rate in [Hz]
    :param float tau: Averaging time constant [s]
    :return: Output signal, e.g. smoothed amplitude envelope
    :rtype: np.ndarray
    """
    sig /= np.max(np.abs(sig))
    sig *= np.power(10, peak_db * 0.05)
    return sig


def leaky_integrator(
    signal_in: np.ndarray = np.zeros(100),
    fs: int = 48000,
    tau: float = 0.1,
) -> np.ndarray:
    """
    Leaky integrator for envelope following, smoothing only applied to
    decaying signal amplitudes

    :param np.ndarray signal_in: Single-channel input signal
    :param float fs: Sample rate in [Hz]
    :param float tau: Averaging time constant [s]
    :return: Output signal, e.g. smoothed amplitude envelope
    :rtype: np.ndarray
    """
    if signal_in.ndim != 1:
        raise ValueError("Input must be single-channel.")

    # averaging coefficient from time constant
    alpha = 1 - np.exp(-1/(tau*fs))

    signal_out = np.zeros(signal_in.shape)
    signal_out[0] = alpha * signal_in[0]
    for i in range(1, signal_out.shape[0]):
        if signal_in[i] > signal_out[i-1]:
            signal_out[i] = signal_in[i]
        else:
            signal_out[i] = alpha * signal_in[i] + (1 - alpha) * signal_out[i-1]

    return signal_out


class NonStationaryInterference:
    """
    Generator class for generating random mixtures

    Attributes
    ----------
    fs : float
        Sample rate [Hz]
    data_dir : str
        Relative path to the `wavs` directory of the dataset
        (https://zenodo.org/record/6974033)

    Methods
    -------
    generate_signal(self, duration: float = 4.0, tau: float = 0.01):
        Generates random mixture
    colored_noise(self, length: float = 3, beta: float = 1.0):
        Generates non-stationary, colored noise signal with the envelope
        of a random sample mixture
    get_random_sample(self):
        Retrieves as random sample from the dataset
    write_wav(self, out_fp, sig):
        Writes a generated signal to wav file
    """

    def __init__(
        self,
        fs: int = 48000,
        data_dir: str ='./AID_folder/wavs/',
        sample_density: int = 2,
        segment_peak: float = -6.0,
        beta: float = 0.0,
        tau: float = 0.05,
        ) -> None:
        """
        Constructs all the necessary attributes for the 
        NonStationaryInterference object.

        Parameters
        ----------
            fs : float
                Sample rate [Hz]
            data_dir : str
                Relative path to the `wavs` directory of the dataset
                (https://zenodo.org/record/6974033)
        """
        self.fs = fs
        self.data_dir = data_dir

        # get list of all samples in dataset
        self.file_list = [x for x in os.listdir(data_dir) if x.endswith('.wav')]
        self.n_files = len(self.file_list)


    def generate_signal(
        self,
        duration: float = 4.0,
        sample_density: int = 2,
        segment_peak: float = -6.0,
        beta: float = 0.0,
        tau: float = 0.01,
    ) -> np.ndarray:
        """
        Generates a single-channel signal of specified length
        
        :param float duration: Mixture signal length [s]
        :param int sample_density: Number of samples concurrently playing (>=1)
        :param float segment_peak: Peak amplitude of random mixture (<=0.) [dBFS]
        :param float beta: Noise PSD exponent; 0-white, 1-pink, 2-brown, etc.
        :param float tau: Leaky integrator time constant [s]
        """
        len_out = int(np.round(self.fs * duration))
        out_sig = np.zeros(len_out)

        noise = self.colored_noise(duration=duration, beta=beta)

        for _ in range(sample_density):
            start_ind = 0
            while start_ind < len_out:
                sample = self.get_random_sample()
                len_sample = sample.shape[0]

                # check if next samples fit in output
                overlap = max(start_ind + len_sample - len_out, 0)
                if overlap > 0:
                    offset = np.random.randint(overlap)
                    sample = sample[offset:offset+len_sample-overlap]

                # add sample
                out_sig[start_ind:start_ind+len_sample] += sample
                start_ind += len_sample

        # normalize
        out_sig /= np.max(np.abs(out_sig))
        out_sig *= np.power(10, segment_peak * 0.05)

        # generate amplitude-modulated colored noise
        env = leaky_integrator(np.abs(out_sig), self.fs, tau)
        noise *= env
        noise /= np.max(np.abs(noise))
        noise *= np.power(10, segment_peak * 0.05)

        return (out_sig, noise, env)


    def colored_noise(
        self,
        duration: float = 4.0,
        beta: float = 1.0
        ) -> np.ndarray:
        """
        Returns colored noise vector, called from self.generate_signal()
        
        :param float duration: Noise signal length [s]
        :param float beta: Noise PSD exponent; 0-white, 1-pink, 2-brown, etc.
        """
        half_length = int(duration * self.fs / 2 + 1)
        f = np.linspace(0, self.fs//2, num=half_length)
        phase = np.exp(1j * 2 * np.pi * np.random.uniform(size=half_length))
        mag = np.ones(half_length)

        # noise magnitude slope
        mag[1:] /= np.sqrt(np.power(f[1:], beta))
        mag[0] = 0

        # construct conjugate symmetric output spectrum
        cplx = mag * phase
        cplx = np.append(cplx, np.conj(np.flip(cplx[1:-1])))
        # get time-domain signal
        sig = np.real(np.fft.ifft(cplx))
        sig -= np.mean(sig)
        sig /= np.sqrt(np.mean(sig**2))
        return sig


    def get_random_sample(self):
        """Retrieves as random sample from the data set"""
        rand_ind = np.random.randint(self.n_files)
        filename = f"{self.data_dir}{self.file_list[rand_ind]}"
        sample = sf.read(filename)[0]
        return sample


    def write_wav(self, fp, sig) -> None:
        """Writes a generated signal to wav file"""
        print(f"Writing file: {fp}")
        sf.write(fp, sig, samplerate=self.fs)


if __name__ == "__main__":

    nsi = NonStationaryInterference(data_dir="./AID/wavs/")

    num_mixtures = 10

    for i in range(num_mixtures):
        sig, noise, env = nsi.generate_signal(duration=8)
        # write sample mixture
        out_file = f"./output/mixture_{i:02d}.wav"
        nsi.write_wav(out_file, sig)
        # write colored noise signal
        out_file = f"./output/noise_{i:02d}.wav"
        nsi.write_wav(out_file, noise)
