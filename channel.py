import numpy as np
from scipy import fft
from scipy.signal import firls, lfilter, minimum_phase
from scipy.io.wavfile import read, write
import os
from datetime import datetime
import warnings


fs = 44100
def save(signal, directory: str="channel_outputs/", filename: str=""):
    filename += "no_noise" + datetime.now().strftime("%m%d-%H%M%S%f") + ".wav"
    write(os.path.join(directory, filename), fs, signal.astype(np.float32))
    print(f"Saved output to {directory}{filename}")

_default_response = np.array([0.014, 0.019, 0.02, 0.043, 0.07, 0.103, 0.137, 0.126, 0.122, 0.227, 0.334, 0.313, 0.173, 0.183, 0.158, 0.163, 0.061, 0.06, 0.11, 0.162, 0.152, 0.076, 0.074, 0.053, 0.048, 0.032, 0.029, 0.022, 0.018, 0.01])
_default_freqs = np.logspace(np.log10(200), np.log10(5000), 30)

def _delay(signal, time):
    delay_idx = min(int(time * fs), signal.size)
    output = np.roll(signal, delay_idx)
    output[:delay_idx] = 0
    return output
def _smooth_ramp(x):
    #infinitely smooth sigmoid (antiderivative of bump function)
    output = np.ones_like(x)/2
    mask = x<1
    output[mask] = 1/(1+np.exp(4*x[mask]/(x[mask]**2-1)))-1/2
    return output
def _clip(signal, knee=.5):
    #interpolate between identity and smooth ramp
    k = knee/(1-knee)
    shifted_ramp = lambda x: 2*_smooth_ramp((x-k)/2)+k
    t = signal*(1+k)
    output = signal.copy()
    pos = signal>knee
    neg = signal<-knee
    output[pos] = shifted_ramp(t[pos])/(1+k)
    output[neg] = -shifted_ramp(-t[neg])/(1+k)
    return output
def _load_wavs(dir_name):
    directory = dir_name
    loaded = []
    for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".wav"):
                loaded.append(read(os.path.join(directory, filename))[1])
    return loaded

class Channel:
    def __init__(self, measured_freqs=None, response=None, claps=None, talking=None):
        if measured_freqs is None:
            measured_freqs = _default_freqs
        if response is None:
            response = _default_response
            
        #generate filters
        bands = np.pad(measured_freqs, 1, constant_values=(0,fs/2))
        desired = np.pad(response, 1)
        # desired /= np.max(desired)
        self.system_filter = firls(101, bands, desired, fs=fs)
        self.system_filter = minimum_phase(self.system_filter)
        self.noise_filter = firls(101, bands, np.sqrt(desired), fs=fs)
        self.noise_filter = minimum_phase(self.noise_filter)
        self.noise_filter /= np.linalg.norm(self.noise_filter)

        if claps is None:
            claps = _load_wavs("claps/")
        self.claps_list = [x/np.max(x) for x in claps]
        if talking is None:
            talking = _load_wavs("talking/")
        self.talking_list = [x/np.mean(x**2) for x in talking]

        self.rng = np.random.default_rng()
    
    def _clap_noise(self, n, amp, avg_wait, shape=1, min_wait=.15):
        indices = [] #when claps occur
        i = 0
        while True:
            delay = self.rng.gamma(shape, avg_wait) + min_wait
            i += (np.floor(delay*fs).astype(np.int_)).item()
            if i >= n:
                break
            indices.append(i)
        repeats = 1 + ((len(indices)-1)//len(self.claps_list)) #number of times to reuse claps
        clap_idx = self.rng.permutation(list(range(len(self.claps_list)))*repeats)[:len(indices)]
        output = np.zeros(n)
        #insert claps
        for j,c in zip(indices, clap_idx):
            duration = min(self.claps_list[c].size, n-j)
            output[j:j+duration] += self.claps_list[c][:duration]*amp
        return output
    def _talking_noise(self, n, power=0.1):
        #choose enough clips to exceed desired length
        len_list = [x.size for x in self.talking_list]
        length = 0
        chosen = []
        while length<n:
            idx = self.rng.integers(len(self.talking_list))
            length += len_list[idx]
            chosen.append(self.talking_list[idx])
        #randomly choose segment of size n
        long = np.concatenate(chosen)
        shift = self.rng.integers(long.size-n+1)
        output = long[shift:shift+n]*np.sqrt(power)
        return output.copy()
    def _generate_loudness(self, n, power=1.0, cutoff_f=2.0)-> np.ndarray:
        new_len = fft.next_fast_len(16*n, True)
        cutoff = int(cutoff_f * new_len // fs)
        output_fft = self.rng.normal(size=cutoff)
        output = fft.irfft(output_fft, new_len)[:n].copy()
        actual_power = np.mean(output**2)
        output *= np.sqrt(power/actual_power)
        return output
    def _noise(self, signal, noise_var, avg_wait, clap_amp, talking_pow, gain_variation=1/900, gain=None, speaker_noise_var=0):
        delay0 = self.rng.uniform(low=0.15, high=0.35)
        signal = _delay(signal, delay0)
        signal = lfilter(self.system_filter, 1, signal)
        noise = self.rng.normal(loc=0, scale=np.sqrt(noise_var), size=signal.shape)
        noise = lfilter(self.noise_filter, 1, noise)
        if speaker_noise_var > 0:
            speaker_noise = self.rng.normal(loc=0, scale=np.sqrt(speaker_noise_var), size=signal.shape)
            speaker_noise = lfilter(self.system_filter, 1, speaker_noise)
            noise += _delay(speaker_noise, delay0)
        if gain is None:
            gain = self.rng.uniform(low=0.9, high=1.1)
        signal *= gain
        if avg_wait is not None:
            noise += self._clap_noise(signal.size, clap_amp, avg_wait)
        if talking_pow is not None:
            noise += _delay(self._talking_noise(signal.size, talking_pow), delay0)
        signal += noise
        signal *= np.exp(self._generate_loudness(signal.size, gain_variation))
        return _delay(_clip(signal), self.rng.uniform(low=0.15, high=0.35))
    def transmit(self, signal:np.ndarray, alt: bool=False, clap: bool=False, talking: bool=False, save_wav: bool=True, directory: str="channel_outputs/", filename: str="") -> np.ndarray:
        """Transmit a signal through the channel.
        
        Input variables:
            - signal:     1-D array of floats representing the input signal
            - alt:        if True, apply an alternate noise level
            - clap:       if True, add clapping interference
            - talking:    if True, add talking interference
            - save_wav:   if True, save the output to a .wav file
            - directory:  path to output directory for saved .wav file
            - filename:   optional base filename to prepend to output file
            
        Output variables:
            - received:   1-D array of floats representing the received signal
        """
        if not isinstance(signal, np.ndarray):
            raise TypeError(f"Input is of type {type(signal).__name__}, but should be a NumPy array.")
        if signal.ndim != 1:
            raise ValueError(f"Invalid input shape {signal.shape}.")
        if not np.issubdtype(signal.dtype, np.floating):
            raise ValueError(f"Signal has dtype {signal.dtype}, but should be floating point and real.")
        input_max_idx = np.argmax(np.abs(signal))
        if np.abs(signal[input_max_idx]) > 1:
            warnings.warn(f"Input contains large value {signal[input_max_idx]} at {input_max_idx}, which may cause clipping.", warnings.RuntimeWarning)
        received = self._noise(signal, 0.0005, 0.5 if clap else None, 2, 0.0005 if talking else None, gain=None, speaker_noise_var=0.5 if alt else 0)
        if save_wav:
            filename += ("_clap" if clap else "") + ("_talk" if talking else "")
            filename = filename + datetime.now().strftime("%m%d-%H%M%S%f")  + ".wav"
            write(os.path.join(directory, filename), fs, received.astype(np.float32))
            print(f"Saved output to {directory}{filename}")
        return received






            
