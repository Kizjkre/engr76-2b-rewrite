import numpy as np
from scipy import fft
from scipy.signal import firls, lfilter, minimum_phase
from scipy.io.wavfile import read, write
import os
from datetime import datetime


fs = 44100
_default_response = np.array([0.010, 0.011, 0.008, 0.018, 0.024, 0.025, 0.054, 0.088, 0.129, 0.166, 0.152, 0.149, 0.275, 0.405, 0.379, 0.189, 0.185, 0.133, 0.121, 0.080, 0.073, 0.054, 0.045, 0.024, 0.018, 0.041, 0.061, 0.055, 0.013, 0.046])
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

class SimulatedChannel:
    def __init__(self, measured_freqs=None, response=None, claps=None, talking=None):
        #TODO: input validation

        if measured_freqs is None:
            measured_freqs = _default_freqs
        if response is None:
            response = _default_response
            
        #generate filters
        bands = np.pad(measured_freqs, 1, constant_values=(0,fs/2))
        desired = np.pad(response, 1)
        desired /= np.max(desired)
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
    def _noise(self, signal, noise_var, avg_wait, clap_amp, talking_pow, gain_variation=1/900):
        delay0 = self.rng.uniform(low=0.1, high=0.5)
        signal = _delay(signal, delay0)
        signal = lfilter(self.system_filter, 1, signal)
        noise = self.rng.normal(loc=0, scale=np.sqrt(noise_var), size=signal.shape)
        noise = lfilter(self.noise_filter, 1, noise)
        if avg_wait is not None:
            noise += _delay(self._clap_noise(signal.size, clap_amp, avg_wait), delay0)
        if talking_pow is not None:
            noise += _delay(self._talking_noise(signal.size, talking_pow), delay0)
        signal += noise
        signal *= np.exp(self._generate_loudness(signal.size, gain_variation))
        return _delay(_clip(signal), self.rng.uniform(low=0.1, high=0.5))
    def simulate(self, signal:np.ndarray, clap: bool=False, talking: bool=False, save_wav: bool=True, directory: str="channel_outputs/") -> np.ndarray:
        #TODO: input validation
        output = self._noise(signal, 0.0005, 1 if clap else None, 2, 0.0005 if talking else None)
        if save_wav:
            filename = ("_clap" if clap else "") + ("_talk" if talking else "") + ".wav"
            filename = datetime.now().strftime("%m%d-%H%M%S%f") + filename
            write(os.path.join(directory, filename), fs, output.astype(np.float32))
            print(f"Saved output to {directory}{filename}")
        return output
    def play(self, signal, directory: str="channel_outputs/"):
        filename = datetime.now().strftime("%m%d-%H%M%S%f") + "no_noise.wav"
        write(os.path.join(directory, filename), fs, signal.astype(np.float32))
        print(f"Saved output to {directory}{filename}")






            
