from scipy import signal
import math
import numpy as np
from scipy.interpolate import interp1d
from fractions import Fraction
import time
import matplotlib.pyplot as plt
plt.style.use('seaborn')


class IRASA:
    """
    Irregular Resampling Auto Spectral Analysis.
    Separates 1/f fractal component from oscillatory component.
    """

    def __init__(self, sig, freqs=None, samplerate=1000,
                 hset=np.arange(1.1, 1.95, .05), flag_filter=1,
                 flag_detrend=1):
        """
        Parameters
        __________

        sig - timeseries data (last axis (axis = -1) must be time/samples)
        freqs - frequencies to be included in power spectrum (1D array)
        samplerate - sample rate in Hz
        hset - array of resampling factors (>1)
        flag_filter  - 1 or 0 (default 1): 1 means filtering before downsampling to avoid aliasing.
        flag_detrend - 1 or 0 (default 1): 1 means detrending data before fft
        
        Returns
        __________
        IRASA class instance

        """
        self.sig = sig
        self.freqs = freqs
        self.samplerate = samplerate
        self.hset = hset
        self.flag_filter = flag_filter
        self.flag_detrend = flag_detrend
        self.ndim = sig.ndim
        self.mixed, self.fractal = self.__separate_fractal()

    def __separate_fractal(self):
        """
        Separate the fractal and oscillatory components of a timeseries.
        """
        if self.flag_detrend:
            print('Removing linear trend')
            self.sig = signal.detrend(self.sig)
            # by default, over last axis, which should be time

        # Discrete timeseries of length N_total
        N_total = np.shape(self.sig)[-1]
        # N_data is the power of 2 that does not exceed 90% of N_total
        N_data = 2 ** math.floor(np.log2(N_total * 0.9))
        # N_subset is fixed at 15
        N_subset = 15
        # The increment for the sliding window
        L = math.floor((N_total - N_data) / (N_subset - 1))
        # Need to do fft without truncating
        n_fft = 2 ** math.ceil(
            np.log2(math.ceil(round(self.hset[-1], 2) * N_data)))
        N_frac = int(n_fft / 2) + 1
        # sample values in frequency space
        fft_freq = self.samplerate / 2 * np.linspace(0, 1, N_frac)

        # Compute spectrum of mixed data
        S_mixed = np.zeros(self.sig.shape[:-1] + (N_frac,))
        # Multi-dimensional input handled in taper func
        taper = self.__get_taper(np.take(self.sig, range(N_data), -1))
        for k in range(N_subset):  # Sliding window
            i0 = L * k
            x1 = np.take(self.sig, range(i0, i0 + N_data), -1)
            p1 = np.fft.fft(x1 * taper, n_fft) / min(n_fft, x1.shape[-1])
            p1[1:] = p1[1:] * 2
            S_mixed = S_mixed + np.abs(np.take(p1, range(N_frac), -1)) ** 2
        S_mixed = S_mixed / N_subset

        if self.flag_filter:
            print('Filtering to avoid aliasing')
            self.sig_filtered, filt = self.fft_filter(
                self.sig, self.samplerate, 0,
                self.samplerate / (2 * math.ceil(self.hset[-1]))
            )

        print('Computing fractal PSD')
        S_frac = np.zeros((len(self.hset),) + self.sig.shape[:-1] + (N_frac,))
        tic = time.time()
        for ih, h in enumerate(self.hset):
            Sh = np.zeros(self.sig.shape[:-1] + (N_frac,))
            fr = Fraction(h)
            for k in range(N_subset):  # upsampling
                i0 = L * k
                x1 = np.take(self.sig, range(i0, i0 + N_data), -1)
                func = interp1d(
                    np.linspace(0, 1, x1.shape[-1]),
                    x1)
                xh = func(np.linspace(
                    0, 1,
                    np.round(x1.shape[-1] * fr.numerator / fr.denominator)
                    .astype(int)))
                taperh = self.__get_taper(xh)
                ph = np.fft.fft(xh * taperh, n_fft) / min(n_fft, xh.shape[-1])
                ph[1:] = ph[1:] * 2
                Sh = Sh + np.abs(np.take(ph, range(N_frac), -1))**2
            Sh = Sh / N_subset

            S1h = np.zeros(self.sig.shape[:-1] + (N_frac,))
            for k in range(N_subset):  # downsampling
                i0 = L * k
                if self.flag_filter:
                    x1 = np.take(self.sig_filtered, range(i0, i0 + N_data), -1)
                else:
                    x1 = np.take(self.sig, range(i0, i0 + N_data), -1)
                func = interp1d(
                    np.linspace(0, 1, x1.shape[-1]), x1)
                x1h = func(np.linspace(
                    0, 1,
                    np.round(x1.shape[-1] * fr.denominator / fr.numerator).astype(int)))
                taper1h = self.__get_taper(x1h)
                p1h = np.fft.fft(x1h * taper1h, n_fft) / min(n_fft, x1h.shape[-1])
                p1h[1:] = p1h[1:] * 2
                S1h = S1h + np.abs(np.take(p1h, range(N_frac), -1))**2
            S1h = S1h / N_subset
            S_frac[ih, :] = np.sqrt(Sh * S1h)  # geometric mean
        toc = time.time()
        print(f"Time elapsed for FFT: {toc-tic:.4f} s")
        S_frac = np.median(S_frac, 0)

        if self.freqs is None:
        	# assume max h is 2.0, want max frequency to be half the lowest samplerate
            mask = (fft_freq >= 0) & (fft_freq <= self.samplerate / 4)
            self.freqs = fft_freq[mask]
            S_mixed = np.compress(mask, S_mixed, -1)
            S_frac = np.compress(mask, S_frac, -1)
        else:
            func = interp1d(fft_freq, S_mixed)
            S_mixed = func(self.freqs)
            func = interp1d(fft_freq, S_frac)
            S_frac = func(self.freqs)

        return S_mixed, S_frac

    def __get_taper(self, sig):
        """
        Taper the signal with a Hanning window.
        """
        taper = np.hanning(sig.shape[-1])
        if sig.ndim > 1:
            taper = np.tile(taper, tuple(list(sig.shape[:-1]) + [1]))
        return taper

    def fft_filter(self, ts, fs, lowcut=np.nan, highcut=np.nan,
                   rev_filt=0, trans=0.15):
        n_fft = 2**math.ceil(np.log2(ts.shape[-1]))
        freqs = fs / 2 * np.linspace(0, 1, int(n_fft / 2 + 1))
        res = (freqs[-1] - freqs[0]) / (n_fft / 2)
        filt = np.ones(n_fft)

        ts_old = ts
        ts = signal.detrend(ts)
        trend = ts_old - ts

        if np.logical_and(~np.isnan(lowcut), lowcut > 0) & \
                np.logical_or(np.isnan(highcut), highcut <= 0):  # highpass
            idxl = int(lowcut / res) + 1
            idxlmt = int(lowcut * (1 - trans) / res) + 1
            idxlmt = max(idxlmt, 1)
            filt[:idxlmt] = 0
            filt[idxlmt - 1:idxl] = \
                0.5 * (1 + np.sin(
                    -np.pi / 2 + np.linspace(0, np.pi, idxl - idxlmt + 1)))
            filt[n_fft - idxl:n_fft] = filt[idxl - 1::-1]
        elif np.logical_or(np.isnan(lowcut), lowcut <= 0) & \
                np.logical_and(~np.isnan(highcut), highcut > 0):  # lowpass
            idxh = int(highcut / res) + 1
            idxhpt = int(highcut * (1 + trans) / res) + 1
            filt[idxh - 1:idxhpt] = 0.5 * (1 + np.sin(np.pi / 2 + np.linspace(
                0, np.pi, idxhpt - idxh + 1)))
            filt[idxhpt:int(n_fft / 2) + 1] = 0  # double check the +1
            filt[int(n_fft / 2):n_fft - idxh + 1] = \
                filt[int(n_fft / 2):idxh - 1:-1]  # double check!
        elif (lowcut > 0) & (highcut > 0) & (highcut > lowcut):
            if rev_filt == 0:
                transition = (highcut - lowcut) / 2 * trans
                idxl = int(lowcut / res) + 1
                idxlmt = int(lowcut * (1 - transition) / res) + 1
                idxh = int(highcut / res) + 1
                idxhpt = int(highcut * (1 + transition) / res) + 1
                idxl = max(idxl, 1)
                idxlmt = max(idxlmt, 1)
                idxh = min(int(n_fft / 2) + 1, idxh)
                idxhpt = min(int(n_fft / 2) + 1, idxhpt)
                filt[:idxlmt] = 0
                filt[idxlmt - 1:idxl] = 0.5 * (1 + np.sin(
                    -np.pi / 2 + np.linspace(0, np.pi, idxl - idxlmt + 1)))
                filt[idxh - 1:idxhpt] = 0.5 * (1 + np.sin(
                    np.pi / 2 + np.linspace(0, np.pi, idxhpt - idxh + 1)))
                filt[idxhpt:int(n_fft / 2) + 1] = 0
                filt[n_fft - idxl:n_fft] = filt[idxl - 1::-1]
                filt[int(n_fft / 2):n_fft - idxh + 1] = \
                    filt[int(n_fft / 2):idxh - 1:-1]  # double check
            else:
                transition = (highcut - lowcut) / 2 * trans
                idxl = int(lowcut / res) + 1
                idxlmt = int(lowcut * (1 - transition) / res) + 1
                idxh = int(highcut / res) + 1
                idxhpt = int(highcut * (1 + transition) / res) + 1
                idxl = max(idxl, 1)
                idxlmt = max(idxlmt, 1)
                idxh = min(n_fft / 2, idxh)
                idxhpt = min(n_fft / 2, idxhpt)
                filt[idxlmt - 1:idxl] = \
                    0.5 * (1 + np.sin(-np.pi / 2 + np.linspace(0, np.pi, idxl - idxlmt + 1)))
                filt[idxl - 1:idxh] = 0
                filt[idxh - 1:idxhpt] = 0.5 * (1 + np.sin(
                    np.pi / 2 + np.linspace(0, np.pi, idxhpt - idxh + 1)))
                filt[n_fft - idxhpt + 1:(n_fft - idxlmt + 1)] = \
                    filt[idxhpt:idxlmt:-1]
        else:
            raise ValueError('Lowcut and highcut settings not compatible')
        X = np.fft.fft(ts, n_fft)
        ts_new = np.real(np.fft.ifft(X * filt, n_fft))
        ts_new = np.take(ts_new, range(ts.shape[-1]), -1)
        ts_new = ts_new + trend
        return ts_new, filt

    def plaw_fit(self):
        """
        Linear fit to the power spectrum in log-log coordinates. Works up to 3 dimensions.

        Returns:
        p - the fit parameters, with highest order first
        plaw - the fit values at the frequencies being anlyzed (chosen during initialization)
        """
        log_freq = np.log10(self.freqs)
        log_frac = np.log10(self.fractal)
        x2 = np.linspace(min(log_freq), max(log_freq), len(log_freq))
        f = interp1d(log_freq, log_frac)
        y2 = f(x2)

        if self.ndim == 1:
            p = np.polyfit(x2, y2.T, 1)
            plaw = 10**np.polyval(p, log_freq)
        elif self.ndim == 2:
            p = np.polyfit(x2, y2.T, 1)
            plaw = np.vstack([10**np.polyval(p[:, pi], log_freq) for pi in range(p.shape[-1])])
        elif self.ndim == 3:
            plaw = np.zeros_like(log_frac)
            p = []
            for i in range(len(log_frac)):
                this_p = np.polyfit(x2, y2[i, :, :].T, 1)
                p.append(this_p)
                plaw[i] = np.stack(
                    [10**np.polyval(this_p[:, pi], log_freq) for pi in range(this_p.shape[-1])])
            p = np.stack(p)
        else:
            raise Exception('Cannot compute power-law fit for ndim > 3')
        self.fit_params = p
        self.fit = plaw
        return p, plaw

    def psdplot(self, xlim=(None, None), ylim=(None, None), fit=False):
        """
        Plot the fractal and mixed components of power spectral decomposition.
        Automatically averages over all non-frequency dimensions
        """
        if fit:
            p, p_law = self.plaw_fit()
        if self.ndim > 1:
            frac = np.mean(self.fractal, axis=tuple(range(self.ndim - 1)))
            mix = np.mean(self.mixed, axis=tuple(range(self.ndim - 1)))
            if fit:
                p_law = np.mean(p_law, axis=tuple(range(self.ndim - 1)))
        else:
            frac = self.fractal
            mix = self.mixed
        plt.plot(self.freqs, frac, c='r', label='Fractal', lw=1)
        plt.plot(self.freqs, mix, c='b', alpha=.4, label='Mixed', lw=1)
        if fit:
            plt.plot(self.freqs, p_law, 'g--', label='Power Law Fit')
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        plt.xlabel('Frequency')
        plt.ylabel('Power')
        plt.legend(loc=1)

    def loglogplot(self, xlim=(None, None), ylim=(None, None), fit=False):
        """
        Plot the fractal and mixed components in log-log scale.
        Automatically averages over all non-frequency dimensions.
        """
        if fit:
            p, p_law = self.plaw_fit()
        if self.ndim > 1:
            frac = np.mean(self.fractal, axis=tuple(range(self.ndim - 1)))
            mix = np.mean(self.mixed, axis=tuple(range(self.ndim - 1)))
            if fit:
                p_law = np.mean(p_law, axis=tuple(range(self.ndim - 1)))
        else:
            frac = self.fractal
            mix = self.mixed
        plt.loglog(self.freqs, frac, c='r', label='Fractal', lw=1)
        plt.loglog(self.freqs, mix, c='b', alpha=.4, label='Mixed', lw=1)
        if fit:
            plt.loglog(self.freqs, p_law, 'g--', label='Power Law Fit')
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        plt.xlabel('Frequency')
        plt.ylabel('Power')
        plt.legend(loc=3)

    def plot_oscillatory(self, xlim=(None, None), ylim=(None, None)):
        """
        Plot the oscillatory component, which is equal to the difference of the
        mixed signal and fractal components.
        """
        if self.ndim > 1:
            frac = np.mean(self.fractal, axis=tuple(range(self.ndim - 1)))
            mix = np.mean(self.mixed, axis=tuple(range(self.ndim - 1)))
        else:
            frac = self.fractal
            mix = self.mixed
        plt.plot(self.freqs, mix - frac)
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])

def SSL_transform(x):
    """
    Shifted, Symmetric Log Transform to suppress extrema
    Input:
    x - a power spectrum to be transformed
    """
    return np.sign(x) * np.log10(np.abs(x)+1)
