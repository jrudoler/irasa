from scipy import signal
import math
import numpy as np
import scipy as scp
from fractions import Fraction
import matplotlib.pyplot as plt
plt.style.use('seaborn')

class IRASA:
    def __init__(self, sig, f_range, samplerate = 1000,
                hset = np.arange(1.1, 1.95, .05), flag_filter = 1, flag_detrend =  1):
        self.sig = sig
        if f_range is None:
            self.f_range = (0, samplerate/4)
        else:
            self.f_range = f_range
        self.samplerate = samplerate
        self.hset = hset
        self.flag_filter = flag_filter
        self.flag_detrend = flag_detrend
        self.ndim = sig.ndim
        
        self.mixed, self.fractal, self.oscill, self.freq = self.__separate_fractal()
    

    def __separate_fractal(self):
        """
        Inputs:
            sig - timeseries data (last axis (axis = -1) is time)
            f_range - frequency range (1D array)
            samplerate - sample rate in Hz
            hset - array of scaling factors (>1)
            flag_filter  - 1 or 0 (default 1): 1 means filtering before downsampling to avoid aliasing.
            flag_detrend - 1 or 0 (default 1): 1 means detrending data before fft, otherwise 0
        """
        
        
        #should this be in __init__ or __separate_fractal?
        if self.flag_detrend:
            print('Removing linear trend')
            self.sig = signal.detrend(self.sig) # by default, over last axis, which should be time

        #Discrete timeseries of length N_total
        N_total = np.shape(self.sig)[-1]
        #N_data is the power of 2 that does not exceed 90% of N_total
        N_data = 2** math.floor(np.log2(N_total*0.9))
        # N_subset is fixed at 15
        N_subset = 15
        # compute the auto-power spectrum of the originally sampled time series
        L = math.floor((N_total-N_data)/(N_subset-1))
        #need to do fft without truncating
        n_fft = 2**math.ceil(np.log2(math.ceil(round(self.hset[-1], 2)*N_data)))

        N_frac = int(n_fft/2) +1
        freq = self.samplerate/2 * np.linspace(0, 1, N_frac)

        #Compute spectrum of mixed data
        S_mixed = np.take(np.zeros_like(self.sig), range(N_frac), -1)

        taper = self.__get_taper(np.take(self.sig, range(N_data), -1)) #multi-dimensional input handled in taper func
        #sliding window
        for k in range(N_subset):
            i0 = L*k
            x1 = np.take(self.sig, range(i0,i0+N_data), -1)
            p1 = np.fft.fft(x1*taper, n_fft)/min(n_fft, x1.shape[-1])
            p1[1:] = p1[1:]*2
            S_mixed = S_mixed + np.abs(np.take(p1, range(N_frac), -1))**2
        S_mixed = S_mixed/N_subset

        if self.flag_filter:
            print('Filtering to avoid aliasing')
            self.sig_filtered, filt = self.fft_filter(
                self.sig, self.samplerate, 0, self.samplerate/(2*math.ceil(self.hset[-1]))
            )

        print('Computing fractal PSD')
        S_frac = np.take(np.stack([np.zeros_like(self.sig) for i in self.hset]), range(N_frac), -1)

        for ih, h in enumerate(self.hset):
            Sh = np.take(np.zeros_like(self.sig), range(N_frac), -1)
            fr = Fraction(h)
            for k in range(N_subset):
                i0 = L*k
                x1 = np.take(self.sig, range(i0,i0+N_data), -1)
                func = scp.interpolate.interp1d(
                    np.linspace(0, 1, x1.shape[-1]),
                    x1)
                xh = func(np.linspace(0, 1, 
                        np.round(x1.shape[-1]*fr.numerator/fr.denominator).astype(int)))
                taperh = self.__get_taper(xh)
                ph = np.fft.fft(xh*taperh, n_fft)/min(n_fft, xh.shape[-1])
                ph[1:] = ph[1:]*2
                Sh = Sh + np.abs(np.take(ph, range(N_frac), -1))**2
            Sh = Sh/N_subset

            S1h = np.take(np.zeros_like(self.sig), range(N_frac), -1)
            for k in range(N_subset):
                i0 = L*k
                if self.flag_filter:
                    x1 = np.take(self.sig_filtered, range(i0,i0+N_data), -1)
                else:
                    x1 = np.take(self.sig, range(i0,i0+N_data), -1)
                func = scp.interpolate.interp1d(
                    np.linspace(0, 1, x1.shape[-1]),
                    x1)
                x1h = func(np.linspace(0, 1, 
                        np.round(x1.shape[-1]*fr.numerator/fr.denominator).astype(int)))
                taper1h = self.__get_taper(xh)
                p1h = np.fft.fft(x1h*taper1h, n_fft)/min(n_fft, x1h.shape[-1])
                p1h[1:] = p1h[1:]*2
                S1h = S1h + np.abs(np.take(p1h, range(N_frac), -1))**2
            S1h = S1h/N_subset
            S_frac[ih, :] = np.sqrt(Sh*S1h)

        S_frac = np.median(S_frac, 0)

        mask = (freq>=self.f_range[0])&(freq<=self.f_range[1])
        freq = freq[mask]
        S_mixed = np.compress(mask, S_mixed, -1)
        S_frac = np.compress(mask, S_frac, -1)

        return S_mixed, S_frac, S_mixed - S_frac, freq
    
    def __get_taper(self, sig):
        taper = np.hanning(sig.shape[-1])
        if sig.ndim>1:
            taper = np.tile(taper, tuple(list(sig.shape[:-1]) + [1]))
        return taper
    
    def fft_filter(self, ts, fs, lowcut = np.nan, highcut = np.nan, rev_filt = 0, trans = 0.15):
        orig_shape = ts.shape
        n_fft = 2**math.ceil(np.log2(ts.shape[-1]))
        freqs = fs/2 * np.linspace(0, 1, int(n_fft/2 +1))
        res = (freqs[-1]-freqs[0])/(n_fft/2)
        filt = np.ones(n_fft)

        ts_old = ts
        ts = signal.detrend(ts)
        trend = ts_old - ts

        if np.logical_and(~np.isnan(lowcut), lowcut>0) & \
            np.logical_or(np.isnan(highcut), highcut<=0): #highpass
            idxl = int(lowcut/res)+1
            idxlmt = int(lowcut*(1-trans)/res)+1
            idxlmt = max(idxlmt, 1)
            filt[:idxlmt] = 0
            filt[idxlmt-1:idxl] = 0.5*(1+np.sin(-np.pi/2+np.linspace(0, np.pi, idxl-idxlmt+1)))
            filt[n_fft-idxl:n_fft] = filt[idxl-1::-1]         
        elif np.logical_or(np.isnan(lowcut), lowcut<=0)& \
            np.logical_and(~np.isnan(highcut), highcut>0): #lowpass
            idxh = int(highcut/res)+1
            idxhpt = int(highcut*(1+trans)/res)+1
            filt[idxh-1:idxhpt] = 0.5*(1+np.sin(np.pi/2+np.linspace(0, np.pi, idxhpt-idxh+1)))
            filt[idxhpt:int(n_fft/2)+1] = 0  #double check the +1
            filt[int(n_fft/2):n_fft-idxh+1] = filt[int(n_fft/2):idxh-1:-1] #double check!
        elif (lowcut>0)&(highcut>0)&(highcut>lowcut):
            if rev_filt==0:
                transition = (highcut-lowcut)/2 * trans
                idxl = int(lowcut/res)+1
                idxlmt = int(lowcut*(1-transition)/res)+1
                idxh = int(highcut/res)+1
                idxhpt = int(highcut*(1+transition)/res)+1
                idxl = max(idxl, 1)
                idxlmt = max(idxlmt, 1)
                idxh = min(int(n_fft/2)+1, idxh) 
                idxhpt = min(int(n_fft/2)+1, idxhpt)
                filt[:idxlmt] = 0
                filt[idxlmt-1:idxl] = 0.5*(1+np.sin(-np.pi/2+np.linspace(0, np.pi, idxl-idxlmt+1)))
                filt[idxh-1:idxhpt] = 0.5*(1+np.sin(np.pi/2+np.linspace(0, np.pi, idxhpt-idxh+1)))
                filt[idxhpt:int(n_fft/2)+1] = 0
                filt[n_fft-idxl:n_fft] = filt[idxl-1::-1]
                filt[int(n_fft/2):n_fft-idxh+1] = filt[int(n_fft/2):idxh-1:-1] #double check
            else:
                transition = (highcut-lowcut)/2 * trans
                idxl = int(lowcut/res)+1
                idxlmt = int(lowcut*(1-transition)/res)+1
                idxh = int(highcut/res)+1
                idxhpt = int(highcut*(1+transition)/res)+1
                idxl = max(idxl, 1)
                idxlmt = max(idxlmt, 1)
                idxh = min(n_fft/2, idxh)
                idxhpt = min(n_fft/2, idxhpt)
                filt[idxlmt-1:idxl] = 0.5*(1+np.sin(-np.pi/2+np.linspace(0, np.pi, idxl-idxlmt+1)))
                filt[idxl-1:idxh] = 0
                filt[idxh-1:idxhpt] = 0.5*(1+np.sin(np.pi/2+np.linspace(0, np.pi, idxhpt-idxh+1)))
                filt[n_fft-idxhpt+1:n_fft-idxlmt+1] = filt[idxhpt:idxlmt:-1] 
        else:
            raise ValueError('Lowcut and highcut settings not compatible')
        X = np.fft.fft(ts, n_fft)
        ts_new = np.real(np.fft.ifft(X*filt, n_fft))
        ts_new = np.take(ts_new, range(ts.shape[-1]), -1)
        ts_new = ts_new + trend
        
        return ts_new, filt
        
        
    def plaw_fit(self, f_range = None):
        if self.ndim > 2:
            raise Exception('Cannot compute power-law fit for ndim > 2')
        if f_range is None:
            f_range = self.f_range
        mask = (self.freq>=f_range[0])&(self.freq<=f_range[1])
        log_freq = np.log10(self.freq[mask])
        log_frac = np.log10(np.compress(mask, self.fractal, -1))
        
        x2 = np.linspace(min(log_freq), max(log_freq), len(log_freq))
        f = scp.interpolate.interp1d(log_freq, log_frac)
        y2 = f(x2)
        
        p = np.polyfit(x2, y2.T, 1)

        if self.ndim == 1:
            plaw = 10**np.polyval(p, log_freq)
        else:
            plaw = np.vstack([10**np.polyval(p[:, i], log_freq) for i in range(p.shape[-1])])
        self.fit_params = p
        self.fit = plaw
        
        return p, plaw  
    
    def logplot(self, xlim = None, ylim = (None,None), fit = False):
        """
        Plot the fractal and mixed components in of logged power.
        Automatically averages over all non-frequency dimensions
        """
        if fit:
            p, p_law = self.plaw_fit()
        if self.ndim>1:
            frac = np.mean(self.fractal, axis = tuple(range(self.ndim-1)))
            mix = np.mean(self.mixed, axis = tuple(range(self.ndim-1)))
            if fit:
                p_law = np.mean(p_law, axis = tuple(range(self.ndim-1)))
        else:
            frac = self.fractal
            mix = self.mixed
        plt.plot(self.freq, np.log10(frac), c = 'r', label = 'Fractal', lw = 1)
        plt.plot(self.freq, np.log10(mix), c = 'b', alpha = .4, label = 'Mixed', lw = 1)
        if fit:
            plt.plot(self.freq, np.log10(p_law), 'g--', label = 'Power Law Fit')
        if xlim is None:
            xlim = self.f_range
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        plt.legend()
    
    def loglogplot(self, xlim = None, ylim = (None,None), fit = False):
        """
        Plot the fractal and mixed components in log-log scale.
        Automatically averages over all non-frequency dimensions
        """
        if fit:
            p, p_law = self.plaw_fit()
        if self.ndim>1:
            frac = np.mean(self.fractal, axis = tuple(range(self.ndim-1)))
            mix = np.mean(self.mixed, axis = tuple(range(self.ndim-1)))
            if fit:
                p_law = np.mean(p_law, axis = tuple(range(self.ndim-1)))
        else:
            frac = self.fractal
            mix = self.mixed
        plt.loglog(self.freq, frac, c = 'r', label = 'Fractal', lw = 1)
        plt.loglog(self.freq, mix, c = 'b', alpha = .4, label = 'Mixed', lw = 1)
        if fit:
            plt.loglog(self.freq, p_law, 'g--', label = 'Power Law Fit')
        if xlim is None:
            xlim = self.f_range
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        plt.legend()