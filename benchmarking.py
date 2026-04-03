""" This module implements testing procedures for retrieval algorithms.
"""
# import path_helper
import copy
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import EngFormatter
from pypret import (FourierTransform, Pulse, random_gaussian, random_pulse,
                    PNPS, material, Retriever, lib)
from pypret.graphics import plot_complex


def benchmark_retrieval(pulse, scheme, algorithm, additive_noise=0.0,
                        repeat=10, maxiter=300, verbose=False,
                        initial_guess="random_gaussian", **kwargs):
    """ Benchmarks a pulse retrieval algorithm. Uses the parameters from our
    paper.

    If you want to benchmark other pulses/configurations you can use the
    procedure below as a starting point.
    """
    # instantiate the result object
    res = SimpleNamespace()
    res.pulse = pulse.copy()
    res.original_spectrum = pulse.spectrum

    # split the scheme
    process, method = scheme.lower().split("-")

    if method == "miips":
        # MIIPS
        parameter = np.linspace(0.0, 2.0*np.pi, 128)  # delta in rad
        pnps = PNPS(pulse, method, process, gamma=22.5e-15, alpha=1.5 * np.pi)
    elif method == "dscan":
        # d-scan
        parameter = np.linspace(-0.025, 0.025, 128)  # insertion in m
        pnps = PNPS(pulse, method, process, material=material.BK7)
    elif method == "ifrog":
        # ifrog
        if process == "sd":
            parameter = np.linspace(pulse.t[0], pulse.t[-1], pulse.N * 4)
        else:
            parameter = pulse.t  # delay in s
        pnps = PNPS(pulse, method, process)
    elif method == "frog":
        # frog
        parameter = pulse.t  # delay in s
        pnps = PNPS(pulse, method, process)
    elif method == "tdp":
        # d-scan
        parameter = np.linspace(pulse.t[0], pulse.t[-1], 128)  # delay in s
        pnps = PNPS(pulse, method, process, center=790e-9, width=10.6e-9)
    else:
        raise ValueError("Method not supported!")
    pnps.calculate(pulse.spectrum, parameter)
    measurement = pnps.trace

    # add noise
    std = measurement.data.max() * additive_noise
    measurement.data += std * np.random.normal(size=measurement.data.shape)

    ret = Retriever(pnps, algorithm, verbose=verbose, logging=True,
                    maxiter=maxiter, **kwargs)

    res.retrievals = []
    for i in range(repeat):
        if initial_guess == "random_gaussian":
            # create random Gaussian pulse
            random_gaussian(pulse, 50e-15, 0.3 * np.pi)
        elif initial_guess == "random":
            pulse.spectrum = (np.random.uniform(size=pulse.N) *
                              np.exp(2.0j * np.pi *
                                     np.random.uniform(size=pulse.N)))
        elif initial_guess == "original":
            pulse.spectrum = res.original_spectrum
        else:
            raise ValueError("Initial guess mode '%s' not supported." % initial_guess)
        ret.retrieve(measurement, pulse.spectrum)
        res.retrievals.append(ret.result(res.original_spectrum))

    return res

def my_benchmark_retrieval(pulse, scheme, algorithm, additive_noise=0.0,
                           repeat=10, maxiter=300, verbose=False,
                           initial_guess="random_gaussian", maskinds=[40,50], 
                           enforce_unmasked_zeros=False, spectral_subtraction=False, **kwargs):
    res = SimpleNamespace()
    res.pulse = pulse.copy()
    res.original_spectrum = pulse.spectrum

    res_m = SimpleNamespace()
    res_m.pulse = pulse.copy()
    res_m.original_spectrum = pulse.spectrum

    # --- Apply masking region to the pulse ---
    N = pulse.N
    if maskinds != [0,0]:
        a, b = maskinds
        i0 = int(np.ceil(a * N))     
        i1 = int(np.floor(b * N))    
        mask_index = (i0, i1)
    else:
        mask_index = None 

    # Blanking indices (15% / 85% edges):
    if maskinds != [0,0]:
        a = 0.15
        b = 0.85
        i0 = int(np.ceil(a * N))     
        i1 = int(np.floor(b * N))    
        blanking_range = (i0, i1)
    else:
        blanking_range = None
    # ----------------------------------------

    process, method = scheme.lower().split("-")

    if method == "miips":
        parameter = np.linspace(0.0, 2.0*np.pi, 128)
        pnps = PNPS(pulse, method, process, gamma=22.5e-15, alpha=1.5 * np.pi)
    elif method == "dscan":
        parameter = np.linspace(-0.025, 0.025, 128)
        pnps = PNPS(pulse, method, process, material=material.BK7)
    elif method == "ifrog":
        if process == "sd":
            parameter = np.linspace(pulse.t[0], pulse.t[-1], pulse.N * 4)
        else:
            parameter = pulse.t
        pnps = PNPS(pulse, method, process)
    elif method == "frog":
        parameter = pulse.t
        pnps = PNPS(pulse, method, process)
    elif method == "tdp":
        parameter = np.linspace(pulse.t[0], pulse.t[-1], 128)
        pnps = PNPS(pulse, method, process, center=790e-9, width=10.6e-9)
    else:
        raise ValueError("Method not supported!")
        
    pnps.calculate(pulse.spectrum, parameter)
    measurement = pnps.trace

    # Save clean data for reference (bounds and background estimation)
    clean_data = measurement.data.copy()
    
    # 1. Add noise
    std = clean_data.max() * additive_noise
    measurement.data += std * np.random.normal(size=measurement.data.shape)

    # 2. Spectral Subtraction
    if spectral_subtraction:
        # Find background region (where clean signal is < 1% of max)
        bg_mask = clean_data < (clean_data.max() * 0.01)
        if np.any(bg_mask):
            noise_mean = np.mean(measurement.data[bg_mask])
            measurement.data -= noise_mean
            measurement.data[measurement.data < 0.0] = 0.0 # Clip negative values

    # 3. Intelligent Zero Enforcement Bounds
    if enforce_unmasked_zeros:
        # delay (rows) bounds
        d_marginal = np.sum(clean_data, axis=1)
        d_valid = np.where(d_marginal > np.max(d_marginal) * 0.01)[0]
        d_start, d_end = (d_valid[0], d_valid[-1]) if len(d_valid) > 0 else (0, clean_data.shape[0]-1)
        
        # frequency (cols) bounds
        f_marginal = np.sum(clean_data, axis=0)
        f_valid = np.where(f_marginal > np.max(f_marginal) * 0.01)[0]
        f_start, f_end = (f_valid[0], f_valid[-1]) if len(f_valid) > 0 else (0, clean_data.shape[1]-1)

        measurement.data[:, :f_start] = 0.0
        measurement.data[:, f_end+1:] = 0.0

    # Masked setup inherits the processed (subtracted/enforced) data
    measurement_masked = copy.deepcopy(measurement)
    if blanking_range is not None:
        measurement_masked.data[:, 0:blanking_range[0]] = 0.0
        measurement_masked.data[:, blanking_range[1]:] = 0.0

    # Full Retrieval
    ret = Retriever(pnps, algorithm, verbose=verbose, logging=True, maxiter=maxiter, **kwargs)
    res.retrievals = []
    for i in range(repeat):
        if initial_guess == "random_gaussian":
            random_gaussian(pulse, 50e-15, 0.3 * np.pi)
        elif initial_guess == "random":
            pulse.spectrum = (np.random.uniform(size=pulse.N) * np.exp(2.0j * np.pi * np.random.uniform(size=pulse.N)))
        elif initial_guess == "original":
            pulse.spectrum = res.original_spectrum
        ret.retrieve(measurement, pulse.spectrum)
        res.retrievals.append(ret.result(res.original_spectrum))

    # Masked Retrieval
    ret_m = Retriever(pnps, algorithm, verbose=verbose, logging=True, maxiter=maxiter, **kwargs)
    M, N_cols = measurement_masked.data.shape
    weights_masked = np.zeros((M, N_cols))

    if mask_index is not None:
        weights_masked[:, mask_index[0]:mask_index[1]] = 1.0
    if blanking_range is not None:
        weights_masked[:, 0:blanking_range[0]] = 1.0
        weights_masked[:, blanking_range[1]:] = 1.0

    measurement_masked.data[weights_masked == 0] = 0.0

    res_m.retrievals = []
    for i in range(repeat):
        if initial_guess == "random_gaussian":
            random_gaussian(pulse, 50e-15, 0.3 * np.pi)
        elif initial_guess == "random":
            pulse.spectrum = (np.random.uniform(size=pulse.N) * np.exp(2.0j * np.pi * np.random.uniform(size=pulse.N)))
        elif initial_guess == "original":
            pulse.spectrum = res.original_spectrum
        ret_m.retrieve(measurement_masked, pulse.spectrum, weights=weights_masked)
        res_m.retrievals.append(ret_m.result(res_m.original_spectrum))

    return res, res_m

def fast_benchmark_retrieval(pulse, scheme, algorithm, additive_noise=0.0,
                             repeat=10, maxiter=300, verbose=False,
                             initial_guess="random_gaussian", maskinds=[40,50], 
                             enforce_unmasked_zeros=False, spectral_subtraction=False, **kwargs):
    res_m = SimpleNamespace()
    res_m.pulse = pulse.copy()
    res_m.original_spectrum = pulse.spectrum

    N = pulse.N
    if maskinds != [0,0]:
        a, b = maskinds
        i0 = int(np.ceil(a * N))     
        i1 = int(np.floor(b * N))    
        mask_index = (i0, i1)
        
        # Blanking indices
        a_blank = 0.15
        b_blank = 0.85
        blanking_range = (int(np.ceil(a_blank * N)), int(np.floor(b_blank * N)))
    else:
        mask_index = None 
        blanking_range = None

    process, method = scheme.lower().split("-")
    if method == "miips":
        parameter = np.linspace(0.0, 2.0*np.pi, 128)
        pnps = PNPS(pulse, method, process, gamma=22.5e-15, alpha=1.5 * np.pi)
    elif method == "dscan":
        parameter = np.linspace(-0.025, 0.025, 128)
        pnps = PNPS(pulse, method, process, material=material.BK7)
    elif method == "ifrog":
        parameter = np.linspace(pulse.t[0], pulse.t[-1], pulse.N * 4) if process == "sd" else pulse.t
        pnps = PNPS(pulse, method, process)
    elif method == "frog":
        parameter = pulse.t
        pnps = PNPS(pulse, method, process)
    elif method == "tdp":
        parameter = np.linspace(pulse.t[0], pulse.t[-1], 128)
        pnps = PNPS(pulse, method, process, center=790e-9, width=10.6e-9)
    else:
        raise ValueError("Method not supported!")
        
    pnps.calculate(pulse.spectrum, parameter)
    measurement_masked = pnps.trace

    clean_data = measurement_masked.data.copy()

    # 1. Add Noise
    std = clean_data.max() * additive_noise
    measurement_masked.data += std * np.random.normal(size=measurement_masked.data.shape)

    # 2. Spectral Subtraction
    if spectral_subtraction:
        bg_mask = clean_data < (clean_data.max() * 0.01)
        if np.any(bg_mask):
            noise_mean = np.mean(measurement_masked.data[bg_mask])
            measurement_masked.data -= noise_mean
            measurement_masked.data[measurement_masked.data < 0.0] = 0.0

    # 3. Intelligent Zero Enforcement
    if enforce_unmasked_zeros:
        # delay (rows) bounds
        d_marginal = np.sum(clean_data, axis=1)
        d_valid = np.where(d_marginal > np.max(d_marginal) * 0.01)[0]
        d_start, d_end = (d_valid[0], d_valid[-1]) if len(d_valid) > 0 else (0, clean_data.shape[0]-1)
        
        # frequency (cols) bounds
        f_marginal = np.sum(clean_data, axis=0)
        f_valid = np.where(f_marginal > np.max(f_marginal) * 0.01)[0]
        f_start, f_end = (f_valid[0], f_valid[-1]) if len(f_valid) > 0 else (0, clean_data.shape[1]-1)

        measurement_masked.data[:, :f_start] = 0.0
        measurement_masked.data[:, f_end+1:] = 0.0

    # Apply Blanking
    if blanking_range is not None:
        measurement_masked.data[:, 0:blanking_range[0]] = 0.0
        measurement_masked.data[:, blanking_range[1]:] = 0.0

    ret_m = Retriever(pnps, algorithm, verbose=verbose, logging=True, maxiter=maxiter, **kwargs)
    M, N_cols = measurement_masked.data.shape

    if blanking_range is not None:
        weights_masked = np.zeros((M, N_cols))
        weights_masked[:, mask_index[0]:mask_index[1]] = 1.0
        weights_masked[:, 0:blanking_range[0]] = 1.0
        weights_masked[:, blanking_range[1]:] = 1.0
    else:
        weights_masked = np.ones((M, N_cols))

    measurement_masked.data[weights_masked == 0] = 0.0

    res_m.retrievals = []
    for i in range(repeat):
        if initial_guess == "random_gaussian":
            random_gaussian(pulse, 50e-15, 0.3 * np.pi)
        elif initial_guess == "random":
            pulse.spectrum = (np.random.uniform(size=pulse.N) * np.exp(2.0j * np.pi * np.random.uniform(size=pulse.N)))
        elif initial_guess == "original":
            pulse.spectrum = res_m.original_spectrum
        ret_m.retrieve(measurement_masked, pulse.spectrum, weights=weights_masked)
        res_m.retrievals.append(ret_m.result(res_m.original_spectrum))

    return res_m

# Modified Plotting Code to compare masked and unmasked results for single pulse:

class my_RetrievalResultPlot:

    def __init__(self, retrieval_result, 
                 masked_result=None, maskinds=[0,0], 
                 error_mask=None, masked_error_mask=None, plot=True, **kwargs):

        self.rr = retrieval_result
        self.og = self.rr.pnps.trace
        
        if masked_result is not None:
            self.masked_rr = masked_result
        else:
            self.masked_rr = self.rr
        
        self.maskinds = maskinds
        self.use_mask = maskinds != [0,0]
        
        # Store the boolean masks for highlighting
        self.error_mask = error_mask
        self.masked_error_mask = masked_error_mask

        if self.rr.pulse_original is None:
            raise ValueError("This plot requires an original pulse.")

        if plot:
            self.plot(**kwargs)

    def _highlight_region(self, ax, x):
        if not self.use_mask:
            return
        a, b = self.maskinds
        N = len(x)
        i0 = int(np.ceil(a * N))
        i1 = int(np.floor(b * N))
        if i1 > i0:
            ax.axvspan(x[i0], x[i1-1], color='green', alpha=0.4)

    def _plot_single(self, use_masked=False, show=True):
        xaxis = "wavelength"
        active_rr = self.masked_rr if use_masked else self.rr
        og = self.og
        
        pulse = Pulse(active_rr.pnps.ft, active_rr.pnps.w0, unit="om")
        fig = plt.figure(figsize=(30.0/2.54, 20.0/2.54))

        gs1 = gridspec.GridSpec(2, 2)
        gs2 = gridspec.GridSpec(2, 8)

        ax1 = plt.subplot(gs1[0, 0])
        ax2 = plt.subplot(gs1[0, 1])
        ax3 = plt.subplot(gs2[1, :2])
        ax4 = plt.subplot(gs2[1, 2:4])
        ax5 = plt.subplot(gs2[1, 4:6])
        ax6 = plt.subplot(gs2[1, 6:])

        ax12, ax22 = ax1.twinx(), ax2.twinx()

        # ---------- time domain plot ----------
        pulse.spectrum = active_rr.pulse_original
        plot_complex(pulse.t, pulse.field, ax1, ax12, amplitude_line="k", phase_line="k")

        pulse.spectrum = active_rr.pulse_retrieved
        li11, li12, _, _ = plot_complex(pulse.t, pulse.field, ax1, ax12)
        for li in [li11, li12]:
            li.set_linewidth(3.0)
            li.set_alpha(0.6)
            
        if use_masked:
            ax1.set_title(f"Time domain (masked {self.maskinds[0]:.2f}-{self.maskinds[1]:.2f})")
        else:
            ax1.set_title("Time domain")

        ax1.set_xlabel("time")
        ax1.set_ylabel("intensity")
        ax12.set_ylabel("phase (rad)")
        ax1.xaxis.set_major_formatter(EngFormatter(unit="s"))

        # ---------- frequency domain plot ----------
        x = pulse.wl
        plot_complex(x, active_rr.pulse_original, ax2, ax22, amplitude_line="k", phase_line="k")
        li21, li22, _, _ = plot_complex(x, active_rr.pulse_retrieved, ax2, ax22)
        for li in [li21, li22]:
            li.set_linewidth(3.0)
            li.set_alpha(0.6)

        # HIGHLIGHT ERROR CALCULATION REGION
        active_mask = self.masked_error_mask if use_masked else self.error_mask
        if active_mask is not None:
            ymin, ymax = ax2.get_ylim()
            # fill_between handles boolean masks seamlessly
            ax2.fill_between(x, ymin, ymax, where=active_mask, color='yellow', alpha=0.3)

        if use_masked:
            ax2.set_title(f"Frequency domain (masked {self.maskinds[0]:.2f}-{self.maskinds[1]:.2f})\nYellow = Phase Error Calc Region")
        else:
            ax2.set_title("Frequency domain\nYellow = Phase Error Calc Region")

        unit = "m" if xaxis == 'wavelength' else "rad Hz"
        label = "wavelength" if xaxis == 'wavelength' else "frequency"

        ax2.set_xlabel(label)
        ax2.set_ylabel("intensity")
        ax22.set_ylabel("phase (rad)")
        ax2.xaxis.set_major_formatter(EngFormatter(unit=unit))

        # ---------- spectrogram normalization ----------
        sc_og = 1.0 / og.data.max()
        sc_noisy = 1.0 / active_rr.trace_input.max()
        sc_ret = 1.0 / active_rr.trace_retrieved.max()

        original = og.data * sc_og
        noisy = active_rr.trace_input * sc_noisy
        retrieved = active_rr.trace_retrieved * sc_ret
        diff = retrieved - original

        traces = [original, noisy, retrieved, diff]
        titles = ["original", "noisy", "retrieved", "difference"]
        cmaps = ["nipy_spectral", "nipy_spectral", "nipy_spectral", "RdBu"]

        md = active_rr.measurement
        axes = [ax3, ax4, ax5, ax6]

        for ax, trace, title, cmap in zip(axes, traces, titles, cmaps):
            xg, yg = lib.edges(active_rr.pnps.process_w), lib.edges(active_rr.parameter)
            im = ax.pcolormesh(xg, yg, trace, cmap=cmap)
            fig.colorbar(im, ax=ax)
            ax.set_title(title)
            ax.set_xlabel(md.labels[1])
            ax.set_ylabel(md.labels[0])
            ax.xaxis.set_major_formatter(EngFormatter(unit=md.units[1]))
            ax.yaxis.set_major_formatter(EngFormatter(unit=md.units[0]))

            if use_masked and title == "original":
                self._highlight_region(ax, xg)

        gs1.update(left=0.05, right=0.95, top=0.9, bottom=0.1, hspace=0.25, wspace=0.5)
        gs2.update(left=0.1, right=0.95, top=0.9, bottom=0.1, hspace=0.5, wspace=1.0)

        if show:
            plt.show()
        return fig

    def plot(self, show=True):
        self.fig_unmasked = self._plot_single(use_masked=False, show=show)
        if self.use_mask:
            self.fig_masked = self._plot_single(use_masked=True, show=show)


class RetrievalResultPlot:

    def __init__(self, retrieval_result, plot=True, **kwargs):
        rr = self.retrieval_result = retrieval_result
        if rr.pulse_original is None:
            raise ValueError("This plot requires an original pulse to compare"
                             " to.")
        if plot:
            self.plot(**kwargs)

    def plot(self, xaxis='wavelength', yaxis='intensity', limit=True,
             phase_blanking=False, phase_blanking_threshold=1e-3, show=True):
        rr = self.retrieval_result
        # reconstruct a pulse from that
        pulse = Pulse(rr.pnps.ft, rr.pnps.w0, unit="om")

        # construct the figure
        fig = plt.figure(figsize=(30.0/2.54, 20.0/2.54))
        gs1 = gridspec.GridSpec(2, 2)
        gs2 = gridspec.GridSpec(2, 6)
        ax1 = plt.subplot(gs1[0, 0])
        ax2 = plt.subplot(gs1[0, 1])
        ax3 = plt.subplot(gs2[1, :2])
        ax4 = plt.subplot(gs2[1, 2:4])
        ax5 = plt.subplot(gs2[1, 4:])
        ax12 = ax1.twinx()
        ax22 = ax2.twinx()

        # Plot in time domain
        pulse.spectrum = rr.pulse_original  # the test pulse
        li011, li012, samp, spha = plot_complex(pulse.t, pulse.field, ax1, ax12, yaxis=yaxis,
                          phase_blanking=phase_blanking, limit=limit,
                          phase_blanking_threshold=phase_blanking_threshold,
                          amplitude_line="k", phase_line="k")
        pulse.spectrum = rr.pulse_retrieved  # the retrieved pulse
        li11, li12, samp, spha = plot_complex(pulse.t, pulse.field, ax1, ax12, yaxis=yaxis,
                          phase_blanking=phase_blanking, limit=limit,
                          phase_blanking_threshold=phase_blanking_threshold)
        li11.set_linewidth(3.0)
        li012.set_linewidth(3.0)
        li11.set_color("#1f77b4")
        li11.set_alpha(0.6)
        li12.set_linewidth(3.0)
        li12.set_color("#ff7f0e")
        li12.set_alpha(0.6)

        fx = EngFormatter(unit="s")
        ax1.xaxis.set_major_formatter(fx)
        ax1.set_title("time domain")
        ax1.set_xlabel("time")
        ax1.set_ylabel(yaxis)
        ax12.set_ylabel("phase (rad)")
        ax1.legend([li011, li11, li12], ["original", "intensity",
                   "phase"])

        # frequency domain
        if xaxis == "wavelength":
            x = pulse.wl
            unit = "m"
            label = "wavelength"
        elif xaxis == "frequency":
            x = pulse.w
            unit = " rad Hz"
            label = "frequency"
        # Plot in spectral domain
        li021, li022, samp, spha = plot_complex(x, rr.pulse_original, ax2, ax22, yaxis=yaxis,
                          phase_blanking=phase_blanking, limit=limit,
                          phase_blanking_threshold=phase_blanking_threshold,
                          amplitude_line="k", phase_line="k")
        li21, li22, samp, spha = plot_complex(x, rr.pulse_retrieved, ax2, ax22, yaxis=yaxis,
                          phase_blanking=phase_blanking, limit=limit,
                          phase_blanking_threshold=phase_blanking_threshold)
        li022.set_linewidth(3.0)
        li21.set_linewidth(3.0)
        li21.set_color("#1f77b4")
        li21.set_alpha(0.6)
        li22.set_linewidth(3.0)
        li22.set_color("#ff7f0e")
        li22.set_alpha(0.6)

        fx = EngFormatter(unit=unit)
        ax2.xaxis.set_major_formatter(fx)
        ax2.set_title("frequency domain")
        ax2.set_xlabel(label)
        ax2.set_ylabel(yaxis)
        ax22.set_ylabel("phase (rad)")
        ax2.legend([li021, li21, li22], ["original", "intensity",
                   "phase"])

        axes = [ax3, ax4, ax5]
        sc = 1.0 / rr.trace_input.max()
        traces = [rr.trace_input * sc, rr.trace_retrieved * sc,
                  (rr.trace_input - rr.trace_retrieved) * sc]
        titles = ["input", "retrieved", "difference"]
        cmaps = ["nipy_spectral", "nipy_spectral", "RdBu"]
        md = rr.measurement
        for ax, trace, title, cmap in zip(axes, traces, titles, cmaps):
            x, y = lib.edges(rr.pnps.process_w), lib.edges(rr.parameter)
            im = ax.pcolormesh(x, y, trace, cmap=cmap)
            fig.colorbar(im, ax=ax)
            ax.set_xlabel(md.labels[1])
            ax.set_ylabel(md.labels[0])
            fx = EngFormatter(unit=md.units[1])
            ax.xaxis.set_major_formatter(fx)
            fy = EngFormatter(unit=md.units[0])
            ax.yaxis.set_major_formatter(fy)
            ax.set_title(title)

        self.fig = fig
        self.ax1, self.ax2 = ax1, ax2
        self.ax12, self.ax22 = ax12, ax22
        self.li11, self.li12, self.li21, self.li22 = li11, li12, li21, li22
        self.ax3, self.ax4, self.ax5 = ax3, ax4, ax5

        if show:
            #gs.tight_layout(fig)
            gs1.update(left=0.05, right=0.95, top=0.9, bottom=0.1,
                      hspace=0.25, wspace=0.5)
            gs2.update(left=0.1, right=0.95, top=0.9, bottom=0.1,
                      hspace=0.5, wspace=1.0)
            plt.show()
