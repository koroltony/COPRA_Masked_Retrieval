"""
Pulse Retrieval Simulation Testbench
------------------------------------
This module triggers a retrieval simulation, compares full vs. masked traces, 
computes phase and magnitude error metrics over significant regions, and plots 
the results. 
"""

import pypret
import numpy as np
from benchmarking import my_benchmark_retrieval, my_RetrievalResultPlot

def get_peak_energy_indices(pulse, percentage):
    """Finds the spectral window containing the peak energy."""
    intensity = np.abs(pulse.spectrum)**2
    N = len(intensity)
    
    window_size = int(np.floor(percentage * N))
    if window_size < 1: window_size = 1
    
    energies = np.convolve(intensity, np.ones(window_size), mode='same')
    peak_center_idx = np.argmax(energies)
    
    half_width = window_size / 2
    low_idx = np.clip(peak_center_idx - half_width, 0, N)
    high_idx = np.clip(peak_center_idx + half_width, 0, N)
    
    return [low_idx / N, high_idx / N]


def calculate_full_metrics(original, retrieved, rel_threshold=0.01):
    """Calculates magnitude, phase, and unified error over the significant pulse region."""
    intensity_orig = np.abs(original)**2
    max_intensity = np.max(intensity_orig)
    threshold = rel_threshold * max_intensity
    
    valid_indices = np.where(intensity_orig > threshold)[0]
    significant_mask = np.zeros(len(original), dtype=bool)
    
    if len(valid_indices) > 0:
        start_idx = valid_indices[0]
        end_idx = valid_indices[-1]
        significant_mask[start_idx:end_idx+1] = True

    mag_error_vec = np.abs(np.abs(original) - np.abs(retrieved))
    mae_mag = np.mean(mag_error_vec)

    phase_diff_all = np.angle(original * np.conj(retrieved))
    if np.any(significant_mask):
        phase_error_significant = np.abs(phase_diff_all[significant_mask])
        mae_phase = np.mean(phase_error_significant)
    else:
        mae_phase = 0.0

    unified_error_vec = np.abs(original - retrieved)
    mae_unified = np.mean(unified_error_vec)

    return {
        "mag_mae": mae_mag,
        "phase_mae": mae_phase,
        "unified_mae": mae_unified,
        "points_included": np.sum(significant_mask),
        "threshold_used": threshold,
        "significant_mask": significant_mask
    }

def run_retrieval_testbench(
    pulse_bank_file="pulse_bank.hdf5",
    trace_index=1,
    scheme="shg-frog",
    algorithm="copra",
    noise_level=0.01,
    mask_fraction=0.1,
    enforce_unmasked_zeros=False,
    spectral_subtraction=False,
    maxiter=300,
    repeat=1,
    verbose=True
):
    """
    Executes a benchmark comparison between a standard and a masked pulse retrieval.
    
    Parameters:
    -----------
    pulse_bank_file : str
        Path to the HDF5 file containing the simulated pulses.
    trace_index : int
        Index of the specific pulse to test.
    scheme : str
        Measurement scheme (e.g., "shg-frog", "sd-dscan", "tg-frog").
    algorithm : str
        Retrieval algorithm to use (e.g., "copra", "pie").
    noise_level : float
        Fraction of additive Gaussian noise (e.g., 0.01 for 1%).
    mask_fraction : float
        Fraction of the frequency spectrum to preserve around the peak energy.
    enforce_unmasked_zeros : bool
        If True, clears noise in areas with zero spectral/temporal signal in the full trace.
    spectral_subtraction : bool
        If True, estimates background noise from signal-free regions and subtracts it from the trace.
    maxiter : int
        Maximum number of iterations for the retrieval algorithm.
    repeat : int
        Number of times to repeat the retrieval to check consistency.
    verbose : bool
        Prints algorithm progress if True.
    """
    print(f"=== Starting Testbench for Trace {trace_index} | Scheme: {scheme} ===")
    
    # Load Data
    pulses = pypret.load(pulse_bank_file)
    pulse = pulses[trace_index]

    # Setup Masking
    mask_inds = get_peak_energy_indices(pulse, mask_fraction)
    print(f"Masking indices set to: {mask_inds[0]:.3f} - {mask_inds[1]:.3f}")

    # Run Simulation
    res, masked = my_benchmark_retrieval(
        pulse, 
        scheme, 
        algorithm, 
        repeat=repeat,
        verbose=verbose, 
        maxiter=maxiter,
        additive_noise=noise_level,
        maskinds=mask_inds,
        enforce_unmasked_zeros=enforce_unmasked_zeros,
        spectral_subtraction=spectral_subtraction
    )

    # Extract Results
    res_out = res.retrievals[0]
    masked_out = masked.retrievals[0]

    # Calculate Metrics
    res_m = calculate_full_metrics(res_out.pulse_original, res_out.pulse_retrieved)
    mask_m = calculate_full_metrics(masked_out.pulse_original, masked_out.pulse_retrieved)

    diff_mag = mask_m['mag_mae'] - res_m['mag_mae']
    diff_phase = mask_m['phase_mae'] - res_m['phase_mae']
    diff_unified = mask_m['unified_mae'] - res_m['unified_mae']

    # Display Performance Summary
    print(f"\n{'Metric':<15} | {'Full Trace':<12} | {'Masked Trace':<12} | {'Difference':<12}")
    print("-" * 62)
    print(f"{'Magnitude MAE':<15} | {res_m['mag_mae']:<12.6f} | {mask_m['mag_mae']:<12.6f} | {diff_mag:<+12.6f}")
    print(f"{'Phase MAE':<15} | {res_m['phase_mae']:<12.6f} | {mask_m['phase_mae']:<12.6f} | {diff_phase:<+12.6f}")
    print(f"{'Complex MAE':<15} | {res_m['unified_mae']:<12.6f} | {mask_m['unified_mae']:<12.6f} | {diff_unified:<+12.6f}")

    # Visualization
    rrp = my_RetrievalResultPlot(
        res_out, 
        masked_out, 
        maskinds=mask_inds, 
        error_mask=res_m['significant_mask'], 
        masked_error_mask=mask_m['significant_mask']
    )
    
    return res, masked, rrp

if __name__ == "__main__":
    
    full_result, masked_result, plot_object = run_retrieval_testbench(
        pulse_bank_file="pulse_bank.hdf5",
        trace_index=1,
        scheme="shg-frog",
        algorithm="copra",
        noise_level=0.01,
        mask_fraction=0.10,                
        enforce_unmasked_zeros=False,     
        spectral_subtraction=False,         
        maxiter=300,
        repeat=1,
        verbose=True
    )