import pypret
import numpy as np
import pandas as pd
from benchmarking import fast_benchmark_retrieval
import itertools

def calculate_full_metrics(original, retrieved, rel_threshold=0.01):
    intensity_orig = np.abs(original)**2
    max_intensity = np.max(intensity_orig)
    threshold = rel_threshold * max_intensity
    valid_indices = np.where(intensity_orig > threshold)[0]
    significant_mask = np.zeros(len(original), dtype=bool)
    if len(valid_indices) > 0:
        significant_mask[valid_indices[0]:valid_indices[-1]+1] = True

    mag_error_vec = np.abs(np.abs(original) - np.abs(retrieved))
    mae_mag = np.mean(mag_error_vec)

    phase_diff_all = np.angle(original * np.conj(retrieved))
    mae_phase = np.mean(np.abs(phase_diff_all[significant_mask])) if np.any(significant_mask) else 0.0

    unified_error_vec = np.abs(original - retrieved)
    mae_unified = np.mean(unified_error_vec)

    return {"mag_mae": mae_mag, "phase_mae": mae_phase, "unified_mae": mae_unified}

def get_peak_energy_indices(pulse, percentage):
    intensity = np.abs(pulse.spectrum)**2
    N = len(intensity)
    window_size = max(1, int(np.floor(percentage * N)))
    energies = np.convolve(intensity, np.ones(window_size), mode='same')
    peak_center_idx = np.argmax(energies)
    half_width = window_size / 2
    return [np.clip(peak_center_idx - half_width, 0, N) / N, 
            np.clip(peak_center_idx + half_width, 0, N) / N]

print("Initializing Pulse Bank and Parameters...")
pulses = pypret.load("pulse_bank.hdf5")
schemes = ["shg-frog", "thg-ifrog"]
mask_sizes = [0.0, 0.01, 0.05, 0.10]
num_pulses = 10
results_list = []

# Generate all 4 combinations of True/False for processing techniques

boolean_combos = list(itertools.product([False, True], repeat=2))

print(f"Sweeping {num_pulses} pulses across {len(schemes)} schemes.")
rand_idx = np.arange(0, num_pulses)

for scheme_name in schemes:
    for p_idx in range(num_pulses):
        current_pulse = pulses[rand_idx[p_idx]]
        
        for mask_size in mask_sizes:
            if mask_size == 0.0:
                m_type = "unmasked"
                m_inds = [0, 0]
            else:
                m_type = "peak_energy"
                m_inds = get_peak_energy_indices(current_pulse, mask_size)

            for zero_enf, spec_sub in boolean_combos:
                
                if mask_size > 0.0 and zero_enf == True:
                    continue
                
                print(f"[{scheme_name.upper()} | Pulse {p_idx+1}/{num_pulses}] {m_type} {mask_size*100}% | Zeros: {zero_enf} | SpecSub: {spec_sub}")
                
                res_obj = fast_benchmark_retrieval(
                    current_pulse.copy(), scheme_name, "copra", 
                    maxiter=300, repeat=1, additive_noise=0.01, 
                    maskinds=m_inds, 
                    enforce_unmasked_zeros=zero_enf, 
                    spectral_subtraction=spec_sub
                )
                
                out = res_obj.retrievals[0]
                metrics = calculate_full_metrics(out.pulse_original, out.pulse_retrieved)

                results_list.append({
                    "scheme": scheme_name,
                    "pulse_id": rand_idx[p_idx],
                    "type": m_type,
                    "mask_size": mask_size,
                    "zeros_enforced": zero_enf,
                    "spectral_subtraction": spec_sub,
                    "mag_mae": metrics['mag_mae'],
                    "phase_mae": metrics['phase_mae'],
                    "unified_mae": metrics['unified_mae']
                })

df = pd.DataFrame(results_list)

baselines = df[df['type'] == 'unmasked'].set_index(['scheme', 'pulse_id', 'zeros_enforced', 'spectral_subtraction'])

def calculate_deltas(row):

    base = baselines.loc[(row['scheme'], row['pulse_id'], row['zeros_enforced'], row['spectral_subtraction'])]
    return pd.Series({
        'mag_diff': row['mag_mae'] - base['mag_mae'],
        'phase_diff': row['phase_mae'] - base['phase_mae'],
        'unified_diff': row['unified_mae'] - base['unified_mae']
    })

diffs = df.apply(calculate_deltas, axis=1)
df = pd.concat([df, diffs], axis=1)

csv_filename = "full_bank_retrieval_results.csv"
df.to_csv(csv_filename, index=False)
print(f"\nData generation complete. Saved to {csv_filename}")

df_results = pd.read_csv(csv_filename)

def create_label(row):
    if row['type'] == 'unmasked':
        base = "UNMASKED"
    else:
        base = f"Peak Energy ({row['mask_size'] * 100:.0f}%)"

    tags = []
    if row['zeros_enforced']: tags.append("+Zeros")
    if row['spectral_subtraction']: tags.append("+SpecSub")
    
    tag_str = f" [{' '.join(tags)}]" if tags else " [Raw Noise]"
    return base + tag_str

df_results['config_label'] = df_results.apply(create_label, axis=1)

metrics_summary = df_results.pivot_table(
    index=['scheme', 'config_label'],
    values=['unified_mae', 'mag_mae', 'phase_mae'],
    aggfunc='mean'
)

unique_schemes = df_results['scheme'].unique()

print(f"\n{'='*105}")
print(f"{'PHASE RETRIEVAL PERFORMANCE RANKING':^105}")
print(f"{'='*105}")

for scheme in unique_schemes:
    print(f"\n>>> SCHEME: {scheme.upper()}")
    print(f"{'Rank':<5} | {'Configuration':<45} | {'Complex Error':<15} | {'Mag MAE':<12} | {'Phase MAE':<12}")
    print(f"{'-'*105}")
    
    scheme_data = metrics_summary.loc[scheme].copy()
    ranked = scheme_data.sort_values(by='unified_mae')
    
    for i, (label, row) in enumerate(ranked.iterrows(), 1):
        display_label = f">> {label} <<" if "UNMASKED" in label else label
        print(f"{i:<5} | {display_label:<45} | {row['unified_mae']:<15.6f} | {row['mag_mae']:<12.6f} | {row['phase_mae']:<12.6f}")

print(f"\n{'='*105}")