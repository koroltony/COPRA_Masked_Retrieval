import pandas as pd

df = pd.read_csv("full_bank_retrieval_results.csv")

def create_label(row):
    if row['type'] == 'unmasked':
        base = "UNMASKED"
    else:
        base = f"Peak Energy ({row['mask_size'] * 100:.0f}%)"
    
    # Add tags for the processing used
    tags = []
    if row['zeros_enforced']: tags.append("+Zeros")
    if row['spectral_subtraction']: tags.append("+SpecSub")
    
    tag_str = f" [{' '.join(tags)}]" if tags else " [Raw Noise]"
    return base + tag_str

df['config_label'] = df.apply(create_label, axis=1)

metrics_summary = df.pivot_table(
    index=['scheme', 'config_label'],
    values=['unified_mae', 'mag_mae', 'phase_mae'],
    aggfunc='mean'
)

schemes = df['scheme'].unique()

print(f"\n{'='*105}")
print(f"{'PHASE RETRIEVAL PERFORMANCE RANKING':^105}")
print(f"{'='*105}")

for scheme in schemes:
    print(f"\n>>> SCHEME: {scheme.upper()}")
    print(f"{'Rank':<5} | {'Configuration':<45} | {'Complex Error':<15} | {'Mag MAE':<12} | {'Phase MAE':<12}")
    print(f"{'-'*105}")
    
    scheme_data = metrics_summary.loc[scheme].copy()
    ranked = scheme_data.sort_values(by='unified_mae')
    
    for i, (label, row) in enumerate(ranked.iterrows(), 1):
        display_label = f">> {label} <<" if "UNMASKED" in label else label
        print(f"{i:<5} | {display_label:<45} | {row['unified_mae']:<15.6f} | {row['mag_mae']:<12.6f} | {row['phase_mae']:<12.6f}")

print(f"\n{'='*105}")
print(f"Note: If a masked config ranks ABOVE its Unmasked Baseline, the mask improved noise robustness.")