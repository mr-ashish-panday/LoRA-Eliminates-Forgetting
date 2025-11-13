import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

figures_dir = Path('figures')
figures_dir.mkdir(exist_ok=True)

ranks_data = {4: [], 8: [], 16: [], 32: []}

# Extract rank from filename (e.g., phase1_cola_r8_seed42.json)
import re
for filename in Path('results/phase1').glob('*.json'):
    if 'summary' in filename.name:
        continue
    
    match = re.search(r'_r(\d+)_', filename.name)
    if match:
        rank = int(match.group(1))
        with open(filename) as f:
            data = json.load(f)
            acc = data['metrics']['accuracy']
            if rank in ranks_data:
                ranks_data[rank].append(acc)

# Calculate stats
rank_stats = {}
for rank in sorted(ranks_data.keys()):
    if ranks_data[rank]:
        rank_stats[rank] = {
            'mean': np.mean(ranks_data[rank]) * 100,
            'std': np.std(ranks_data[rank]) * 100
        }

print("Extracted:", rank_stats)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ranks = sorted(rank_stats.keys())
means = [rank_stats[r]['mean'] for r in ranks]
stds = [rank_stats[r]['std'] for r in ranks]

ax.errorbar(ranks, means, yerr=stds, fmt='o-', color='#27ae60', linewidth=2.5, markersize=10, capsize=8)
ax.fill_between(ranks, np.array(means) - np.array(stds), np.array(means) + np.array(stds), alpha=0.2, color='#27ae60')
ax.set_xlabel('LoRA Rank (r)', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('FIGURE 3: LoRA Rank Analysis', fontsize=13, fontweight='bold', pad=20)
ax.set_xticks(ranks)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir / 'fig3_rank_analysis.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figure 3 saved!")
