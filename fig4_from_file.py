"""
FIGURE 4 - READ DIRECTLY FROM phase4_summary.json FILE
No hardcoding - 100% from actual data
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

figures_dir = Path('figures')
figures_dir.mkdir(exist_ok=True)

# READ from file
with open('results/phase4/phase4_summary.json', 'r') as f:
    phase4_data = json.load(f)

# EXTRACT accuracy for each module across all tasks
modules_data = {
    'qv': [],
    'qkv': [],
    'all_linear': []
}

# Parse all results
for key, experiment in phase4_data['results'].items():
    module_type = experiment['module_type']
    accuracy = experiment['metrics']['accuracy']
    
    if module_type in modules_data:
        modules_data[module_type].append(accuracy)

# Calculate statistics
module_stats = {}
for mod, accuracies in modules_data.items():
    if accuracies:
        module_stats[mod] = {
            'mean': np.mean(accuracies),
            'std': np.std(accuracies),
            'count': len(accuracies)
        }

print("=" * 60)
print("EXTRACTED FROM phase4_summary.json FILE:")
print("=" * 60)
for mod, stats in module_stats.items():
    print(f"{mod:12} | Mean: {stats['mean']*100:5.1f}% | Std: {stats['std']*100:4.1f}% | N={stats['count']}")

# Create figure with FILE data
modules = ['Query+Value\n(qv)', 'Query+Key+Value\n(qkv)', 'All-Linear']
module_keys = ['qv', 'qkv', 'all_linear']
means = [module_stats[k]['mean']*100 for k in module_keys]
stds = [module_stats[k]['std']*100 for k in module_keys]

colors_lora = '#27ae60'

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(modules, means, yerr=stds, capsize=10, color=colors_lora, 
               alpha=0.8, edgecolor='black', linewidth=2, 
               error_kw={'linewidth': 2})

ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
improvement = means[2] - means[0]
ax.set_title(f'FIGURE 4: Module Coverage vs Accuracy\nAll-Linear +{improvement:.1f}% Improvement', 
             fontsize=13, fontweight='bold', pad=20)
ax.set_ylim([min(means) - 5, max(means) + 5])

# Add value labels
for bar, mean, std in zip(bars, means, stds):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.3,
            f'{mean:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add improvement annotation
ax.text(2, max(means) + 2, f'*** +{improvement:.1f}%', fontsize=11, fontweight='bold', color='red')

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / 'fig4_module_coverage.png', dpi=300, bbox_inches='tight')
plt.savefig(figures_dir / 'fig4_module_coverage.pdf', bbox_inches='tight')
plt.close()

print("=" * 60)
print("✅ FIGURE 4 CREATED FROM FILE!")
print(f"📊 Results: QV={means[0]:.1f}%, QKV={means[1]:.1f}%, All-Linear={means[2]:.1f}%")
print(f"📁 Saved: figures/fig4_module_coverage.png/pdf")
print("=" * 60)
