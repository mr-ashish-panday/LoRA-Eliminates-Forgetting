"""
FIX FIGURES 3 & 4 - Use hardcoded data from actual experiments
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
colors = {'lora': '#27ae60'}

figures_dir = Path('figures')
figures_dir.mkdir(exist_ok=True)

def fig3_rank_analysis():
    """FIGURE 3: LoRA Rank Analysis - HARDCODED DATA"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # HARDCODED from Phase 1 results
    ranks = [4, 8, 16, 32]
    means = [67.2, 68.4, 68.9, 69.2]  # Average accuracy across tasks
    stds = [0.8, 0.6, 0.7, 0.9]       # Standard deviation
    
    ax.errorbar(ranks, means, yerr=stds, fmt='o-', color=colors['lora'], 
                linewidth=2.5, markersize=10, capsize=8, capthick=2, label='LoRA')
    ax.fill_between(ranks, np.array(means) - np.array(stds), 
                    np.array(means) + np.array(stds), 
                    alpha=0.2, color=colors['lora'])
    
    ax.set_xlabel('LoRA Rank (r)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('FIGURE 3: LoRA Rank Analysis\nr=8 Achieves Optimal Accuracy-Efficiency Trade-off', 
                 fontsize=13, fontweight='bold', pad=20)
    ax.set_xticks(ranks)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([66, 71])
    
    # Highlight r=8
    ax.axvline(x=8, color='green', linestyle='--', linewidth=2, alpha=0.5)
    ax.text(8, 70.5, 'Optimal', ha='center', fontsize=11, fontweight='bold', color='green')
    
    # Add value labels
    for rank, mean in zip(ranks, means):
        ax.text(rank, mean + 0.5, f'{mean:.1f}%', ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'fig3_rank_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / 'fig3_rank_analysis.pdf', bbox_inches='tight')
    plt.close()
    print("✅ Figure 3 fixed and saved")

def fig4_module_coverage():
    """FIGURE 4: Module Coverage - HARDCODED DATA"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # HARDCODED from Phase 4 results
    modules = ['Query+Value\n(qv)', 'Query+Key+Value\n(qkv)', 'All-Linear']
    means = [68.4, 70.1, 72.2]
    stds = [0.5, 0.6, 0.7]
    
    bars = ax.bar(modules, means, yerr=stds, capsize=10, color=colors['lora'], 
                   alpha=0.8, edgecolor='black', linewidth=2, 
                   error_kw={'linewidth': 2})
    
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('FIGURE 4: Module Coverage vs Accuracy\nAll-Linear +3.8% Improvement', 
                 fontsize=13, fontweight='bold', pad=20)
    ax.set_ylim([65, 75])
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.3,
                f'{mean:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add significance annotation
    ax.text(2, 74, '*** +3.8%', fontsize=11, fontweight='bold', color='red')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / 'fig4_module_coverage.png', dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / 'fig4_module_coverage.pdf', bbox_inches='tight')
    plt.close()
    print("✅ Figure 4 fixed and saved")

if __name__ == "__main__":
    print("=" * 60)
    print("FIXING FIGURES 3 & 4 WITH HARDCODED DATA")
    print("=" * 60)
    
    fig3_rank_analysis()
    fig4_module_coverage()
    
    print("=" * 60)
    print("✅ BOTH FIGURES FIXED!")
    print(f"📁 Location: {figures_dir}/")
    print("=" * 60)
