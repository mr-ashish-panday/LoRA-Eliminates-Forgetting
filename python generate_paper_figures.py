"""
MetaLoRA Paper Figure Generator
Generates publication-quality figures from verified experimental data
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

def create_figure1_importance_distribution():
    """
    Figure 1: Module Importance Distribution with Zero Variance
    Shows gradient-based importance across 6 module types
    Emphasizes zero variance (perfect reproducibility)
    """
    # Verified data from enhanced_importance_stats.csv
    modules = ['out_proj', 'mlp_fc', 'mlp_proj', 'q_proj', 'k_proj', 'v_proj']
    importance = [23.6, 18.56, 15.64, 14.07, 14.07, 14.07]  # Percentages
    std_dev = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Zero variance!
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Color scheme
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    # Create bars
    bars = ax.bar(modules, importance, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=1.5)
    
    # Add error bars (will be invisible but emphasizes zero variance)
    ax.errorbar(modules, importance, yerr=std_dev, fmt='none', 
                ecolor='black', capsize=5, capthick=2)
    
    # Add value labels on bars
    for bar, val in zip(bars, importance):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Styling
    ax.set_xlabel('Module Type', fontweight='bold', fontsize=12)
    ax.set_ylabel('Relative Importance (%)', fontweight='bold', fontsize=12)
    ax.set_title('Figure 1: Gradient-Based Module Importance with Zero-Variance Reproducibility\n' +
                 '(5 seeds × 63,432 samples each = 317,160 total samples)',
                 fontweight='bold', fontsize=11, pad=15)
    ax.set_ylim(0, 28)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add annotation about zero variance
    ax.text(0.98, 0.97, 'std = 0.000 for all modules\n(Perfect reproducibility)',
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('figure1_importance_distribution.pdf', bbox_inches='tight')
    plt.savefig('figure1_importance_distribution.png', bbox_inches='tight')
    print("✓ Figure 1 created: figure1_importance_distribution.pdf")
    plt.close()


def create_figure2_cross_budget_performance():
    """
    Figure 2: Cross-Budget Performance and Memory Efficiency
    Shows F1/EM consistency across budgets with actual memory usage
    """
    # Verified data from phase3b_memory_sweep_final_complete.csv
    budgets = [6, 8, 10, 12]
    
    # Mean values
    f1_mean = [81.404, 81.438, 81.470, 81.450]
    em_mean = [74.200, 74.267, 74.267, 74.300]
    vram_mean = [0.597, 0.810, 0.810, 0.703]  # Average of actual VRAM
    
    # Standard deviations
    f1_std = [0.096, 0.106, 0.170, 0.097]
    em_std = [0.265, 0.208, 0.208, 0.173]
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot F1 and EM on left axis
    color1 = '#2c3e50'
    color2 = '#e74c3c'
    
    line1 = ax1.plot(budgets, f1_mean, 'o-', color=color1, linewidth=2.5, 
                     markersize=8, label='F1 Score', markeredgewidth=2, markeredgecolor='white')
    ax1.fill_between(budgets, 
                     np.array(f1_mean) - np.array(f1_std),
                     np.array(f1_mean) + np.array(f1_std),
                     alpha=0.2, color=color1)
    
    line2 = ax1.plot(budgets, em_mean, 's-', color=color2, linewidth=2.5,
                     markersize=8, label='Exact Match', markeredgewidth=2, markeredgecolor='white')
    ax1.fill_between(budgets,
                     np.array(em_mean) - np.array(em_std),
                     np.array(em_mean) + np.array(em_std),
                     alpha=0.2, color=color2)
    
    ax1.set_xlabel('Memory Budget (GiB)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Performance (%)', fontweight='bold', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_ylim(72, 84)
    ax1.grid(axis='both', alpha=0.3, linestyle='--')
    
    # Plot actual VRAM usage on right axis
    ax2 = ax1.twinx()
    color3 = '#27ae60'
    
    line3 = ax2.bar(budgets, vram_mean, alpha=0.6, color=color3, 
                    width=0.8, edgecolor='black', linewidth=1.5, label='Actual VRAM Usage')
    
    # Add budget reference line
    ax2.plot(budgets, budgets, 'k--', linewidth=2, alpha=0.5, label='Allocated Budget')
    
    ax2.set_ylabel('Memory (GiB)', fontweight='bold', fontsize=12, color=color3)
    ax2.tick_params(axis='y', labelcolor=color3)
    ax2.set_ylim(0, 14)
    
    # Title
    fig.suptitle('Figure 2: Cross-Budget Performance Consistency and Memory Efficiency\n' +
                 '(3 seeds per budget, shaded regions show ±1 std)',
                 fontweight='bold', fontsize=11, y=0.98)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', 
               framealpha=0.9, edgecolor='black')
    
    # Add annotation about conservative usage
    ax2.text(0.98, 0.30, 
             'Actual usage: 0.49-0.81 GiB\n' +
             'Budget: 6-12 GiB\n' +
             'Utilization: 4.1-13.5%\n' +
             '(Conservative allocation)',
             transform=ax2.transAxes, ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
             fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('figure2_cross_budget_performance.pdf', bbox_inches='tight')
    plt.savefig('figure2_cross_budget_performance.png', bbox_inches='tight')
    print("✓ Figure 2 created: figure2_cross_budget_performance.pdf")
    plt.close()


def create_figure3_calibration_results():
    """
    Figure 3: Top Calibration Configurations
    Shows best configurations from exhaustive 4,400-config search
    """
    # Verified data from Table 4
    configs = ['OTHERGIB6.70\n(Best)', 'OTHERGIB6.60', 'OTHERGIB6.80', 
               'OTHERGIB6.50', 'OTHERGIB6.40']
    errors = [0.64, 3.30, 4.60, 7.30, 11.20]
    diversity = [48, 45, 46, 44, 43]
    budgets = [6.7, 6.6, 6.8, 6.5, 6.4]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left plot: Prediction Error
    colors = ['#27ae60', '#3498db', '#3498db', '#f39c12', '#e74c3c']
    bars1 = ax1.bar(range(len(configs)), errors, color=colors, alpha=0.8,
                    edgecolor='black', linewidth=1.5)
    
    ax1.set_xlabel('Configuration', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Memory Prediction Error (%)', fontweight='bold', fontsize=11)
    ax1.set_title('(a) Prediction Error (Lower is Better)', fontweight='bold', fontsize=10)
    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels(configs, rotation=0, ha='center', fontsize=8)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 13)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, errors)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{val}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Add horizontal line at 1% (sub-1% goal)
    ax1.axhline(y=1.0, color='green', linestyle='--', linewidth=2, alpha=0.7,
                label='Sub-1% threshold')
    ax1.legend(loc='upper left')
    
    # Right plot: Rank Diversity
    bars2 = ax2.bar(range(len(configs)), diversity, color=colors, alpha=0.8,
                    edgecolor='black', linewidth=1.5)
    
    ax2.set_xlabel('Configuration', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Rank Diversity (Distinct Values)', fontweight='bold', fontsize=11)
    ax2.set_title('(b) Allocation Diversity (Higher is Better)', fontweight='bold', fontsize=10)
    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels(configs, rotation=0, ha='center', fontsize=8)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(40, 52)
    
    # Add value labels
    for bar, val in zip(bars2, diversity):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{val}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Main title
    fig.suptitle('Figure 3: Top Configurations from Exhaustive Calibration\n' +
                 '(4,400 configurations tested: 115 budgets × 50 spike factors)',
                 fontweight='bold', fontsize=11, y=1.00)
    
    # Add annotation
    fig.text(0.5, -0.02, 
             'All top 5 configurations use spike factor = 0.001 (optimal default parameter)',
             ha='center', fontsize=9, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('figure3_calibration_results.pdf', bbox_inches='tight')
    plt.savefig('figure3_calibration_results.png', bbox_inches='tight')
    print("✓ Figure 3 created: figure3_calibration_results.pdf")
    plt.close()


def main():
    """Generate all paper figures"""
    print("\n" + "="*60)
    print("MetaLoRA Paper Figure Generator")
    print("="*60 + "\n")
    
    print("Generating figures from verified data...")
    print()
    
    try:
        create_figure1_importance_distribution()
        create_figure2_cross_budget_performance()
        create_figure3_calibration_results()
        
        print("\n" + "="*60)
        print("✓ All figures generated successfully!")
        print("="*60)
        print("\nOutput files created:")
        print("  - figure1_importance_distribution.pdf/png")
        print("  - figure2_cross_budget_performance.pdf/png")
        print("  - figure3_calibration_results.pdf/png")
        print("\nUse PDF versions for paper submission (vector graphics)")
        print("Use PNG versions for quick preview")
        
    except Exception as e:
        print(f"\n✗ Error generating figures: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()