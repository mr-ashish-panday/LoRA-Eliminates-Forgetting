import json
import numpy as np

with open('results/phase3/phase3_summary.json', 'r') as f:
    data = json.load(f)

print("=" * 80)
print("PHASE 3 ANALYSIS: Hierarchical vs Uniform LoRA")
print("=" * 80)

# Organize by task and strategy
results_by_task = {}
for key, result in data['results'].items():
    task = result['task']
    strategy = result['strategy']
    
    if task not in results_by_task:
        results_by_task[task] = {'progressive': [], 'uniform': []}
    
    results_by_task[task][strategy].append({
        'acc': result['metrics']['accuracy'],
        'mem': result['metrics']['peak_gpu_memory_gb'],
        'params': result['model_info']['trainable_params'],
        'seed': result['seed']
    })

print("\n📊 ACCURACY & PARAMETERS BY TASK:\n")
print(f"{'Task':<8} {'Strategy':<12} {'Avg Acc':<10} {'Params':<12} {'Memory':<10} {'Seeds'}")
print("-" * 80)

for task in ['rte', 'mrpc', 'cola', 'sst2']:
    for strategy in ['progressive', 'uniform']:
        results = results_by_task[task][strategy]
        avg_acc = np.mean([r['acc'] for r in results])
        avg_params = int(np.mean([r['params'] for r in results]))
        avg_mem = np.mean([r['mem'] for r in results])
        accs = [r['acc'] for r in results]
        
        print(f"{task:<8} {strategy:<12} {avg_acc:.3f} ({avg_acc*100:.1f}%)  {avg_params:>8,}  {avg_mem:.2f} GB    {accs}")

print("\n" + "=" * 80)
print("COMPARISON:")
print("=" * 80)

# Overall comparison
prog_accs = []
unif_accs = []

for task in ['rte', 'mrpc', 'cola', 'sst2']:
    prog_accs.extend([r['acc'] for r in results_by_task[task]['progressive']])
    unif_accs.extend([r['acc'] for r in results_by_task[task]['uniform']])

print(f"\nProgressive (r=12): Avg Acc = {np.mean(prog_accs):.3f} ({np.mean(prog_accs)*100:.1f}%)")
print(f"                    Params = 443,906 (0.40%)")
print(f"                    Memory = 1.32 GB")

print(f"\nUniform (r=8):      Avg Acc = {np.mean(unif_accs):.3f} ({np.mean(unif_accs)*100:.1f}%)")
print(f"                    Params = 296,450 (0.27%)")
print(f"                    Memory = 1.32 GB")

diff = (np.mean(prog_accs) - np.mean(unif_accs)) * 100
param_increase = ((443906 - 296450) / 296450) * 100

print(f"\n{'='*80}")
print(f"FINDING:")
print(f"{'='*80}")
print(f"Progressive improves accuracy by: {diff:+.2f}%")
print(f"But requires {param_increase:.1f}% more parameters")
print(f"Verdict: {'Worth it!' if diff > 1 else 'Not worth the extra params'}")
print(f"{'='*80}")

