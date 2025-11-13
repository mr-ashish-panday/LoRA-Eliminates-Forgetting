import json
import numpy as np

with open('results/phase4/phase4_summary.json', 'r') as f:
    data = json.load(f)

print("=" * 80)
print("PHASE 4 ANALYSIS: Module Selection")
print("=" * 80)

# Organize by task and module type
results_by_task = {}
for key, result in data['results'].items():
    task = result['task']
    module_type = result['module_type']
    
    if task not in results_by_task:
        results_by_task[task] = {'qv': [], 'qkv': [], 'all_linear': []}
    
    results_by_task[task][module_type].append({
        'acc': result['metrics']['accuracy'],
        'params': result['model_info']['trainable_params'],
        'mem': result['metrics']['peak_gpu_memory_gb'],
        'seed': result['seed']
    })

print("\n📊 ACCURACY & PARAMETERS BY MODULE TYPE:\n")
print(f"{'Task':<8} {'Modules':<12} {'Avg Acc':<10} {'Params':<12} {'Memory':<10} {'Seeds'}")
print("-" * 80)

for task in ['rte', 'mrpc', 'cola', 'sst2']:
    for module_type in ['qv', 'qkv', 'all_linear']:
        results = results_by_task[task][module_type]
        avg_acc = np.mean([r['acc'] for r in results])
        avg_params = int(np.mean([r['params'] for r in results]))
        avg_mem = np.mean([r['mem'] for r in results])
        accs = [r['acc'] for r in results]
        
        print(f"{task:<8} {module_type:<12} {avg_acc:.3f} ({avg_acc*100:.1f}%)  {avg_params:>10,}  {avg_mem:.2f} GB    {accs}")

print("\n" + "=" * 80)
print("OVERALL COMPARISON:")
print("=" * 80)

# Overall comparison
overall = {}
for module_type in ['qv', 'qkv', 'all_linear']:
    accs = []
    params_list = []
    for task in ['rte', 'mrpc', 'cola', 'sst2']:
        for r in results_by_task[task][module_type]:
            accs.append(r['acc'])
            params_list.append(r['params'])
    
    overall[module_type] = {
        'avg_acc': np.mean(accs),
        'avg_params': int(np.mean(params_list))
    }

print(f"\nQuery+Value (qv):        Acc = {overall['qv']['avg_acc']:.3f} ({overall['qv']['avg_acc']*100:.1f}%)")
print(f"                         Params = {overall['qv']['avg_params']:,}")

print(f"\nQuery+Key+Value (qkv):   Acc = {overall['qkv']['avg_acc']:.3f} ({overall['qkv']['avg_acc']*100:.1f}%)")
print(f"                         Params = {overall['qkv']['avg_params']:,}")
print(f"                         Improvement: {(overall['qkv']['avg_acc'] - overall['qv']['avg_acc'])*100:+.2f}%")

print(f"\nAll Linear (all_linear): Acc = {overall['all_linear']['avg_acc']:.3f} ({overall['all_linear']['avg_acc']*100:.1f}%)")
print(f"                         Params = {overall['all_linear']['avg_params']:,}")
print(f"                         Improvement: {(overall['all_linear']['avg_acc'] - overall['qv']['avg_acc'])*100:+.2f}%")

print(f"\n{'='*80}")
print(f"VERDICT:")
print(f"{'='*80}")

# Find winner
winner = max(overall.items(), key=lambda x: x[1]['avg_acc'])
baseline_acc = overall['qv']['avg_acc']

if winner[0] == 'qv':
    print(f"✅ Query+Value (baseline) is OPTIMAL!")
    print(f"   More modules don't help - stick with qv!")
else:
    improvement = (winner[1]['avg_acc'] - baseline_acc) * 100
    param_increase = ((winner[1]['avg_params'] - overall['qv']['avg_params']) / overall['qv']['avg_params']) * 100
    print(f"✅ {winner[0].upper()} wins with {improvement:+.2f}% improvement")
    print(f"   But requires {param_increase:.1f}% more parameters")
    
    if improvement > 1.0:
        print(f"   Verdict: Worth it! Use {winner[0]}")
    else:
        print(f"   Verdict: Marginal gain, qv is more efficient")

print(f"{'='*80}")

