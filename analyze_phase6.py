import json
import numpy as np

with open('results/phase6/phase6_summary.json', 'r') as f:
    data = json.load(f)

print("=" * 80)
print("PHASE 6 ANALYSIS: Best Technique Combinations")
print("=" * 80)

# Organize by task and combination
results_by_task = {}
for key, result in data['results'].items():
    task = result['task']
    combo = result['combination']
    
    if task not in results_by_task:
        results_by_task[task] = {'qv_4bit': [], 'qv_8bit': [], 'all_4bit': [], 'all_8bit': []}
    
    results_by_task[task][combo].append({
        'acc': result['metrics']['accuracy'],
        'mem': result['metrics']['peak_gpu_memory_gb'],
        'params': result['model_info']['trainable_params'],
        'seed': result['seed']
    })

print("\n📊 RESULTS BY COMBINATION:\n")
print(f"{'Task':<8} {'Combination':<12} {'Avg Acc':<10} {'Memory':<10} {'Params':<12} {'Seeds'}")
print("-" * 80)

for task in ['rte', 'mrpc', 'cola', 'sst2']:
    for combo in ['qv_4bit', 'qv_8bit', 'all_4bit', 'all_8bit']:
        results = results_by_task[task][combo]
        if results:
            avg_acc = np.mean([r['acc'] for r in results])
            avg_mem = np.mean([r['mem'] for r in results])
            avg_params = int(np.mean([r['params'] for r in results]))
            accs = [r['acc'] for r in results]
            
            print(f"{task:<8} {combo:<12} {avg_acc:.3f} ({avg_acc*100:.1f}%)  {avg_mem:.2f} GB   {avg_params:>10,}  {accs}")

print("\n" + "=" * 80)
print("OVERALL COMPARISON:")
print("=" * 80)

# Overall stats
overall = {}
for combo in ['qv_4bit', 'qv_8bit', 'all_4bit', 'all_8bit']:
    accs = []
    mems = []
    params_list = []
    
    for task in ['rte', 'mrpc', 'cola', 'sst2']:
        for r in results_by_task[task][combo]:
            accs.append(r['acc'])
            mems.append(r['mem'])
            params_list.append(r['params'])
    
    overall[combo] = {
        'avg_acc': np.mean(accs),
        'avg_mem': np.mean(mems),
        'avg_params': int(np.mean(params_list))
    }

print(f"\n{'Combination':<15} {'Avg Acc':<12} {'Memory':<12} {'Params':<15}")
print("-" * 80)

for combo in ['qv_4bit', 'qv_8bit', 'all_4bit', 'all_8bit']:
    acc = overall[combo]['avg_acc']
    mem = overall[combo]['avg_mem']
    params = overall[combo]['avg_params']
    print(f"{combo:<15} {acc:.3f} ({acc*100:.1f}%)  {mem:.2f} GB      {params:>10,}")

print(f"\n{'='*80}")
print(f"WINNER:")
print(f"{'='*80}")

# Find best by accuracy
best_acc = max(overall.items(), key=lambda x: x[1]['avg_acc'])
print(f"\n🏆 Best Accuracy: {best_acc[0].upper()}")
print(f"   Accuracy: {best_acc[1]['avg_acc']:.3f} ({best_acc[1]['avg_acc']*100:.1f}%)")
print(f"   Memory: {best_acc[1]['avg_mem']:.2f} GB")
print(f"   Params: {best_acc[1]['avg_params']:,}")

# Find best efficiency (accuracy per GB memory)
efficiency = {k: v['avg_acc'] / v['avg_mem'] for k, v in overall.items()}
best_eff = max(efficiency.items(), key=lambda x: x[1])
print(f"\n💡 Most Efficient: {best_eff[0].upper()}")
print(f"   Efficiency Score: {best_eff[1]:.2f} (acc/GB)")
print(f"   Accuracy: {overall[best_eff[0]]['avg_acc']:.3f}")
print(f"   Memory: {overall[best_eff[0]]['avg_mem']:.2f} GB")

print(f"\n{'='*80}")
print("RECOMMENDATION:")
print(f"{'='*80}")

if best_acc[0] == best_eff[0]:
    print(f"✅ Use {best_acc[0].upper()} - Best in both accuracy AND efficiency!")
else:
    acc_diff = (best_acc[1]['avg_acc'] - overall[best_eff[0]]['avg_acc']) * 100
    mem_diff = best_acc[1]['avg_mem'] - overall[best_eff[0]]['avg_mem']
    print(f"Trade-off decision:")
    print(f"  - {best_acc[0].upper()}: +{acc_diff:.1f}% accuracy, +{mem_diff:.2f} GB memory")
    print(f"  - {best_eff[0].upper()}: Most efficient")
    
    if acc_diff < 2:
        print(f"\n✅ Recommendation: Use {best_eff[0].upper()} (marginal accuracy loss, much more efficient)")
    else:
        print(f"\n✅ Recommendation: Use {best_acc[0].upper()} (significant accuracy gain worth the memory)")

print(f"{'='*80}")

