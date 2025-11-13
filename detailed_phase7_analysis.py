import json

with open('results/phase7/phase7_summary.json', 'r') as f:
    data = json.load(f)

print("\n" + "="*80)
print("PER-TASK FORGETTING BREAKDOWN:")
print("="*80)

for method in ['full', 'lora', '4bit']:
    print(f"\n{method.upper()}:")
    
    # Get all seeds for this method
    forgetting_by_task = {'rte': [], 'mrpc': [], 'cola': []}
    
    for key, result in data['results'].items():
        if result['method'] == method:
            per_task = result['forgetting_metrics']['forgetting_per_task']
            for task, value in per_task.items():
                forgetting_by_task[task].append(value)
    
    for task in ['rte', 'mrpc', 'cola']:
        avg = sum(forgetting_by_task[task]) / len(forgetting_by_task[task])
        print(f"  {task:6} forgot: {avg:6.3f} ({avg*100:5.1f}%) - seeds: {forgetting_by_task[task]}")

