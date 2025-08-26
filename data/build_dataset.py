import csv
import os

def save_results_to_csv(results, features, path='data/mitigation_results.csv'):
    os.makedirs('data', exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'strategy', 'accuracy', 'circuit_depth',
            'num_gates', 'entanglement', 'num_qubits'
        ])
        writer.writeheader()
        for strat, acc in results.items():
            row = features.copy()
            row.update({'strategy': strat, 'accuracy': acc})
            writer.writerow(row)
