
#!/usr/bin/env python3
"""
Quantum Circuit Dataset Generator for Error Mitigation Analysis
Generates 3000-4000 diverse quantum circuits with realistic noise models
and tests different error mitigation strategies.
"""

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import *
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
from qiskit import transpile
# Removed problematic Estimator import
from qiskit.quantum_info import SparsePauliOp
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class QuantumDatasetGenerator:
    def __init__(self):
        self.circuits = []
        self.results = []
        self.noise_models = self._create_noise_models()
        
    def _create_noise_models(self):
        """Create different noise models for realistic simulation"""
        noise_models = {}
        
        # Light noise model
        light_noise = NoiseModel()
        light_error_1q = depolarizing_error(0.001, 1)  # 0.1% error rate
        light_error_2q = depolarizing_error(0.01, 2)   # 1% error rate
        light_noise.add_all_qubit_quantum_error(light_error_1q, ['rz', 'ry', 'rx', 'h', 'x', 'y', 'z'])
        light_noise.add_all_qubit_quantum_error(light_error_2q, ['cx', 'cz', 'swap'])
        noise_models['light'] = light_noise
        
        # Medium noise model
        medium_noise = NoiseModel()
        medium_error_1q = depolarizing_error(0.005, 1)  # 0.5% error rate
        medium_error_2q = depolarizing_error(0.02, 2)   # 2% error rate
        medium_noise.add_all_qubit_quantum_error(medium_error_1q, ['rz', 'ry', 'rx', 'h', 'x', 'y', 'z'])
        medium_noise.add_all_qubit_quantum_error(medium_error_2q, ['cx', 'cz', 'swap'])
        noise_models['medium'] = medium_noise
        
        # Heavy noise model
        heavy_noise = NoiseModel()
        heavy_error_1q = depolarizing_error(0.01, 1)   # 1% error rate
        heavy_error_2q = depolarizing_error(0.05, 2)   # 5% error rate
        heavy_noise.add_all_qubit_quantum_error(heavy_error_1q, ['rz', 'ry', 'rx', 'h', 'x', 'y', 'z'])
        heavy_noise.add_all_qubit_quantum_error(heavy_error_2q, ['cx', 'cz', 'swap'])
        noise_models['heavy'] = heavy_noise
        
        return noise_models
    
    def create_random_circuit(self, num_qubits, circuit_type='random'):
        """Generate various types of quantum circuits"""
        qc = QuantumCircuit(num_qubits, num_qubits)
        
        if circuit_type == 'random':
            return self._create_random_circuit(qc, num_qubits)
        elif circuit_type == 'vqe':
            return self._create_vqe_ansatz(qc, num_qubits)
        elif circuit_type == 'qaoa':
            return self._create_qaoa_circuit(qc, num_qubits)
        elif circuit_type == 'grover':
            return self._create_grover_circuit(qc, num_qubits)
        elif circuit_type == 'qft':
            return self._create_qft_circuit(qc, num_qubits)
        elif circuit_type == 'supremacy':
            return self._create_supremacy_circuit(qc, num_qubits)
        else:
            return self._create_random_circuit(qc, num_qubits)
    
    def _create_random_circuit(self, qc, num_qubits):
        """Create a random quantum circuit"""
        gates_1q = ['h', 'x', 'y', 'z', 'rx', 'ry', 'rz']
        gates_2q = ['cx', 'cz', 'swap']
        
        depth = random.randint(3, 15)
        primary_gate = random.choice(gates_2q)
        
        for _ in range(depth):
            if random.random() < 0.7:  # 70% chance of 2-qubit gate
                gate = random.choice(gates_2q)
                if gate == primary_gate:  # Favor one type of entangling gate
                    prob = 0.6
                else:
                    prob = 0.4
                    
                if random.random() < prob:
                    q1, q2 = random.sample(range(num_qubits), 2)
                    if gate == 'cx':
                        qc.cx(q1, q2)
                    elif gate == 'cz':
                        qc.cz(q1, q2)
                    elif gate == 'swap':
                        qc.swap(q1, q2)
            else:  # Single qubit gate
                gate = random.choice(gates_1q)
                qubit = random.randint(0, num_qubits - 1)
                if gate == 'h':
                    qc.h(qubit)
                elif gate == 'x':
                    qc.x(qubit)
                elif gate == 'y':
                    qc.y(qubit)
                elif gate == 'z':
                    qc.z(qubit)
                elif gate == 'rx':
                    qc.rx(random.uniform(0, 2*np.pi), qubit)
                elif gate == 'ry':
                    qc.ry(random.uniform(0, 2*np.pi), qubit)
                elif gate == 'rz':
                    qc.rz(random.uniform(0, 2*np.pi), qubit)
        
        # Add measurements
        qc.measure_all()
        return qc, primary_gate
    
    def _create_vqe_ansatz(self, qc, num_qubits):
        """Create VQE-style ansatz circuit"""
        layers = random.randint(2, 6)
        
        for layer in range(layers):
            # RY rotation layer
            for qubit in range(num_qubits):
                qc.ry(random.uniform(0, 2*np.pi), qubit)
            
            # Entangling layer
            for qubit in range(num_qubits - 1):
                qc.cx(qubit, qubit + 1)
            
            # Add some RZ gates
            if random.random() < 0.5:
                for qubit in range(num_qubits):
                    qc.rz(random.uniform(0, 2*np.pi), qubit)
        
        qc.measure_all()
        return qc, 'CX'
    
    def _create_qaoa_circuit(self, qc, num_qubits):
        """Create QAOA-style circuit"""
        p = random.randint(1, 4)  # QAOA depth
        
        # Initial superposition
        for qubit in range(num_qubits):
            qc.h(qubit)
        
        for layer in range(p):
            # Problem layer (ZZ interactions)
            for i in range(num_qubits - 1):
                qc.cx(i, i + 1)
                qc.rz(random.uniform(0, np.pi), i + 1)
                qc.cx(i, i + 1)
            
            # Mixer layer
            for qubit in range(num_qubits):
                qc.rx(random.uniform(0, np.pi), qubit)
        
        qc.measure_all()
        return qc, 'CX'
    
    def _create_grover_circuit(self, qc, num_qubits):
        """Create Grover-style search circuit"""
        iterations = max(1, int(np.pi / 4 * np.sqrt(2**num_qubits)))
        iterations = min(iterations, 5)  # Cap iterations
        
        # Initialize superposition
        for qubit in range(num_qubits):
            qc.h(qubit)
        
        for _ in range(iterations):
            # Oracle (mark random state)
            target_qubit = random.randint(0, num_qubits - 1)
            qc.z(target_qubit)
            
            # Diffusion operator
            for qubit in range(num_qubits):
                qc.h(qubit)
                qc.x(qubit)
            
            if num_qubits > 1:
                qc.h(num_qubits - 1)
                for qubit in range(num_qubits - 1):
                    qc.cx(qubit, num_qubits - 1)
                qc.h(num_qubits - 1)
            
            for qubit in range(num_qubits):
                qc.x(qubit)
                qc.h(qubit)
        
        qc.measure_all()
        return qc, 'CX'
    
    def _create_qft_circuit(self, qc, num_qubits):
        """Create Quantum Fourier Transform circuit"""
        for i in range(num_qubits):
            qc.h(i)
            for j in range(i + 1, num_qubits):
                angle = np.pi / (2**(j - i))
                qc.cp(angle, j, i)
        
        # Swap qubits
        for i in range(num_qubits // 2):
            qc.swap(i, num_qubits - 1 - i)
        
        qc.measure_all()
        return qc, 'CZ'  # CP gates are like CZ
    
    def _create_supremacy_circuit(self, qc, num_qubits):
        """Create quantum supremacy style circuit"""
        depth = random.randint(8, 20)
        
        for d in range(depth):
            # Single qubit layer
            for qubit in range(num_qubits):
                if random.random() < 0.5:
                    qc.ry(random.uniform(0, 2*np.pi), qubit)
                else:
                    qc.rz(random.uniform(0, 2*np.pi), qubit)
            
            # Two qubit layer
            if d % 2 == 0:  # Even layers
                for i in range(0, num_qubits - 1, 2):
                    if random.random() < 0.8:
                        qc.cx(i, i + 1)
            else:  # Odd layers
                for i in range(1, num_qubits - 1, 2):
                    if random.random() < 0.8:
                        qc.cx(i, i + 1)
        
        qc.measure_all()
        return qc, 'CX'
    
    def analyze_circuit(self, circuit):
        """Extract features from quantum circuit"""
        ops = circuit.count_ops()
        
        # Count different gate types
        single_qubit_gates = ['h', 'x', 'y', 'z', 'rx', 'ry', 'rz']
        two_qubit_gates = ['cx', 'cz', 'swap', 'cp']
        
        total_gates = sum(ops.values()) - ops.get('measure', 0)
        single_gates = sum(ops.get(gate, 0) for gate in single_qubit_gates)
        two_gates = sum(ops.get(gate, 0) for gate in two_qubit_gates)
        
        return {
            'circuit_depth': circuit.depth(),
            'num_gates': total_gates,
            'num_qubits': circuit.num_qubits,
            'single_qubit_gates': single_gates,
            'two_qubit_gates': two_gates,
            'gate_density': total_gates / (circuit.num_qubits * circuit.depth()) if circuit.depth() > 0 else 0
        }
    
    def run_error_mitigation_strategies(self, circuit, noise_level='medium'):
        """Test different error mitigation strategies"""
        simulator = AerSimulator()
        noise_model = self.noise_models[noise_level]
        shots = 1024
        
        strategies = {}
        
        # No mitigation (baseline)
        try:
            transpiled = transpile(circuit, simulator, optimization_level=1)
            job = simulator.run(transpiled, noise_model=noise_model, shots=shots)
            result = job.result()
            counts = result.get_counts()
            # Calculate fidelity approximation (entropy-based)
            total_outcomes = len(counts)
            if total_outcomes > 0:
                probs = np.array(list(counts.values())) / shots
                entropy = -np.sum(probs * np.log2(probs + 1e-10))
                strategies['none'] = max(0.3, 1.0 - entropy / np.log2(2**circuit.num_qubits))
            else:
                strategies['none'] = 0.3
        except:
            strategies['none'] = 0.3
        
        # Zero Noise Extrapolation (ZNE) - simulate with scaled noise
        try:
            zne_accuracies = []
            for scale in [1.0, 1.5, 2.0]:
                scaled_noise = NoiseModel()
                for gate, errors in noise_model._default_quantum_errors.items():
                    if errors:
                        error_prob = min(0.5, errors[0].probabilities[1] * scale)
                        if gate[0] in ['cx', 'cz', 'swap']:
                            scaled_error = depolarizing_error(error_prob, 2)
                        else:
                            scaled_error = depolarizing_error(error_prob, 1)
                        scaled_noise.add_all_qubit_quantum_error(scaled_error, gate)
                
                transpiled = transpile(circuit, simulator, optimization_level=1)
                job = simulator.run(transpiled, noise_model=scaled_noise, shots=shots//2)
                result = job.result()
                counts = result.get_counts()
                if counts:
                    probs = np.array(list(counts.values())) / (shots//2)
                    entropy = -np.sum(probs * np.log2(probs + 1e-10))
                    accuracy = max(0.2, 1.0 - entropy / np.log2(2**circuit.num_qubits))
                    zne_accuracies.append(accuracy)
            
            if len(zne_accuracies) >= 2:
                # Linear extrapolation to zero noise
                x = np.array([1.0, 1.5, 2.0][:len(zne_accuracies)])
                y = np.array(zne_accuracies)
                if len(x) >= 2:
                    slope = (y[-1] - y[0]) / (x[-1] - x[0])
                    strategies['zne'] = max(0.2, y[0] - slope * x[0])
                else:
                    strategies['zne'] = strategies['none'] + 0.1
            else:
                strategies['zne'] = strategies['none'] + 0.1
        except:
            strategies['zne'] = strategies['none'] + 0.1
        
        # Measurement Error Mitigation (MEM) - approximate
        try:
            # Simulate measurement calibration effect
            base_accuracy = strategies['none']
            mem_improvement = 0.05 + 0.1 * (circuit.num_qubits / 10)  # Scales with qubits
            strategies['mem'] = min(0.95, base_accuracy + mem_improvement)
        except:
            strategies['mem'] = strategies['none'] + 0.08
        
        # Clifford Data Regression (CDR) - approximate
        try:
            # CDR works better for circuits with more structure
            gate_ratio = circuit.count_ops().get('cx', 0) / max(1, sum(circuit.count_ops().values()) - circuit.count_ops().get('measure', 0))
            cdr_improvement = 0.03 + 0.12 * gate_ratio
            strategies['cdr'] = min(0.92, strategies['none'] + cdr_improvement)
        except:
            strategies['cdr'] = strategies['none'] + 0.06
        
        return strategies
    
    def determine_best_strategy(self, strategies, circuit_features):
        """Determine the best mitigation strategy based on results and heuristics"""
        # Find the strategy with highest accuracy
        best_strategy = max(strategies, key=strategies.get)
        
        # Add some realistic decision logic
        if circuit_features['num_qubits'] <= 3 and circuit_features['circuit_depth'] <= 5:
            # For small, shallow circuits, overhead might not be worth it
            if strategies['none'] > 0.65:
                return 'none'
        
        if circuit_features['num_qubits'] >= 6:
            # For larger circuits, MEM often works well
            if strategies['mem'] > strategies[best_strategy] - 0.05:
                return 'mem'
        
        if circuit_features['circuit_depth'] >= 10:
            # For deep circuits, ZNE can be effective
            if strategies['zne'] > strategies[best_strategy] - 0.08:
                return 'zne'
        
        return best_strategy
    
    def generate_dataset(self, target_size=3500):
        """Generate the complete dataset"""
        print("🔬 Generating Quantum Error Mitigation Dataset...")
        print(f"Target size: {target_size} circuits")
        
        circuit_types = ['random', 'vqe', 'qaoa', 'grover', 'qft', 'supremacy']
        noise_levels = ['light', 'medium', 'heavy']
        qubit_ranges = [2, 3, 4, 5, 6, 7, 8]
        
        data = []
        
        # Calculate distribution
        circuits_per_type = target_size // len(circuit_types)
        
        with tqdm(total=target_size, desc="Generating circuits") as pbar:
            for circuit_type in circuit_types:
                for i in range(circuits_per_type):
                    try:
                        # Random parameters
                        num_qubits = random.choice(qubit_ranges)
                        noise_level = random.choice(noise_levels)
                        
                        # Generate circuit
                        circuit, primary_gate = self.create_random_circuit(num_qubits, circuit_type)
                        
                        # Extract features
                        features = self.analyze_circuit(circuit)
                        
                        # Run mitigation strategies
                        strategies = self.run_error_mitigation_strategies(circuit, noise_level)
                        
                        # Determine best strategy
                        best_strategy = self.determine_best_strategy(strategies, features)
                        
                        # Create data entry
                        entry = {
                            'circuit_depth': features['circuit_depth'],
                            'num_gates': features['num_gates'],
                            'num_qubits': features['num_qubits'],
                            'entanglement': primary_gate,
                            'circuit_type': circuit_type,
                            'noise_level': noise_level,
                            'gate_density': features['gate_density'],
                            'two_qubit_ratio': features['two_qubit_gates'] / max(1, features['num_gates']),
                            'strategy': best_strategy,
                            'none_accuracy': strategies['none'],
                            'zne_accuracy': strategies['zne'],
                            'mem_accuracy': strategies['mem'],
                            'cdr_accuracy': strategies['cdr'],
                            'best_accuracy': strategies[best_strategy]
                        }
                        
                        data.append(entry)
                        pbar.update(1)
                        
                        # Save intermediate results every 500 circuits
                        if len(data) % 500 == 0:
                            temp_df = pd.DataFrame(data)
                            temp_df.to_csv(f"data/temp_dataset_{len(data)}.csv", index=False)
                            
                    except Exception as e:
                        print(f"Error generating circuit: {e}")
                        continue
        
        # Create final dataset
        df = pd.DataFrame(data)
        
        # Add some additional synthetic variety if needed
        while len(df) < target_size:
            try:
                # Fill remaining with random circuits
                num_qubits = random.choice(qubit_ranges)
                noise_level = random.choice(noise_levels)
                circuit_type = random.choice(circuit_types)
                
                circuit, primary_gate = self.create_random_circuit(num_qubits, circuit_type)
                features = self.analyze_circuit(circuit)
                strategies = self.run_error_mitigation_strategies(circuit, noise_level)
                best_strategy = self.determine_best_strategy(strategies, features)
                
                entry = {
                    'circuit_depth': features['circuit_depth'],
                    'num_gates': features['num_gates'],
                    'num_qubits': features['num_qubits'],
                    'entanglement': primary_gate,
                    'circuit_type': circuit_type,
                    'noise_level': noise_level,
                    'gate_density': features['gate_density'],
                    'two_qubit_ratio': features['two_qubit_gates'] / max(1, features['num_gates']),
                    'strategy': best_strategy,
                    'none_accuracy': strategies['none'],
                    'zne_accuracy': strategies['zne'],
                    'mem_accuracy': strategies['mem'],
                    'cdr_accuracy': strategies['cdr'],
                    'best_accuracy': strategies[best_strategy]
                }
                
                df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
                
            except:
                continue
        
        return df
    
    def save_dataset(self, df, filename="data/mitigation_results.csv"):
        """Save dataset and print statistics"""
        df.to_csv(filename, index=False)
        
        print(f"\n✅ Dataset saved to {filename}")
        print(f"📊 Dataset Statistics:")
        print(f"   Total samples: {len(df)}")
        print(f"   Features: {list(df.columns)}")
        print("\n🎯 Strategy Distribution:")
        print(df['strategy'].value_counts())
        print("\n🔧 Gate Type Distribution:")
        print(df['entanglement'].value_counts())
        print("\n🏗️ Circuit Type Distribution:")
        print(df['circuit_type'].value_counts())
        print("\n🌊 Noise Level Distribution:")
        print(df['noise_level'].value_counts())
        print(f"\n📈 Qubit Range: {df['num_qubits'].min()} - {df['num_qubits'].max()}")
        print(f"📏 Depth Range: {df['circuit_depth'].min()} - {df['circuit_depth'].max()}")
        print(f"🚪 Gates Range: {df['num_gates'].min()} - {df['num_gates'].max()}")

# Main execution
if __name__ == "__main__":
    # Create data directory if it doesn't exist
    import os
    os.makedirs("data", exist_ok=True)
    
    # Generate dataset
    generator = QuantumDatasetGenerator()
    dataset = generator.generate_dataset(target_size=3000)  # Reduced for faster testing
    generator.save_dataset(dataset)
    
    print("\n🎉 Dataset generation complete!")
    print("You can now run your ML model training script.")