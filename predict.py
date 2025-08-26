import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

class QuantumErrorMitigationPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = []
        
    def load_trained_model(self):
        """Load the trained model and preprocessing objects"""
        print("📦 LOADING TRAINED MODEL")
        print("=" * 30)
        
        try:
            # Load model and preprocessors
            self.model = joblib.load('best_quantum_error_model.pkl')
            self.scaler = joblib.load('feature_scaler.pkl')
            self.label_encoder = joblib.load('label_encoder.pkl')
            
            # Load feature names
            with open('feature_names.txt', 'r') as f:
                self.feature_names = [line.strip() for line in f.readlines()]
            
            print("✅ Model loaded successfully")
            print(f"📋 Features: {self.feature_names}")
            print(f"🎯 Classes: {list(self.label_encoder.classes_)}")
            
            return True
            
        except FileNotFoundError as e:
            print(f"❌ Error loading model files: {e}")
            print("💡 Make sure you've run the training script first")
            return False
    
    def _preprocess_params(self, circuit_params):
        """Ensure circuit params have all 9 features"""
        if len(circuit_params) == 7:
            # Auto-fill missing features
            circuit_type = 0  # default category (e.g., optimization)
            # simple heuristic for complexity score
            depth, gates, qubits, ent, noise, density, twoq = circuit_params
            complexity_score = round((depth * density + gates * noise + qubits * ent) / (gates + 1), 2)
            return circuit_params + [circuit_type, complexity_score]
        return circuit_params

    def predict_single_circuit(self, circuit_params, circuit_description=""):
        """Predict strategy for a single circuit"""
        if self.model is None:
            print("❌ Model not loaded. Call load_trained_model() first.")
            return None, None
        
        # Auto-complete params if only 7 provided
        circuit_params = self._preprocess_params(circuit_params)
        
        print(f"\n🔍 ANALYZING CIRCUIT: {circuit_description}")
        print("-" * 40)
        
        # Validate input
        if len(circuit_params) != len(self.feature_names):
            print(f"❌ Error: Expected {len(self.feature_names)} parameters")
            print(f"📋 Required: {self.feature_names}")
            return None, None
        
        # Display circuit parameters
        print("📊 Circuit Parameters:")
        for param_name, value in zip(self.feature_names, circuit_params):
            print(f"   • {param_name}: {value}")
        
        # Scale and predict
        features = pd.DataFrame([circuit_params], columns=self.feature_names)
        features_scaled = self.scaler.transform(features)

        
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Convert to strategy name
        strategy = self.label_encoder.inverse_transform([prediction])[0]
        confidence = probabilities[prediction]
        
        print(f"\n🚀 RECOMMENDED STRATEGY: {strategy.upper()}")
        print(f"🎯 Confidence: {confidence:.1%}")
        
        # Show all probabilities
        print(f"\n📊 Strategy Probabilities:")
        for i, class_name in enumerate(self.label_encoder.classes_):
            bar = "█" * int(probabilities[i] * 20)
            print(f"   • {class_name.upper():<4}: {probabilities[i]:.1%} {bar}")
        
        # Strategy explanations
        strategy_info = {
            'zne': 'Zero Noise Extrapolation - Extrapolates to zero noise limit',
            'mem': 'Measurement Error Mitigation - Corrects readout errors',
            'cdr': 'Clifford Data Regression - Uses Clifford circuits for characterization',
            'none': 'No Mitigation - Run without error correction'
        }
        
        if strategy.lower() in strategy_info:
            print(f"\n💡 About {strategy.upper()}: {strategy_info[strategy.lower()]}")
        
        return strategy, probabilities
    
    def predict_multiple_circuits(self, circuits_data):
        """Predict strategies for multiple circuits"""
        print(f"\n🔄 BATCH PREDICTION ({len(circuits_data)} circuits)")
        print("=" * 50)
        
        results = []
        
        for i, circuit_info in enumerate(circuits_data):
            params = circuit_info['params']
            description = circuit_info.get('description', f'Circuit {i+1}')
            
            strategy, probs = self.predict_single_circuit(params, description)
            
            if strategy is not None:
                results.append({
                    'circuit_id': i+1,
                    'description': description,
                    'recommended_strategy': strategy,
                    'confidence': probs[self.label_encoder.transform([strategy])[0]]
                })
            
            print("\n" + "="*50)
        
        return results
    
    def evaluate_on_test_data(self, test_data_path):
        """Evaluate model on new test data"""
        print(f"\n📊 EVALUATING ON TEST DATA")
        print("=" * 35)
        
        # Load test data
        test_df = pd.read_csv(test_data_path)
        print(f"✅ Loaded {len(test_df)} test samples")
        
        # Prepare features
        X_test = test_df[self.feature_names]
        y_test = test_df['strategy']
        
        # Scale features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_names = self.label_encoder.inverse_transform(y_pred)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred_names)
        print(f"🎯 Test Accuracy: {accuracy:.4f}")
        
        # Classification report
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred_names))
        
        return accuracy, y_pred_names

def demo_predictions():
    """Demonstrate model predictions with various circuit types"""
    
    # Initialize predictor
    predictor = QuantumErrorMitigationPredictor()
    
    # Load model
    if not predictor.load_trained_model():
        return
    
    print(f"\n🎭 DEMONSTRATION: VARIOUS QUANTUM CIRCUITS")
    print("=" * 55)
    
    # Define test circuits with only 7 params (auto-completed to 9)
    test_circuits = [
        {'params': [10, 25, 4, 0.3, 0.05, 0.625, 0.2], 'description': 'Low-noise QAOA circuit'},
        {'params': [50, 150, 6, 0.8, 0.15, 0.75, 0.4], 'description': 'High-noise VQE circuit'},
        {'params': [5, 12, 3, 0.1, 0.02, 0.4, 0.1], 'description': 'Simple quantum teleportation'},
        {'params': [100, 300, 8, 0.9, 0.25, 0.9, 0.6], 'description': 'Complex quantum simulation'},
        {'params': [30, 80, 5, 0.5, 0.08, 0.6, 0.25], 'description': 'Medium-depth optimization circuit'}
    ]
    
    # Make predictions
    results = predictor.predict_multiple_circuits(test_circuits)
    
    # Summary
    print(f"\n📈 PREDICTION SUMMARY")
    print("=" * 25)
    
    strategy_counts = {}
    for result in results:
        strategy = result['recommended_strategy']
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    
    print("📊 Strategy Distribution:")
    for strategy, count in strategy_counts.items():
        print(f"   • {strategy.upper()}: {count} circuits")
    
    avg_confidence = np.mean([r['confidence'] for r in results])
    print(f"🎯 Average Confidence: {avg_confidence:.1%}")

def interactive_prediction():
    """Interactive circuit parameter input"""
    print(f"\n🎮 INTERACTIVE CIRCUIT ANALYSIS")
    print("=" * 35)
    
    predictor = QuantumErrorMitigationPredictor()
    
    if not predictor.load_trained_model():
        return
    
    print(f"\n📋 Enter circuit parameters (7 or 9 values):")
    print(f"Required features: {predictor.feature_names}")
    
    try:
        params = []
        for feature in predictor.feature_names[:7]:  # only first 7 required from user
            value = float(input(f"Enter {feature}: "))
            params.append(value)
        
        description = input("Circuit description (optional): ") or "User-defined circuit"
        
        # Make prediction (auto-fills missing 2 features)
        strategy, probs = predictor.predict_single_circuit(params, description)
        
    except (ValueError, KeyboardInterrupt):
        print("\n❌ Invalid input or interrupted")

if __name__ == "__main__":
    print("🚀 QUANTUM ERROR MITIGATION PREDICTOR")
    print("=" * 45)
    
    # Run demonstration
    # demo_predictions()

    # Run interactive mode
    interactive_prediction()
    
    print(f"\n✅ PREDICTION DEMO COMPLETE!")
    print("💡 Modify the test_circuits list to try different parameters")
    print("🔧 Use interactive_prediction() for manual input")
