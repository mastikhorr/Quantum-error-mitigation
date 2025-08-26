import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

import warnings
import joblib
import os
warnings.filterwarnings('ignore')

class QuantumErrorMitigationRecommender:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = ""
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.df = None
        self.results_df = None
        self.feature_columns = []
        
    def generate_synthetic_data(self, n_samples=1000):
        """Generate synthetic quantum circuit data for demonstration"""
        print("🔬 Generating synthetic quantum circuit data...")
        
        np.random.seed(42)
        
        # Circuit parameters
        circuit_depth = np.random.randint(5, 101, n_samples)
        num_qubits = np.random.randint(2, 11, n_samples)
        num_gates = circuit_depth * num_qubits * np.random.uniform(0.5, 2.0, n_samples)
        num_gates = num_gates.astype(int)
        
        # Noise and complexity parameters
        noise_level = np.random.uniform(0.01, 0.3, n_samples)
        entanglement = np.random.uniform(0.0, 1.0, n_samples)
        gate_density = num_gates / (circuit_depth * num_qubits)
        two_qubit_ratio = np.random.uniform(0.1, 0.7, n_samples)
        
        # Circuit types
        circuit_types = ['optimization', 'simulation', 'variational', 'benchmark']
        circuit_type = np.random.choice(circuit_types, n_samples)
        
        # Create feature for circuit complexity
        complexity_score = (circuit_depth * 0.3 + 
                          num_qubits * 0.2 + 
                          entanglement * 0.3 + 
                          noise_level * 0.2)
        
        # Generate target strategies based on heuristics
        strategies = []
        for i in range(n_samples):
            if noise_level[i] > 0.2 and complexity_score[i] > 0.7:
                strategy = 'zne'  # Zero Noise Extrapolation for high noise
            elif noise_level[i] > 0.15 and entanglement[i] > 0.6:
                strategy = 'cdr'  # Clifford Data Regression for entangled circuits
            elif noise_level[i] > 0.1:
                strategy = 'mem'  # Measurement Error Mitigation for moderate noise
            else:
                strategy = 'none'  # No mitigation for low noise
            strategies.append(strategy)
        
        # Create DataFrame
        data = {
            'circuit_depth': circuit_depth,
            'num_gates': num_gates,
            'num_qubits': num_qubits,
            'entanglement': entanglement,
            'noise_level': noise_level,
            'gate_density': gate_density,
            'two_qubit_ratio': two_qubit_ratio,
            'circuit_type': circuit_type,
            'complexity_score': complexity_score,
            'strategy': strategies
        }
        
        self.df = pd.DataFrame(data)
        print(f"✅ Generated {len(self.df)} synthetic samples")
        print(f"📈 Dataset shape: {self.df.shape}")
        
        # Display target distribution
        print(f"🎯 Target distribution:")
        print(self.df['strategy'].value_counts())
        
        return self.df
    
    def load_data(self, filepath=None):
        """Load dataset from file or generate synthetic data"""
        if filepath and os.path.exists(filepath):
            print(f"📊 Loading dataset from {filepath}...")
            self.df = pd.read_csv(filepath)
            print(f"✅ Dataset loaded: {len(self.df)} samples")
        else:
            print("📊 Dataset file not found. Generating synthetic data...")
            self.df = self.generate_synthetic_data()
        
        print(f"📈 Dataset shape: {self.df.shape}")
        print(f"🎯 Target distribution:")
        print(self.df['strategy'].value_counts())
        
        return self.df
    
    def preprocess_data(self):
        """Preprocess the dataset for training"""
        print("\n🔧 Preprocessing data...")
        
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Separate features and target
        target_col = 'strategy'
        feature_cols = [col for col in self.df.columns if col != target_col]
        
        X = self.df[feature_cols].copy()
        y = self.df[target_col].copy()
        
        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_columns:
            if col in X.columns:
                # Use label encoding for categorical variables
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Handle missing values
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].mean())
        
        # Fill any remaining non-numeric columns with mode or default value
        for col in X.columns:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 0)
        
        # Encode target variable
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        self.feature_columns = X.columns.tolist()
        
        print(f"✅ Preprocessing complete")
        print(f"📊 Features: {len(self.feature_columns)}")
        print(f"🎯 Classes: {list(self.label_encoder.classes_)}")
        
        return X_scaled, y_encoded
    
    def create_models(self):
        """Initialize ML models"""
        print("\n🤖 Creating ML models...")
        
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                n_jobs=-1,
                max_depth=10
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, 
                random_state=42,
                max_depth=6
            ),
            'Support Vector Machine': SVC(
                kernel='rbf', 
                random_state=42, 
                probability=True,
                C=1.0
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                C=1.0
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50), 
                random_state=42, 
                max_iter=500,
                alpha=0.01
            )
        }
        
        print(f"✅ Created {len(self.models)} models")
        return self.models
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train and evaluate all models"""
        print("\n🏋️ Training models...")
        
        results = []
        
        for name, model in self.models.items():
            print(f"\n📚 Training {name}...")
            
            try:
                # Train the model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                print(f"   ✅ Accuracy: {accuracy:.4f}")
                print(f"   📊 CV Score: {cv_mean:.4f} (±{cv_std:.4f})")
                
                results.append({
                    'Model': name,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1,
                    'CV Mean': cv_mean,
                    'CV Std': cv_std
                })
                
            except Exception as e:
                print(f"   ❌ Error training {name}: {str(e)}")
                results.append({
                    'Model': name,
                    'Accuracy': 0.0,
                    'Precision': 0.0,
                    'Recall': 0.0,
                    'F1-Score': 0.0,
                    'CV Mean': 0.0,
                    'CV Std': 0.0
                })
        
        # Convert to DataFrame and find best model
        self.results_df = pd.DataFrame(results)
        best_idx = self.results_df['Accuracy'].idxmax()
        self.best_model_name = self.results_df.loc[best_idx, 'Model']
        self.best_model = self.models[self.best_model_name]
        
        print(f"\n📈 Model Evaluation Results:")
        print("=" * 80)
        print(self.results_df.round(4).to_string(index=False))
        
        print(f"\n🏆 Best Model: {self.best_model_name}")
        print(f"   Accuracy: {self.results_df.loc[best_idx, 'Accuracy']:.4f}")
        
        # Detailed classification report for best model
        if self.best_model is not None:
            y_pred_best = self.best_model.predict(X_test)
            print(f"\n📊 Detailed Classification Report for {self.best_model_name}:")
            print("-" * 60)
            target_names = [f"{cls}" for cls in self.label_encoder.classes_]
            print(classification_report(y_test, y_pred_best, target_names=target_names, zero_division=0))
        
        return self.results_df
    
    def recommend_strategy(self, circuit_params):
        """Recommend best error mitigation strategy for given circuit parameters"""
        print("\n🎯 QUANTUM ERROR MITIGATION STRATEGY RECOMMENDATION")
        print("=" * 60)
        
        if self.best_model is None:
            print("❌ No trained model available. Train models first.")
            return None, None
        
        # Prepare input features
        if len(circuit_params) != len(self.feature_columns):
            print(f"❌ Expected {len(self.feature_columns)} features, got {len(circuit_params)}")
            print(f"Expected features: {self.feature_columns}")
            return None, None
        
        # Create feature array
        features = np.array(circuit_params).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        try:
            # Get prediction and probabilities
            prediction = self.best_model.predict(features_scaled)[0]
            probabilities = self.best_model.predict_proba(features_scaled)[0]
            
            # Convert prediction to strategy name
            recommended_strategy = self.label_encoder.inverse_transform([prediction])[0]
            
            print(f"📋 Circuit Parameters:")
            for i, param_name in enumerate(self.feature_columns):
                print(f"   • {param_name}: {circuit_params[i]}")
            
            print(f"\n🚀 Recommended Strategy: {recommended_strategy.upper()}")
            print(f"🎯 Confidence: {probabilities[prediction]:.2%}")
            
            print(f"\n📊 All Strategy Probabilities:")
            for i, strategy in enumerate(self.label_encoder.classes_):
                print(f"   • {strategy.upper()}: {probabilities[i]:.2%}")
            
            # Strategy explanations
            strategy_info = {
                'zne': '🔧 Zero Noise Extrapolation - Reduces noise by extrapolating to zero noise limit',
                'mem': '🧠 Measurement Error Mitigation - Corrects readout errors in measurements', 
                'cdr': '🎛️ Clifford Data Regression - Uses Clifford circuits for noise characterization',
                'none': '⚡ No Mitigation - Run circuit without error mitigation'
            }
            
            if recommended_strategy.lower() in strategy_info:
                print(f"\n💡 Strategy Info: {strategy_info[recommended_strategy.lower()]}")
            
            return recommended_strategy, probabilities
            
        except Exception as e:
            print(f"❌ Error making prediction: {str(e)}")
            return None, None
    
    def batch_recommendations(self, test_circuits):
        """Make recommendations for multiple circuits"""
        print(f"\n🔄 BATCH STRATEGY RECOMMENDATIONS ({len(test_circuits)} circuits)")
        print("=" * 70)
        
        recommendations = []
        for i, params in enumerate(test_circuits):
            print(f"\n🔸 Circuit {i+1}:")
            strategy, probs = self.recommend_strategy(params)
            if strategy is not None and probs is not None:
                confidence = probs[self.label_encoder.transform([strategy])[0]]
                recommendations.append({
                    'circuit_id': i+1,
                    'recommended_strategy': strategy,
                    'confidence': confidence
                })
            print("-" * 40)
        
        return recommendations
    
    def save_model(self):
        """Save the trained best model and preprocessing objects"""
        if self.best_model is None:
            print("❌ No trained model to save. Train first.")
            return  # Exit early if no model

        try:
            joblib.dump(self.best_model, "best_quantum_error_model.pkl")
            joblib.dump(self.scaler, "feature_scaler.pkl")
            joblib.dump(self.label_encoder, "label_encoder.pkl")

            with open("feature_names.txt", "w") as f:
                for feature in self.feature_columns:
                    f.write(f"{feature}\n")

            print("✅ Model and preprocessing files saved successfully!")
            print("📁 Saved files:")
            print("   • best_quantum_error_model.pkl")
            print("   • feature_scaler.pkl")
            print("   • label_encoder.pkl")
            print("   • feature_names.txt")
            
        except Exception as e:
            print(f"❌ Error saving model: {str(e)}")
    
    def load_saved_model(self):
        """Load a previously saved model and preprocessing objects"""
        try:
            self.best_model = joblib.load("best_quantum_error_model.pkl")
            self.scaler = joblib.load("feature_scaler.pkl")
            self.label_encoder = joblib.load("label_encoder.pkl")
            
            with open("feature_names.txt", "r") as f:
                self.feature_columns = [line.strip() for line in f.readlines()]
            
            print("✅ Model and preprocessing objects loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading saved model: {str(e)}")
            return False
    



def main():
    print("=" * 60)
    print("🚀 QUANTUM ML ERROR MITIGATION - COMPLETE ANALYSIS")
    print("=" * 60)
    
    # Initialize recommender
    recommender = QuantumErrorMitigationRecommender()
    
    # Load data (will generate synthetic data if file not found)
    df = recommender.load_data('mitigation_results.csv')
    
    # Preprocess data
    try:
        X, y = recommender.preprocess_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"\n📊 Training set: {len(X_train)} samples")
        print(f"📊 Test set: {len(X_test)} samples")
        
        # Create and train models
        recommender.create_models()
        results = recommender.train_models(X_train, y_train, X_test, y_test)

        # Save the trained best model and preprocessing objects
        recommender.save_model()
        
        # Demo recommendations
        print("\n" + "="*60)
        print("🎯 STRATEGY RECOMMENDATION EXAMPLES")
        print("="*60)
        
        # Example circuit parameters based on the synthetic data structure
        # [circuit_depth, num_gates, num_qubits, entanglement, noise_level, gate_density, two_qubit_ratio, circuit_type_encoded, complexity_score]
        test_circuits = [
            [10, 25, 4, 0.3, 0.05, 0.625, 0.2, 0, 0.25],  # Low noise, moderate complexity
            [50, 150, 6, 0.8, 0.15, 0.75, 0.4, 1, 0.75],  # High noise, high complexity  
            [5, 12, 3, 0.1, 0.02, 0.4, 0.1, 2, 0.15],     # Very low noise, simple circuit
            [100, 300, 8, 0.9, 0.25, 0.9, 0.6, 3, 0.95]   # Very high noise, complex circuit
        ]
        
        # Make recommendations
        recommendations = recommender.batch_recommendations(test_circuits)
        
        print(f"\n✅ ANALYSIS COMPLETE!")
        print(f"🏆 Best performing model: {recommender.best_model_name}")
        print(f"📊 Total recommendations made: {len(recommendations)}")
        print(f"📈 Model ready for deployment!")
        
    except Exception as e:
        print(f"❌ Error in main execution: {str(e)}")
        print("Please check your data and try again.")


if __name__ == "__main__":
    main()