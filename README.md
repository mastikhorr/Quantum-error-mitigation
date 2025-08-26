
# Quantum Error Mitigation Recommender

This project presents a **machine learning-based framework** for recommending optimal **quantum error mitigation strategies** for quantum circuits. It provides a complete, reproducible pipeline covering **data generation, model training, and prediction**. The work was developed during my internship at **IBM**, under the guidance of **Dr. Anupama Ray**.

---

## 📂 Project Structure

```
QML_ERROR_MITIGATION_RECOMMENDER/
│── data/                         # Dataset files
│── mitigation_strategies/        # (Optional) strategy definitions/scripts
│── models/                       # (Optional) saved models directory
│── venv/                         # Virtual environment (ignored in production)
│── __pycache__/                  # Cache files
│
│── best_quantum_error_model.pkl  # Saved trained model
│── feature_names.txt             # Feature column names
│── feature_scaler.pkl            # Scaler object
│── label_encoder.pkl             # Label encoder for target classes
│
│── quantum_dataset_generator.py  # Script for dataset creation
│── quantum_ml_trainer.py         # Training pipeline (model building, evaluation, saving)
│── predict.py                    # Model loading and prediction (demo + interactive mode)
│── requirements.txt              # Project dependencies
│── README.md                     # Project documentation
```

---

## 🔑 Key Features

- **Synthetic Dataset Generation**  
  - Automatically generates quantum circuit datasets with **9 key parameters**.  
- **Model Training & Evaluation**  
  - Trains multiple ML models: **Random Forest, Gradient Boosting, SVM, Logistic Regression, Neural Network**.  
  - Selects the **best-performing model automatically**.  
- **Error Mitigation Recommendation**  
  - Predicts strategies:  
    - **ZNE** – Zero Noise Extrapolation  
    - **MEM** – Measurement Error Mitigation  
    - **CDR** – Clifford Data Regression  
    - **None** – No mitigation required  
- **Interactive & Batch Predictions**  
  - Allows manual entry of circuit parameters.  
  - Provides preconfigured demonstrations with batch predictions.  
- **Persistence**  
  - Saves model, scaler, and encoders for reproducibility.  

---

## 📊 Features Used for Prediction

Each circuit is represented using **9 parameters**:

1. `circuit_depth`  
2. `num_gates`  
3. `num_qubits`  
4. `entanglement`  
5. `noise_level`  
6. `gate_density`  
7. `two_qubit_ratio`  
8. `circuit_type` (encoded: 0=optimization, 1=simulation, 2=variational, 3=benchmark)  
9. `complexity_score` (calculated heuristic)  

---

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/mastikhorr/Quantum-error-mitigation.git
   cd Quantum-error-mitigation
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📦 Requirements

`requirements.txt`
```
qiskit==1.0.2
qiskit-aer==0.14.1
numpy
pandas
matplotlib
scikit-learn
tqdm
```

---

## 🏋️ Training the Model

Run the training script to build and evaluate models:

```bash
python quantum_ml_trainer.py
```

This will:
- Load or generate the dataset  
- Train multiple models  
- Evaluate and compare them  
- Save the best model along with preprocessing objects  

**Output files created:**
- `best_quantum_error_model.pkl`  
- `feature_scaler.pkl`  
- `label_encoder.pkl`  
- `feature_names.txt`  

---

## 🎯 Making Predictions

### 1. Interactive Mode
Run:
```bash
python predict.py
```
You will be prompted to input **7 circuit parameters** (remaining 2 are auto-completed).

### 2. Batch Demonstration
Enable `demo_predictions()` inside `predict.py` and run:
```bash
python predict.py
```

### 3. Evaluate on External Data
```python
predictor = QuantumErrorMitigationPredictor()
predictor.load_trained_model()
predictor.evaluate_on_test_data("data/test_dataset.csv")
```

---

## 🔗 Repository Link

[![GitHub Repo](https://img.shields.io/badge/View%20on-GitHub-blue?logo=github)](https://github.com/mastikhorr/Explore-AI-methods-for-selecting-Error-Mitigation-Strategies-Objective)

---

## 👨‍💻 Author
**Sidharth Kumar**  
Developed during internship at **IBM**, under guidance of **Dr. Anupama Ray**.

