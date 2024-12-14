import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models.quantum_model import QuantumModel
from models.classical_model import ClassicalModel

# Load dataset
data = pd.read_csv('data/dataset.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Classical Model
print("Training Classical Model...")
classical_model = ClassicalModel()
classical_model.train(X_train, y_train)
classical_accuracy = classical_model.evaluate(X_test, y_test)
print(f"Classical Model Accuracy: {classical_accuracy}")

# Train Quantum Model
print("Training Quantum Model...")
quantum_model = QuantumModel()
quantum_model.train(X_train, y_train)
quantum_predictions = quantum_model.predict(X_test)
quantum_accuracy = sum(quantum_predictions == y_test) / len(y_test)
print(f"Quantum Model Accuracy: {quantum_accuracy}")

# Save results
with open('results/evaluation_metrics.txt', 'w') as f:
    f.write(f"Classical Model Accuracy: {classical_accuracy}\n")
    f.write(f"Quantum Model Accuracy: {quantum_accuracy}\n")
