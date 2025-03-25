import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from models.tumor_classifier import TumorClassifier
import os

def load_data():
    """Load and prepare the breast cancer data from scikit-learn."""
    # Load the breast cancer dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    return X, y, data.target_names

def train_model(model, X_train, y_train, X_test, y_test, epochs=50, learning_rate=0.001):
    """Train the tumor classifier model."""
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train).squeeze()
        loss = criterion(outputs, y_train)

        # Backward pass
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    # Evaluate model
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).squeeze()
        predictions = (predictions > 0.5).float()
        accuracy = (predictions == y_test).sum().item() / y_test.size(0)
        print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
        
        # Convert predictions and y_test to numpy for sklearn metrics
        y_pred = predictions.numpy()
        y_true = y_test.numpy()
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['Malignant', 'Benign']))
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        
        # Calculate detailed statistics
        total_samples = len(y_true)
        actual_malignant = np.sum(y_true == 0)
        actual_benign = np.sum(y_true == 1)
        predicted_malignant = np.sum(y_pred == 0)
        predicted_benign = np.sum(y_pred == 1)
        
        # Print detailed summary table
        print("\nDetailed Tumor Classification Summary")
        print("=====================================")
        print(f"{'Category':<15} {'Predicted':<15} {'Actual':<15} {'Percentage':<15}")
        print("-" * 60)
        print(f"{'Malignant':<15} {predicted_malignant:<15} {actual_malignant:<15} {actual_malignant/total_samples*100:.1f}%")
        print(f"{'Benign':<15} {predicted_benign:<15} {actual_benign:<15} {actual_benign/total_samples*100:.1f}%")
        print("-" * 60)
        print(f"{'Total':<15} {total_samples:<15} {total_samples:<15} {'100%':<15}")
        
        print("\nCorrect Predictions:")
        print(f"Malignant correctly identified: {cm[0,0]} ({cm[0,0]/actual_malignant*100:.1f}%)")
        print(f"Benign correctly identified: {cm[1,1]} ({cm[1,1]/actual_benign*100:.1f}%)")
        
        print("\nFalse Predictions:")
        print(f"False Positives: {cm[1,0]} ({cm[1,0]/actual_benign*100:.1f}% of benign cases)")
        print(f"False Negatives: {cm[0,1]} ({cm[0,1]/actual_malignant*100:.1f}% of malignant cases)")

    return model

def save_model(model, scaler):
    """Save the trained model and scaler."""
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model state
    torch.save(model.state_dict(), 'models/tumor_classifier.pth')
    
    # Save scaler
    import joblib
    joblib.dump(scaler, 'models/scaler.pkl')
    print("\nModel and scaler saved to 'models' directory")

def main():
    # Load and prepare data
    X, y, target_names = load_data()
    
    # Scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Initialize and train model
    model = TumorClassifier()
    trained_model = train_model(model, X_train, y_train, X_test, y_test)
    
    # Save the trained model and scaler
    save_model(trained_model, scaler)

if __name__ == "__main__":
    main() 