import torch
import numpy as np
import joblib
from models.tumor_classifier import TumorClassifier
from sklearn.datasets import load_breast_cancer

def load_model():
    """Load the trained model and scaler."""
    # Load model
    model = TumorClassifier()
    model.load_state_dict(torch.load('models/tumor_classifier.pth'))
    model.eval()
    
    # Load scaler
    scaler = joblib.load('models/scaler.pkl')
    
    return model, scaler

def predict_sample(model, scaler, features):
    """Make a prediction for a single sample."""
    # Scale the features
    features_scaled = scaler.transform([features])
    
    # Convert to tensor and make prediction
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(features_tensor).squeeze()
        probability = prediction.item()
        predicted_class = 0 if probability > 0.5 else 1
    
    return predicted_class, probability

def main():
    # Load the model and scaler
    model, scaler = load_model()
    
    # Load the dataset to get feature names
    data = load_breast_cancer()
    feature_names = data.feature_names
    target_names = data.target_names
    
    print("\nTumor Classification Predictor")
    print("===============================")
    print("Enter the values for each feature (press Enter to quit):")
    
    while True:
        try:
            # Get feature values from user
            features = []
            print("\nEnter values for each feature:")
            for i, name in enumerate(feature_names):
                while True:
                    try:
                        value = input(f"{i+1}. {name}: ")
                        if value.lower() == 'q':
                            return
                        features.append(float(value))
                        break
                    except ValueError:
                        print("Please enter a valid number")
            
            # Make prediction
            predicted_class, probability = predict_sample(model, scaler, features)
            
            # Print prediction results
            print("\nPrediction Results:")
            print("==================")
            print(f"Predicted Class: {target_names[predicted_class]}")
            print(f"Confidence: {abs(probability - 0.5) * 200:.1f}%")
            
            # Print detailed summary
            print("\nDetailed Tumor Classification Summary")
            print("=====================================")
            print(f"{'Category':<15} {'Predicted':<15} {'Actual':<15} {'Percentage':<15}")
            print("-" * 60)
            print(f"{'Malignant':<15} {1 if predicted_class == 0 else 0:<15} {'N/A':<15} {'N/A':<15}")
            print(f"{'Benign':<15} {1 if predicted_class == 1 else 0:<15} {'N/A':<15} {'N/A':<15}")
            print("-" * 60)
            print(f"{'Total':<15} {'1':<15} {'N/A':<15} {'100%':<15}")
            
            print("\nFalse Predictions:")
            print(f"Probability of incorrect prediction: {(1-probability)*100:.1f}%")
            
            print("\nPrediction Details:")
            print(f"Probability of being {target_names[1-predicted_class]}: {(1-probability)*100:.1f}%")
            print(f"Probability of being {target_names[predicted_class]}: {probability*100:.1f}%")
            
            # Ask if user wants to make another prediction
            if input("\nWould you like to make another prediction? (y/n): ").lower() != 'y':
                break
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Please try again or press 'q' to quit")

if __name__ == "__main__":
    main() 