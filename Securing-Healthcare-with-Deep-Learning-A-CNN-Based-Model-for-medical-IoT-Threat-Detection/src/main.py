import os
import argparse
from data_loader import load_and_preprocess_data
from model import create_cnn_model, train_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    data_dir = os.path.join(script_dir, '..', 'data')  
    
    # Charger les données
    X_train, X_test, y_train, y_test, label_encoder = load_and_preprocess_data(data_dir)
    
    # Définir la forme de l'entrée du modèle
    input_shape = (X_train.shape[1], 1) 
    model = create_cnn_model(input_shape, y_train.shape[1])
    
    # Vérifier si un GPU est disponible
    import tensorflow as tf 
    if tf.test.gpu_device_name():
        print('GPU is available!')
    else:
        print('GPU is not available. Using CPU.')
    
    # Entraînement du modèle
    model = train_model(model, X_train, y_train, X_test, y_test)
    
    # Évaluation du modèle
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Prédiction et évaluation des performances
    y_pred_categorical = model.predict(X_test)
    y_pred_encoded = y_pred_categorical.argmax(axis=1)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    
    y_test_decoded = label_encoder.inverse_transform(y_test.argmax(axis=1))
    
    accuracy = accuracy_score(y_test_decoded, y_pred)
    precision = precision_score(y_test_decoded, y_pred, average='weighted')
    recall = recall_score(y_test_decoded, y_pred, average='weighted')
    f1 = f1_score(y_test_decoded, y_pred, average='weighted')
    
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
    print("\nClassification Report:\n", classification_report(y_test_decoded, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test_decoded, y_pred))