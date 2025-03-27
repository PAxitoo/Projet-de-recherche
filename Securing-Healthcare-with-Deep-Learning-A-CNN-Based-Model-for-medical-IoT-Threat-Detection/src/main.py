import os
import tensorflow as tf
import matplotlib.pyplot as plt
from data_loader import load_and_preprocess_data
from model import create_cnn_lstm_model, train_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def plot_training_history(history):
    """Affiche les courbes de perte et d'accuracy pendant l'entraînement."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Graphique de la perte (Loss)
    axs[0].plot(history.history['loss'], label='Train Loss')
    axs[0].plot(history.history['val_loss'], label='Validation Loss')
    axs[0].set_title('Loss per Epoch')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Graphique de l'accuracy
    axs[1].plot(history.history['accuracy'], label='Train Accuracy')
    axs[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axs[1].set_title('Accuracy per Epoch')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    plt.show()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    data_dir = os.path.join(script_dir, '..', 'data')  

    # Charger les données
    X_train, X_test, y_train, y_test, label_encoder = load_and_preprocess_data(data_dir)

    # Définir la forme de l'entrée du modèle hybride CNN + LSTM
    input_shape = (X_train.shape[1], 1)  # LSTM nécessite une structure séquentielle avec (timesteps, features)

    # Création du modèle CNN + LSTM
    model = create_cnn_lstm_model(input_shape, y_train.shape[1])

    # Vérifier si un GPU est disponible
    if tf.test.gpu_device_name():
        print('GPU is available!')
    else:
        print('GPU is not available. Using CPU.')

    # Entraînement du modèle
    model, history = train_model(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=32)

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

    print("\nMetrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    print("\nClassification Report:\n", classification_report(y_test_decoded, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test_decoded, y_pred))

    # Affichage des courbes d'entraînement
    plot_training_history(history)
