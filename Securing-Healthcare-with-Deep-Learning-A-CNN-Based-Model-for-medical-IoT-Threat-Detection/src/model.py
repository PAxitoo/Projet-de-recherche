import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Flatten, Dense, Dropout, BatchNormalization

def create_cnn_lstm_model(input_shape, num_classes):
    """Crée un modèle hybride CNN + LSTM."""
    model = Sequential([
        # Couche CNN pour extraire les caractéristiques spatiales
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        # Deuxième couche CNN
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        # Couche LSTM pour capturer les relations temporelles
        LSTM(64, return_sequences=True),
        LSTM(32),

        # Couche Fully Connected
        Dense(64, activation='relu'),
        Dropout(0.5),

        # Couche de sortie
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    """Entraîne le modèle CNN + LSTM et retourne l'historique."""
    history = model.fit(X_train, y_train, 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        validation_data=(X_val, y_val))
    
    return model, history
