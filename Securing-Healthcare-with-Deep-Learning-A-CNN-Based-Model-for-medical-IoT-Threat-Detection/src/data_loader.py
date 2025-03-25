import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(data_dir, test_size=0.2):
    """Load, preprocess, and prepare data for training."""
    file_path = os.path.join(data_dir, 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} n'existe pas. Vérifiez le chemin.")
    
    # Chargement du dataset
    df = pd.read_csv(file_path)
    
    # Nettoyage des noms de colonnes (suppression des espaces)
    df.columns = df.columns.str.strip()
    
    # Remplacement des valeurs infinies et NaN
    df.replace([float('inf'), float('-inf')], float('nan'), inplace=True)  # Convertit les infs en NaN
    df.dropna(inplace=True)  # Supprime les lignes contenant des NaN
    
    # Vérification si 'Label' est bien présent
    if 'Label' not in df.columns:
        print("Colonnes disponibles :", df.columns)
        raise ValueError("Le dataset ne contient pas de colonne 'Label' après nettoyage. Vérifiez le format du fichier CSV.")
    
    # Séparation des caractéristiques et des labels
    X = df.drop(columns=['Label'])
    y = df['Label']
    
    # Encodage des labels (0 pour Benign, 1 pour DDoS)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    
    # Séparation en jeu d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=test_size, random_state=42)
    
    # Normalisation des données
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Reshape pour CNN (Conv1D nécessite un format spécifique)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    return X_train, X_test, y_train, y_test, label_encoder