import os
import pickle

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def create_model(data):
    X = data.drop(['diagnosis'], axis=1)  # variable explicatives
    y = data['diagnosis']  # variable cible

    # Mettre les donnees à la meme echelle
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Donnees de teste et donnees d'entrainements
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Créer un modèle de régression logistique
    model = LogisticRegression()

    # Entraîner le modèle sur les données d'entrainement
    model.fit(X_train, y_train)

    # Faire des prédictions sur les données de test
    y_pred = model.predict(X_test)
    print('Accuracy of our model: ', accuracy_score(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))

    # Calculer le score du modèle sur les données de test
    score = model.score(X_test, y_test)
    print('Notre model à un score de : ', score)

    return model, scaler


def data_cleaning():
    # Spécifiez le chemin absolu vers votre fichier data.csv
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_csv_path = os.path.join(project_dir, "..", "data", "data.csv")

    # Lisez le fichier data.csv à l'aide du chemin absolu
    data = pd.read_csv(data_csv_path)
    data = data.drop(['id', 'Unnamed: 32'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data


def main():
    data = data_cleaning()

    model, scaler = create_model(data)

    # Sauvegarder le modèle dans un fichier pickle
    with open('./model/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('./model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)


if __name__ == '__main__':
    main()
