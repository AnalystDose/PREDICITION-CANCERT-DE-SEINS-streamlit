import streamlit as st
import pickle
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go



def data_cleaning():
    # Spécifiez le chemin absolu vers votre fichier data.csv
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_csv_path = os.path.join(project_dir, "..", "data", "data.csv")

    # Lisez le fichier data.csv à l'aide du chemin absolu
    data = pd.read_csv(data_csv_path)
    data = data.drop(['id', 'Unnamed: 32'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data


def add_sidebar():
    st.sidebar.title("Mesures du tissu")

    data = data_cleaning()

    slider_labels = [

        ("Rayon (moyen)", "radius_mean"),
        ("Texture (moyenne)", "texture_mean"),
        ("Périmètre (moyen)", "perimeter_mean"),
        ("Aire (moyenne)", "area_mean"),
        ("Lisse (moyen)", "smoothness_mean"),
        ("Compacité (moyenne)", "compactness_mean"),
        ("Concavité (moyenne)", "concavity_mean"),
        ("Points concaves (moyenne)", "concave points_mean"),
        ("Symétrie (moyenne)", "symmetry_mean"),
        ("Dimension fractale (moyenne)", "fractal_dimension_mean"),
        ("Rayon (écart-type)", "radius_se"),
        ("Texture (écart-type)", "texture_se"),
        ("Périmètre (écart-type)", "perimeter_se"),
        ("Aire (écart-type)", "area_se"),
        ("Lisse (écart-type)", "smoothness_se"),
        ("Compacité (écart-type)", "compactness_se"),
        ("Concavité (écart-type)", "concavity_se"),
        ("Points concaves (écart-type)", "concave points_se"),
        ("Symétrie (écart-type)", "symmetry_se"),
        ("Dimension fractale (écart-type)", "fractal_dimension_se"),
        ("Rayon (pire)", "radius_worst"),
        ("Texture (pire)", "texture_worst"),
        ("Périmètre (pire)", "perimeter_worst"),
        ("Aire (pire)", "area_worst"),
        ("Lisse (pire)", "smoothness_worst"),
        ("Compacité (pire)", "compactness_worst"),
        ("Concavité (pire)", "concavity_worst"),
        ("Points concaves (pire)", "concave points_worst"),
        ("Symétrie (pire)", "symmetry_worst"),
        ("Dimension fractale (pire)", "fractal_dimension_worst"),
    ]

    input_dict = {}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
    return input_dict


def scaled_values(input_dict):
    data = data_cleaning()
    X = data.drop(['diagnosis'], axis=1)

    scaled_dict = {}

    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value

    return scaled_dict


def radar_chart(input_data):

    input_data = scaled_values(input_data)

    # Liste des catégories
    categories = ['Rayon', 'Texture', 'Périmètre', 'Aire',
                  'Lisse', 'Compacité',
                  'Concavité', 'Points concaves',
                  'Symétrie', 'Dimension fractale']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
            input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Valeur moyenne'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
            input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
            input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='écart-type'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
            input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Grande Valeur'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True
    )

    return fig


def prediction(input_data):
    # Chargez le modèle d'apprentissage automatique
    model = pickle.load(open("./model/model.pkl", "rb"))
    scaler = pickle.load(open("./model/scaler.pkl", "rb"))

    input_array = np.array(list(input_data.values())).reshape(1, -1)

    input_array_scaled = scaler.transform(input_array)
    prediction = model.predict(input_array_scaled)

    st.write("La cellule est de type:")

    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malicious'>Malicious</span>", unsafe_allow_html=True)

    st.write("Probabilité d'être Benign: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probabilité d'être malicious: ", model.predict_proba(input_array_scaled)[0][1])

    st.write("Cette application peut aider les professionnels de la santé à établir un diagnostic, mais ne doit pas se substituer à un diagnostic professionnel.")


def main():
    st.set_page_config(
        page_title="Prediction du Cancert des seins",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with open("./assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    input_data = add_sidebar()
    # st.write(input_data)  # Verfier les inputs

    with st.container():
        st.title("Prediction du Cancert des seins")
        st.write(
            "Veuillez connecter cette application à votre laboratoire de cytologie pour l'aider à diagnostiquer le cancer du sein à partir de votre échantillon de tissu. Cette application prédit, à l'aide d'un modèle d'apprentissage automatique, si une masse mammaire est bénigne ou maligne en fonction des mesures qu'elle reçoit de votre laboratoire de cytologie. Vous pouvez également mettre à jour les mesures à la main à l'aide des curseurs situés dans la barre latérale.")

    col1, col2 = st.columns([4, 1])
    with col1:
        st.subheader("Mesures du tissu")
        st.write("Veuillez entrer les mesures du tissu à l'aide des curseurs situés dans la barre latérale.")
        chart = radar_chart(input_data)
        st.plotly_chart(chart)

    with col2:
        st.subheader("Résultats")
        st.write("Les résultats de la prédiction sont affichés ci-dessous.")
        prediction(input_data)


if __name__ == '__main__':
    main()
