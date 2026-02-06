import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from datetime import datetime

# Import de notre classe de déploiement
from deployment import SemiconductorQualityControl

# Configuration de la page
st.set_page_config(
    page_title="SC-QC | Contrôle Qualité Semi-conducteurs",
    page_icon="🔬",
    layout="wide"
)

# Style CSS pour un look premium
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .status-ok {
        color: #28a745;
        font-weight: bold;
    }
    .status-fail {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialisation du système
@st.cache_resource
def load_qc_system():
    # Le système détecte maintenant automatiquement les chemins 'results/' ou '../results/'
    return SemiconductorQualityControl()

qc = load_qc_system()

# Sidebar
st.sidebar.image("https://img.icons8.com/wired/100/000000/microchip.png", width=100)
st.sidebar.title("SC-QC v1.0")
st.sidebar.markdown("---")
menu = st.sidebar.radio("Navigation", ["Tableau de bord", "Analyse de lot", "Performances Modèle"])

if menu == "Tableau de bord":
    st.title("🔬 Contrôle Qualité des Semi-conducteurs")
    st.markdown(f"**Date :** {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    
    # KPIs fictifs pour la démo
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Puces analysées (24h)", "1,245", "+12%")
    with col2:
        st.metric("Taux de défaut", "6.2%", "-0.5%")
    with col3:
        st.metric("Précision Modèle", "93.2%", "Stable")
    with col4:
        st.metric("Arrêts Ligne evitables", "14", "+2")

    st.markdown("---")
    
    # Section Prédiction Individuelle
    st.subheader("🤖 Prédiction sur un nouvel échantillon")
    uploaded_file = st.file_uploader("Charger un lot de données (CSV)", type="csv")
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Aperçu des données chargées :")
        st.dataframe(data.head())
        
        if st.button("Lancer l'Analyse"):
            with st.spinner('Analyse IA en cours...'):
                # Extraction des features (on suppose que les colonnes sont dans le bon ordre)
                # Si le fichier contient une colonne 'Target', on l'enlève
                features = data.drop(columns=['Target']) if 'Target' in data.columns else data
                
                preds, probs = qc.predict(features)
                
                # Résultats
                st.success("Analyse terminée !")
                
                res_df = pd.DataFrame({
                    'ID Échantillon': range(len(preds)),
                    'Résultat': ["CONFORME ✅" if p == 0 else "DÉFECTUEUX ❌" for p in preds],
                    'Confiance (%)': [f"{p*100:.1f}%" if p > 0.5 else f"{(1-p)*100:.1f}%" for p in probs]
                })
                
                col_left, col_right = st.columns([2, 1])
                
                with col_left:
                    st.dataframe(res_df.style.applymap(
                        lambda x: 'color: green' if 'CONFORME' in str(x) else 'color: red',
                        subset=['Résultat']
                    ))
                
                with col_right:
                    # Graphique de répartition
                    counts = res_df['Résultat'].value_counts()
                    fig = px.pie(values=counts.values, names=counts.index, 
                                title="Répartition du lot",
                                color_discrete_map={"CONFORME ✅": "#28a745", "DÉFECTUEUX ❌": "#dc3545"})
                    st.plotly_chart(fig)

elif menu == "Analyse de lot":
    st.title("📊 Analyse Statistique des Procédés")
    # Chargement des données de référence pour la démo
    if os.path.exists('data/secom_preprocessed.csv'):
        df = pd.read_csv('data/secom_preprocessed.csv')
        
        st.subheader("Distribution des caractéristiques clés")
        feature_to_plot = st.selectbox("Choisir une variable de capteur", df.columns[:50])
        
        fig = px.histogram(df, x=feature_to_plot, color="Target", 
                          marginal="box", title=f"Distribution de {feature_to_plot} par Qualité",
                          color_discrete_map={0: "#28a745", 1: "#dc3545"})
        st.plotly_chart(fig, use_container_width=True)

elif menu == "Performances Modèle":
    st.title("📈 Métriques du Modèle IA")
    if qc.metrics:
        m = qc.metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{m['accuracy']:.2%}")
        c2.metric("Precision", f"{m['precision']:.2%}")
        c3.metric("Recall", f"{m['recall']:.2%}")
        c4.metric("F1-Score", f"{m['f1_score']:.2%}")
        
        # Matrice de confusion avec Plotly
        cm = m['confusion_matrix']
        fig = px.imshow(cm, 
                        labels=dict(x="Prédit", y="Réel", color="Nombre"),
                        x=['OK', 'DÉFAUT'],
                        y=['OK', 'DÉFAUT'],
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale='Viridis')
        st.plotly_chart(fig)
    else:
        st.warning("Métriques non disponibles. Chargez le fichier 'final_metrics.pkl'.")

st.sidebar.markdown("---")
st.sidebar.info("Développé pour l'optimisation des rendements de production.")
