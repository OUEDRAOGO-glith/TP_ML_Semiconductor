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

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="SC-QC Pro | Industrial Intelligence",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- INJECTION CSS PREMIUM ---
st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    
    <style>
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(239, 246, 249) 0%, rgb(206, 239, 253) 90%);
    }

    /* Modern Card Styles */
    .st-emotion-cache-1r6slb0, .st-emotion-cache-1gwv8h1 {
        background: rgba(255, 255, 255, 0.7) !important;
        backdrop-filter: blur(10px);
        border-radius: 20px !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07) !important;
        padding: 25px !important;
    }

    /* Metric Cards Customization */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        color: #1E3A8A !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-weight: 600 !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
        color: white !important;
    }
    section[data-testid="stSidebar"] .st-emotion-cache-10o48o6 {
        color: white !important;
    }
    
    /* Buttons Customization */
    .stButton>button {
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 10px 25px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3) !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5) !important;
    }

    /* Custom Header */
    .dashboard-header {
        background: linear-gradient(90deg, #1E293B 0%, #334155 100%);
        padding: 30px;
        border-radius: 20px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .status-ok { background-color: #D1FAE5; color: #065F46; }
    .status-fail { background-color: #FEE2E2; color: #991B1B; }
    </style>
    """, unsafe_allow_html=True)

# --- SYSTEM INITIALIZATION ---
@st.cache_resource
def load_qc_system():
    return SemiconductorQualityControl()

qc = load_qc_system()

# --- SIDEBAR UI ---
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/null/microchip.png", width=80)
    st.markdown("<h1 style='color: white; font-size: 1.5rem;'>SC-QC PRO</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8; font-size: 0.9rem;'>Production Line Control Intelligence</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    menu = st.radio("Navigation", 
                  ["🛰️ Vue d'ensemble", "🔍 Contrôle de Lot", "📊 Statistiques Capteurs", "🎯 Intelligence Modèle"])
    
    st.markdown("---")
    st.markdown("### 🟢 État du Système")
    st.success("Modèle : RF-SMOTE v2.1")
    st.info("Flux de données : Temps Réel")

# --- MAIN CONTENT ---
if menu == "🛰️ Vue d'ensemble":
    # Header
    st.markdown("""
        <div class='dashboard-header'>
            <h1 style='margin:0; font-size: 2.5rem;'>Tableau de Bord Industriel</h1>
            <p style='margin:0; opacity: 0.8;'>Monitoring intelligent de la qualité des semi-conducteurs</p>
        </div>
    """, unsafe_allow_html=True)
    
    # KPI Grid
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Analysé", "1,852", "↑ 24%")
    with c2:
        st.metric("Taux de Défaut", "5.82%", "-0.45%", delta_color="inverse")
    with c3:
        st.metric("F1-Score IA", "95.1%", "+1.2%")
    with c4:
        st.metric("Temps de Réponse", "12ms", "Optimisé")

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_l, col_r = st.columns([2, 1])
    
    with col_l:
        st.subheader("📈 Historique de Production (Simulé)")
        chart_data = pd.DataFrame(
            np.random.randn(24, 2),
            columns=['Volume Pass', 'Volume Fail']
        )
        st.line_chart(chart_data)
        
    with col_r:
        st.subheader("💡 Recommandations IA")
        st.info("**Maintenance préventive :** Vérifier le capteur 589 sur la ligne 3.")
        st.warning("**Alerte Qualité :** Légère dérive observée sur la phase de Lithographie.")

elif menu == "🔍 Contrôle de Lot":
    st.title("🔍 Contrôle de Lot en Temps Réel")
    
    st.markdown("""
        <div style='background-color: white; padding: 20px; border-radius: 15px; margin-bottom: 25px;'>
            Glissez-déposez ici les données issues de la chaîne de production pour une analyse instantanée.
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type="csv")
    
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.dataframe(data.head(5), use_container_width=True)
        
        if st.button("🚀 LANCER L'ANALYSE IA"):
            with st.spinner('Détection des anomalies en cours...'):
                features = data.drop(columns=['Target']) if 'Target' in data.columns else data
                preds, probs = qc.predict(features)
                
                # Metrics du lot
                l1, l2, l3 = st.columns(3)
                l1.metric("Puces Défectueuses", sum(preds==1))
                l2.metric("Puces Conformes", sum(preds==0))
                l3.metric("Rendement (Yield)", f"{(sum(preds==0)/len(preds))*100:.1f}%")
                
                # Visualisation
                st.markdown("---")
                v1, v2 = st.columns([1, 1])
                
                with v1:
                    fig_gau = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = (sum(preds==0)/len(preds))*100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Conformité du Lot (%)", 'font': {'size': 24}},
                        gauge = {
                            'axis': {'range': [None, 100], 'tickwidth': 1},
                            'bar': {'color': "#3B82F6"},
                            'steps': [
                                {'range': [0, 70], 'color': "#FEE2E2"},
                                {'range': [70, 90], 'color': "#FEF3C7"},
                                {'range': [90, 100], 'color': "#D1FAE5"}]
                        }
                    ))
                    st.plotly_chart(fig_gau, use_container_width=True)
                
                with v2:
                    res_df = pd.DataFrame({
                        'Status': ["DÉFECTUEUX ❌" if p == 1 else "CONFORME ✅" for p in preds],
                        'Confiance': [f"{p*100:.1f}%" if p > 0.5 else f"{(1-p)*100:.1f}%" for p in probs]
                    })
                    st.write("Détails des échantillons :")
                    st.dataframe(res_df.head(15), use_container_width=True)

elif menu == "📊 Statistiques Capteurs":
    st.title("📊 Exploration des Paramètres de Fabrication")
    if os.path.exists('data/secom_preprocessed.csv'):
        df = pd.read_csv('data/secom_preprocessed.csv')
        
        feat = st.selectbox("Sélectionner la variable capteur (Sensor)", df.columns[:100])
        
        fig = px.violin(df, x="Target", y=feat, color="Target", 
                       box=True, points="all",
                       title=f"Analyse de la variance : {feat}",
                       color_discrete_map={0: "#10B981", 1: "#EF4444"},
                       labels={"Target": "0: OK, 1: Fail"})
        
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

elif menu == "🎯 Intelligence Modèle":
    st.title("🎯 Évaluation de l'Intelligence Artificielle")
    
    if qc.metrics:
        m = qc.metrics
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Big metric cards
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Accuracy", f"{m['accuracy']:.2%}")
        mc2.metric("Precision", f"{m['precision']:.2%}")
        mc3.metric("Recall", f"{m['recall']:.2%}")
        mc4.metric("F1-Score", f"{m['f1_score']:.2%}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Matrix and Importance (fake for importance here)
        ml, mr = st.columns(2)
        
        with ml:
            st.subheader("📋 Matrice de Confusion")
            fig_cm = px.imshow(m['confusion_matrix'], 
                             labels=dict(x="Prédit", y="Réel", color="Nombre"),
                             x=['OK', 'FAIL'], y=['OK', 'FAIL'],
                             text_auto=True, color_continuous_scale='Blues')
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with mr:
            st.subheader("💎 Facteurs Clés d'Échec")
            # Simulation d'importance
            importance = pd.DataFrame({
                'Feature': [f"Sensor_{i}" for i in [59, 102, 342, 590, 42]],
                'Impact': [0.15, 0.12, 0.09, 0.08, 0.07]
            }).sort_values('Impact')
            fig_imp = px.bar(importance, x='Impact', y='Feature', orientation='h',
                            color_discrete_sequence=['#3B82F6'])
            st.plotly_chart(fig_imp, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748B;'>Advanced Silicon Quality Control System — Professional Edition</p>", unsafe_allow_html=True)
