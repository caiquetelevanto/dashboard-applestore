
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import skfuzzy as fuzz

# ==========================================
# 1. CONFIGURAÇÃO DA PÁGINA
# ==========================================

st.title("🧠 Segmentação Fuzzy — Google Play Store")

st.markdown("""
Análise de segmentação dos aplicativos utilizando o algoritmo
**Fuzzy C-Means**, considerando avaliações, reviews,
instalações e preços.
""")

# ==========================================
# 2. CARREGAMENTO DA BASE
# ==========================================

@st.cache_data
def carregar_dados():

    df = pd.read_csv('google_play_store_dataset.csv')

    # Remover registros sem informações essenciais
    df = df.dropna(
        subset=['Rating', 'Reviews', 'Installs', 'Price']
    ).copy()

    # Converter Rating
    df['Rating'] = pd.to_numeric(
        df['Rating'], errors='coerce'
    )

    # Limpar Installs
    df['Installs'] = (
        df['Installs']
        .astype(str)
        .str.replace('+', '', regex=False)
        .str.replace(',', '', regex=False)
    )

    df['Installs'] = pd.to_numeric(
        df['Installs'], errors='coerce'
    )

    # Limpar Price
    df['Price'] = (
        df['Price']
        .astype(str)
        .str.replace('$', '', regex=False)
    )

    df['Price'] = pd.to_numeric(
        df['Price'], errors='coerce'
    )

    # Converter Reviews
    df['Reviews'] = pd.to_numeric(
        df['Reviews'], errors='coerce'
    )

    # Remover valores inválidos
    df = df.dropna(
        subset=['Rating', 'Reviews', 'Installs', 'Price']
    )

    # Instalações precisam ser positivas para a escala log
    df = df[df['Installs'] > 0].copy()

    return df


df = carregar_dados()

# ==========================================
# 3. SELEÇÃO DAS VARIÁVEIS
# ==========================================

features = ['Rating', 'Reviews', 'Installs', 'Price']

data_cluster = df[features].values

# ==========================================
# 4. NORMALIZAÇÃO
# ==========================================

scaler = MinMaxScaler()

data_scaled = scaler.fit_transform(data_cluster)

# Transposição exigida pelo Scikit-Fuzzy
data_fuzzy = data_scaled.T

# ==========================================
# 5. FUZZY C-MEANS
# ==========================================

@st.cache_data
def executar_fuzzy(data_fuzzy):

    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data_fuzzy,
        c=3,
        m=2,
        error=0.005,
        maxiter=1000,
        init=None
    )

    return u, fpc, p


u, fpc, iteracoes = executar_fuzzy(data_fuzzy)

# ==========================================
# 6. GRAUS DE PERTINÊNCIA
# ==========================================

for i in range(3):

    df[f'Pertinencia_Grupo_{i+1}'] = u[i]

df['Grupo_Dominante'] = np.argmax(u, axis=0) + 1

# ==========================================
# 7. INDICADORES
# ==========================================

st.subheader("Indicadores da Segmentação")

col1, col2, col3 = st.columns(3)

col1.metric("Aplicativos analisados", f"{len(df):,}".replace(",", "."))
col2.metric("Clusters identificados", "3")
col3.metric("FPC", f"{fpc:.4f}")

st.caption(
    f"Algoritmo: Fuzzy C-Means | "
    f"Fator de fuzzificação: 2 | "
    f"Iterações: {iteracoes}"
)

st.divider()

# ==========================================
# 8. GRÁFICO DE DISPERSÃO
# ==========================================

st.subheader("Qualidade vs. Popularidade dos Aplicativos")

x_data = df['Rating']
y_data = df['Installs']

cores_fuzzy = df['Pertinencia_Grupo_1']

fig = go.Figure(data=[go.Scatter(
    x=x_data,
    y=y_data,
    mode='markers',
    marker=dict(
        size=6,
        color=cores_fuzzy,
        colorscale='Viridis',
        opacity=0.6,
        showscale=True,
        colorbar=dict(
            title=dict(
                text="Pertinência G1",
                font=dict(color='white')
            ),
            tickfont=dict(color='white')
        )
    )
)])

# ==========================================
# 9. LAYOUT DO GRÁFICO
# ==========================================

fig.update_layout(

    template="plotly_dark",

    paper_bgcolor='#111111',
    plot_bgcolor='#111111',

    xaxis=dict(
        title=dict(
            text='Avaliações (Rating)',
            font=dict(color='white')
        ),
        gridcolor='#222222',
        tickfont=dict(color='white'),
        range=[1, 5]
    ),

    yaxis=dict(
        title=dict(
            text='Volume de Instalações (Escala Log)',
            font=dict(color='white')
        ),
        type='log',
        gridcolor='#222222',
        tickfont=dict(color='white')
    ),

    margin=dict(
        l=40,
        r=40,
        b=40,
        t=50
    ),

    title=dict(
        text="Transição Fuzzy: Qualidade vs Popularidade dos Apps",
        font=dict(color='white', size=16)
    ),

    height=600
)

# ==========================================
# 10. EXIBIÇÃO NO STREAMLIT
# ==========================================

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================
# 11. EXIBIÇÃO DOS GRUPOS
# ==========================================

st.divider()

st.subheader("Distribuição dos Grupos")

perfil_grupos = df.groupby('Grupo_Dominante').agg({
    'Rating': 'mean',
    'Reviews': 'mean',
    'Installs': 'mean',
    'Price': 'mean'
}).round(2)

perfil_grupos.index = [
    f"Grupo {i}" for i in perfil_grupos.index
]

st.dataframe(
    perfil_grupos,
    use_container_width=True
)
