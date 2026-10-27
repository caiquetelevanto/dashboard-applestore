import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN

def app():
    st.title("🤖 Clusterização 3D: K-Means + DBSCAN (Apple Store)")
    st.write("Esta seção aplica aprendizado não supervisionado em 3 dimensões (**Avaliações**, **Idiomas** e **Preço**) para segmentar os aplicativos da base.")

    # 1. Carregamento dos dados (ajuste o caminho se necessário)
    @st.cache_data
    def carregar_dados():
        df = pd.read_csv('AppleStore.csv')
        df.columns = df.columns.str.lower().str.replace('.', '_', regex=False).str.strip()
        return df

    df = carregar_dados()

    # 2. Configurações e Filtros na Barra Lateral
    st.sidebar.header("⚙️ Parâmetros do Modelo 3D")
    optimal_k = st.sidebar.slider("Número de Clusters (k)", min_value=2, max_value=6, value=3)
    eps_val = st.sidebar.slider("DBSCAN (eps)", min_value=0.2, max_value=1.5, value=0.6, step=0.1)
    min_samp = st.sidebar.slider("DBSCAN (min_samples)", min_value=3, max_value=20, value=8)

    # 3. Preparação das Features
    features = ['rating_count_tot', 'lang_num', 'price']
    df_clean = df[(df['rating_count_tot'] <= 30000) & (df['lang_num'] <= 15)].dropna(subset=features).copy()
    X = df_clean[features].values

    # Padronização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Execução do K-Means e Centróides
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters_kmeans = kmeans.fit_predict(X_scaled)
    centroids = kmeans.cluster_centers_

    # 5. Execução do DBSCAN e Tratamento de Outliers
    dbscan = DBSCAN(eps=eps_val, min_samples=min_samp)
    dbscan_labels = dbscan.fit_predict(X_scaled)

    final_labels = clusters_kmeans.copy()
    for i in range(len(X_scaled)):
        if dbscan_labels[i] == -1:
            distances = [np.linalg.norm(X_scaled[i] - centroid) for centroid in centroids]
            final_labels[i] = np.argmin(distances) # Força a classe do centróide mais próximo

    # 6. Gráfico 3D Interativo com Plotly
    fig = go.Figure()

    # Adicionando os aplicativos
    fig.add_trace(go.Scatter3d(
        x=X_scaled[:, 0],
        y=X_scaled[:, 1],
        z=X_scaled[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=final_labels,
            colorscale='Viridis',
            opacity=0.7,
            line=dict(width=0)  # Remove as bordas pretas indesejadas
        ),
        text=[f"Cluster: {c}<br>Avaliações: {r}<br>Idiomas: {l}<br>Preço: {p}" 
              for c, r, l, p in zip(final_labels, df_clean['rating_count_tot'], df_clean['lang_num'], df_clean['price'])],
        hoverinfo='text',
        name='Apps Agrupados'
    ))

    # Adicionando os Centróides (X)
    fig.add_trace(go.Scatter3d(
        x=centroids[:, 0],
        y=centroids[:, 1],
        z=centroids[:, 2],
        mode='markers',
        marker=dict(
            size=10,
            color='red',
            symbol='x',
            line=dict(width=2, color='black')
        ),
        name='Centróides (X)'
    ))

    # Layout do Gráfico 3D
    fig.update_layout(
        title=f'Espaço Tridimensional (k={optimal_k})',
        scene=dict(
            xaxis_title='Avaliações Totais (Padronizado)',
            yaxis_title='Idiomas (Padronizado)',
            zaxis_title='Preço (Padronizado)'
        ),
        margin=dict(r=0, b=0, l=0, t=40),
        height=700
    )

    # Exibindo no Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Métricas e Insights rápidos abaixo do gráfico
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Apps Analisados", len(df_clean))
    with col2:
        st.metric("Clusters Definidos (k)", optimal_k)
    with col3:
        st.metric("Variáveis Utilizadas", "3 (Avaliações, Idiomas, Preço)")

if __name__ == "__main__":
    app()
