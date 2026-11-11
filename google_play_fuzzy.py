!pip install scikit-fuzzy
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import skfuzzy as fuzz

# 1. Carregar a base
df = pd.read_csv('google_play_store_dataset.csv')

# 2. Limpar e converter colunas para numérico
df = df.dropna(subset=['Rating', 'Reviews', 'Installs', 'Price'])

# Limpar 'Installs' (remover '+' e ',')
df['Installs'] = df['Installs'].astype(str).str.replace('+', '').str.replace(',', '')
df = df[df['Installs'].str.isnumeric()]
df['Installs'] = df['Installs'].astype(float)

# Limpar 'Price' (remover '$')
df['Price'] = df['Price'].astype(str).str.replace('$', '')
df = df[df['Price'].str.replace('.', '', 1).str.isnumeric()]
df['Price'] = df['Price'].astype(float)

# Converter 'Reviews' para numérico
df['Reviews'] = df['Reviews'].astype(float)

# 3. Selecionar as variáveis para a Lógica Fuzzy
features = ['Rating', 'Reviews', 'Installs', 'Price']
data_cluster = df[features].values

# 4. Normalizar os dados (Fuzzy é sensível à escala)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_cluster)
# O algoritmo skfuzzy precisa dos dados transpostos (linhas = variáveis, colunas = amostras)
data_fuzzy = data_scaled.T

# Definir número de clusters (c=3) e o fator de fuzzificação (m=2)
n_clusters = 3
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data_fuzzy, c=n_clusters, m=2, error=0.005, maxiter=1000, init=None
)

# 'u' contém a matriz de pertinência (grau de 0 a 1 de cada app em cada cluster)
# Vamos adicionar esses resultados de volta ao DataFrame original
for i in range(n_clusters):
    df[f'Pertinencia_Grupo_{i+1}'] = u[i]

# Identificar o grupo principal (maior pertinência), embora o ganho real seja o percentual
df['Grupo_Dominante'] = np.argmax(u, axis=0) + 1
import plotly.graph_objects as go
import numpy as np

# 1. Trocar o Preço por Instalações (ou Reviews) para espalhar a concentração
x_data = df['Rating']
y_data = df['Installs']  # Variável de volume mostra o real sucesso do app

# 2. Cor baseada na pertinência fuzzy do Grupo 1
cores_fuzzy = df['Pertinencia_Grupo_1']

# 3. Gerar o novo gráfico
fig = go.Figure(data=[go.Scatter(
    x=x_data,
    y=y_data,
    mode='markers',
    marker=dict(
        size=6,
        color=cores_fuzzy,
        colorscale='Viridis',
        opacity=0.6,             # Um pouco mais de transparência ajuda a ver onde há acúmulo de pontos
        showscale=True,
        colorbar=dict(
            title="Pertinência G1",
            tickfont=dict(color='white'),
            titlefont=dict(color='white')
        )
    )
)])

# 4. Customização essencial do Layout
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='#111111',
    plot_bgcolor='#111111',
    xaxis=dict(
        title='Avaliações (Rating)',
        gridcolor='#222222',
        tickfont=dict(color='white'),
        titlefont=dict(color='white'),
        range=[1, 5]             # Limita o eixo X ao intervalo real de notas
    ),
    yaxis=dict(
        title='Volume de Instalações (Escala Log)',
        type='log',              # <--- ESTA LINHA ESPALHA OS DADOS VERTICALMENTE
        gridcolor='#222222',
        tickfont=dict(color='white'),
        titlefont=dict(color='white')
    ),
    margin=dict(l=40, r=40, b=40, t=50),
    title=dict(
        text="Transição Fuzzy: Qualidade vs Popularidade dos Apps",
        font=dict(color='white', size=16)
    )
)

fig.show()
