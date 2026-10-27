import streamlit as st
import pandas as pd

st.set_page_config(page_title="Dashboard Apple Store", layout="wide")
@st.cache_data
def carregar_dados():
    df = pd.read_csv("AppleStore.csv")
    df['tipo_preco'] = df['price'].apply(lambda x: 'Gratuito' if x == 0 else 'Pago')
    return df

if 'df_original' not in st.session_state:
    st.session_state['df_original'] = carregar_dados()

df_orig = st.session_state['df_original']

st.sidebar.header("🎯 Filtros do Dashboard")

opcoes_preco = ['Todos'] + list(df_orig['tipo_preco'].unique())
filtro_preco = st.sidebar.selectbox("Selecione o Modelo de Preço:", opcoes_preco)

opcoes_categoria = ['Todas'] + sorted(list(df_orig['prime_genre'].unique()))
filtro_categoria = st.sidebar.multiselect("Selecione as Categorias:", opcoes_categoria, default=['Todas'])

df_filtrado = df_orig.copy()

if filtro_preco != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['tipo_preco'] == filtro_preco]

if 'Todas' not in filtro_categoria and len(filtro_categoria) > 0:
    df_filtrado = df_filtrado[df_filtrado['prime_genre'].isin(filtro_categoria)]

st.session_state['df'] = df_filtrado

# Definição das Páginas do Streamlit
pg1 = st.Page("pagina1.py", title="📊 Visão Geral", default=True)
pg2 = st.Page("pagina2.py", title="📈 Análise de Avaliações")
pg3 = st.Page("clusters.py", title="🤖 Clusterização 3D")  # <-- Adicionado o clusters.py

pg = st.navigation([pg1, pg2, pg3])
pg.run()
