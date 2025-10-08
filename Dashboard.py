import streamlit as st
import plotly.express as px
import pandas as pd
import joblib
import numpy as np

# --- 1. CONFIGURAÇÃO DA PÁGINA E TÍTULOS ---
st.set_page_config(layout="wide")
st.title('Análise e Previsão de Preço de Carros')
st.subheader('Por Douglas Gobitsch, Cauã Guerreiro e Vinícius Raiol.')

# --- 2. LÓGICA DE CARREGAMENTO DE DADOS E MODELO ---

@st.cache_data
def carregar_dados_dashboard(filepath='bmwdataset_tratado.csv'):
    """Carrega o CSV, limpa e prepara os dados para o dashboard."""
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
        df.columns = df.columns.str.strip()
        
        if 'Modelo' in df.columns:
            # Lógica melhorada para extrair a marca
            df['Marca'] = df['Modelo'].str.split().str[0]
            # Caso especial para modelos como "3 Series" da BMW
            df['Marca'] = np.where(df['Marca'].str.match(r'^\d$'), 'BMW', df['Marca'])
        else:
            st.error("ERRO: Coluna 'Modelo' não encontrada.")
            return pd.DataFrame()

        df = df.rename(columns={
            'Preço_USD': 'preco',
            'Marca': 'marca',
            'Modelo': 'modelo',
            'Ano': 'ano'
        })
        
        df = df.dropna(subset=['preco', 'marca', 'modelo', 'ano'])
        if df['preco'].dtype == 'object':
             df['preco'] = df['preco'].astype(str).str.replace(r'[^\d.]', '', regex=True).astype(float)
        
        return df

    except FileNotFoundError:
        st.error(f"ERRO: O arquivo '{filepath}' não foi encontrado.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"ERRO ao ler ou processar o CSV: {e}")
        return pd.DataFrame()

@st.cache_resource
def carregar_modelo():
    """Carrega o modelo de machine learning salvo."""
    try:
        model = joblib.load('modelo_previsao_veiculos_com_pipeline.pkl')
        return model
    except FileNotFoundError:
        st.warning("AVISO: 'modelo_previsao_veiculos_com_pipeline.pkl' não encontrado.")
        return None

df_full = carregar_dados_dashboard()
model = carregar_modelo()

def prever_valorizacao_ou_depreciacao(df, ml_model):
    if ml_model is None or df.empty:
        df['previsao_preco'] = np.nan
        return df
    X = df[['ano', 'marca']].copy()
    try:
        df['previsao_preco'] = ml_model.predict(X)
    except Exception as e:
        st.error(f"Erro na previsão do modelo: {e}")
        df['previsao_preco'] = np.nan
    return df

# --- 3. BARRA LATERAL COM OS FILTROS ---
st.sidebar.header("Filtros")

if not df_full.empty:
    opcoes_marcas = list(df_full['marca'].unique())
    opcoes_marcas.sort()
    opcoes_marcas.insert(0, "Todas as Marcas")
    selected_brand = st.sidebar.selectbox("Selecione a Marca", opcoes_marcas)

    # Nomes dos gráficos atualizados para serem mais descritivos
    chart_type = st.sidebar.radio(
        "Selecione a Análise",
        ('Tendência de Preço (Previsão)', 'Distribuição de Preços por Modelo (Box Plot)', 'Top Modelos por Preço Médio', 'Volume de Anúncios por Marca')
    )
else:
    st.sidebar.text("Dados não carregados.")
    selected_brand = "Todas as Marcas"
    chart_type = 'Tendência de Preço (Previsão)'


# --- 4. LÓGICA DE EXIBIÇÃO DOS GRÁFICOS ---
if not df_full.empty:
    # Filtra os dados com base na seleção da barra lateral
    df_filtrado = df_full.copy()
    if selected_brand != "Todas as Marcas":
        df_filtrado = df_full[df_full['marca'] == selected_brand]

    # Mostra o gráfico correspondente
    if chart_type == 'Tendência de Preço (Previsão)':
        st.subheader("Tendência de Preço Previsto por Ano do Modelo")
        df_pred = prever_valorizacao_ou_depreciacao(df_filtrado.copy(), model)
        # Agrupa por ano para ver a tendência
        df_tendencia = df_pred.groupby('ano')['previsao_preco'].mean().reset_index()
        
        fig = px.line(df_tendencia, x="ano", y="previsao_preco", title=f"Tendência de Preço para: {selected_brand}", markers=True,
                      labels={'ano': 'Ano do Modelo', 'previsao_preco': 'Previsão de Preço Médio (USD)'})
        fig.update_traces(line=dict(width=3))
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == 'Distribuição de Preços por Modelo (Box Plot)':
        st.subheader("Distribuição de Preços por Modelo")
        fig = px.box(df_filtrado, x="modelo", y="preco", color="marca", title=f"Distribuição de Preços para: {selected_brand}",
                     labels={'modelo': 'Modelo do Veículo', 'preco': 'Preço Anunciado (USD)'})
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == 'Top Modelos por Preço Médio':
        st.subheader("Modelos com Maior Preço Médio de Anúncio")
        # Agrupa por modelo para obter o preço médio
        df_preco_medio = df_filtrado.groupby(['marca', 'modelo'])['preco'].mean().reset_index().sort_values('preco', ascending=False).head(20)
        fig = px.bar(df_preco_medio, x="modelo", y="preco", color="marca", title=f"Top 20 Modelos por Preço Médio ({selected_brand})",
                     labels={'modelo': 'Modelo do Veículo', 'preco': 'Preço Médio (USD)'})
        st.plotly_chart(fig, use_container_width=True)
            
    elif chart_type == 'Volume de Anúncios por Marca':
        st.subheader("Distribuição do Volume de Anúncios por Marca")
        # Conta o número de carros por marca
        df_volume = df_filtrado['marca'].value_counts().reset_index()
        df_volume.columns = ['marca', 'quantidade']
        fig = px.pie(df_volume, names='marca', values='quantidade', title=f"Volume de Anúncios ({selected_brand})")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Aguardando o carregamento dos dados para exibir os gráficos.")

