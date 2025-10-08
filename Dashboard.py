import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from PIL import Image

# --- 1. CONFIGURAÇÃO DA PÁGINA E TÍTULOS ---
st.set_page_config(layout="wide")
st.title('Análise e Previsão de Preço de Carros')
st.subheader('Por Douglas Gobitsch, Cauã Guerreiro e Vinícius Raiol.')

# --- 2. LÓGICA DE CARREGAMENTO DE DADOS ---
@st.cache_data
def carregar_dados_dashboard(filepath='bmwdataset_tratado.csv'):
    """Carrega o CSV, limpa e prepara os dados para o dashboard."""
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
        # Renomeia colunas para um padrão (minúsculas, sem espaços)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('__', '_')
        
        # Renomeação específica para o seu dataset
        df = df.rename(columns={
            'modelo': 'modelo',
            'ano': 'ano',
            'região': 'regiao',
            'cor': 'cor',
            'tipo_combustível': 'tipo_combustivel',
            'câmbio': 'cambio',
            'potência': 'potencia',
            'quilometragem': 'quilometragem',
            'preço_usd': 'preco',
            'volume_vendas': 'volume_vendas'
        })

        # Extrai a marca a partir do modelo
        if 'modelo' in df.columns:
            df['marca'] = df['modelo'].str.split().str[0]
            df['marca'] = np.where(df['marca'].str.match(r'^\d$'), 'BMW', df['marca'])
        else:
            st.error("ERRO: Coluna 'modelo' não encontrada.")
            return pd.DataFrame()

        return df
    except FileNotFoundError:
        st.error(f"ERRO: O arquivo '{filepath}' não foi encontrado.")
        return pd.DataFrame()

# --- 3. CARREGAMENTO DOS DADOS ---
df = carregar_dados_dashboard('bmwdataset_tratado.csv')

# --- 4. CRIAÇÃO DO DASHBOARD ---
if not df.empty:
    st.sidebar.header("Filtros e Opções")
    
    # --- FILTRO DE MARCAS (COM OPÇÃO 'TODOS') ---
    marcas_disponiveis = ['Todos'] + sorted(df['marca'].unique().tolist())
    selected_brand = st.sidebar.selectbox('Selecione a Marca:', marcas_disponiveis)
    
    # Filtra o DataFrame com base na marca selecionada
    if selected_brand == 'Todos':
        df_filtrado_marca = df
    else:
        df_filtrado_marca = df[df['marca'] == selected_brand]

    # --- NOVO FILTRO DE MODELOS ---
    modelos_disponiveis = sorted(df_filtrado_marca['modelo'].unique().tolist())
    selected_models = st.sidebar.multiselect(
        'Selecione os Modelos:', 
        modelos_disponiveis, 
        default=modelos_disponiveis  # Por padrão, todos os modelos são selecionados
    )

    # Filtro final com base nos modelos selecionados
    if selected_models:
        df_filtrado_final = df_filtrado_marca[df_filtrado_marca['modelo'].isin(selected_models)]
    else:
        # Se nenhum modelo for selecionado, exibe um aviso e um DF vazio para evitar erros
        st.warning("Por favor, selecione pelo menos um modelo.")
        df_filtrado_final = pd.DataFrame()

    # --- SELEÇÃO DE VISUALIZAÇÃO ---
    chart_type = st.sidebar.selectbox(
        "Selecione a Visualização:",
        [
            'Distribuição de Preços por Modelo', 
            'Top Modelos por Preço Médio', 
            'Volume de Anúncios por Modelo', # <-- ALTERADO DE 'Marca' PARA 'Modelo'
            'Árvore de Decisão do Modelo'
        ]
    )
    
    # Define o título principal da página
    if selected_brand == 'Todos':
        st.header("Análises para todas as marcas")
    else:
        st.header(f"Análises para a marca: {selected_brand}")

    # --- LÓGICA PARA EXIBIR GRÁFICOS OU A ÁRVORE ---
    if not df_filtrado_final.empty:
        if chart_type == 'Distribuição de Preços por Modelo':
            st.subheader("Distribuição de Preços por Modelo")
            # Usar 'marca' como cor para diferenciar quando 'Todos' estiver selecionado
            fig = px.box(df_filtrado_final, x="modelo", y="preco", color="marca", title="Distribuição de Preços para os Modelos Selecionados",
                         labels={'modelo': 'Modelo do Veículo', 'preco': 'Preço Anunciado (USD)'})
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == 'Top Modelos por Preço Médio':
            st.subheader("Modelos com Maior Preço Médio de Anúncio")
            df_preco_medio = df_filtrado_final.groupby(['marca', 'modelo'])['preco'].mean().reset_index().sort_values('preco', ascending=False).head(20)
            fig = px.bar(df_preco_medio, x="modelo", y="preco", color="marca", title="Top 20 Modelos por Preço Médio",
                         labels={'modelo': 'Modelo do Veículo', 'preco': 'Preço Médio (USD)'})
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == 'Volume de Anúncios por Modelo':
            st.subheader("Distribuição do Volume de Anúncios por Modelo")
            df_volume = df_filtrado_final['modelo'].value_counts().reset_index()
            df_volume.columns = ['modelo', 'quantidade']
            # O gráfico de pizza aqui mostrará a proporção de cada modelo selecionado
            fig = px.pie(df_volume, names='modelo', values='quantidade', title='Volume de Anúncios para os Modelos Selecionados')
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == 'Árvore de Decisão do Modelo':
            st.subheader("Visualização da Árvore de Decisão")
            st.write("""
            Esta imagem representa uma das árvores de decisão que compõem o modelo preditivo (Random Forest). 
            Ela ilustra como o algoritmo toma decisões, dividindo os dados com base nas características dos veículos 
            para chegar a uma estimativa de preço. Para facilitar a visualização, a profundidade da árvore foi limitada.
            """)
            try:
                image = Image.open('resultados/arvore_de_decisao.png')
                st.image(image, caption='Visualização de uma Árvore de Decisão', use_column_width=True)
            except FileNotFoundError:
                st.error("O arquivo 'arvore_de_decisao.png' não foi encontrado. Certifique-se de que ele está na mesma pasta do seu script Dashboard.py.")

else:
    st.warning("Não foi possível carregar os dados. Verifique o console para mais detalhes.")