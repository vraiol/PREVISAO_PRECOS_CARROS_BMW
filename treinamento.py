from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import pandas as pd
import joblib
import os
import numpy as np

# --- 1. Carregar e preparar os novos dados ---
def carregar_novos_dados(filepath='bmwdataset_tratado.csv'):
    """Carrega o novo CSV e prepara as colunas para o treinamento."""
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"O arquivo '{filepath}' não foi encontrado.")
            
        df = pd.read_csv(filepath, encoding='utf-8')
        df.columns = df.columns.str.strip()

    except FileNotFoundError as e:
        print(f"ERRO: {e}")
        return pd.DataFrame() 
    except Exception as e:
        print(f"ERRO ao ler o CSV. Verifique o encoding ou o formato do arquivo: {e}")
        return pd.DataFrame()

    if 'Modelo' in df.columns:
        df['Marca'] = df['Modelo'].str.split().str[0]
        print("Coluna 'Marca' criada a partir da coluna 'Modelo'.")
    else:
        print("ERRO: Coluna 'Modelo' não encontrada para extrair a marca.")
        return pd.DataFrame()

    # --- CORREÇÃO APLICADA AQUI ---
    # Renomear as colunas do novo CSV para os nomes que o pipeline espera
    df = df.rename(columns={'Ano': 'ano', 'Marca': 'marca', 'Preço_USD': 'preco'})

    required_cols = ['ano', 'marca', 'preco']
    if not all(col in df.columns for col in required_cols):
        print("ERRO: Uma ou mais colunas ('ano', 'marca', 'preco') não foram encontradas após o renomeio.")
        print("Verifique se as colunas 'Ano' e 'Preço_USD' existem no seu CSV.")
        print(f"Colunas atuais no DataFrame: {list(df.columns)}")
        return pd.DataFrame()

    df = df.dropna(subset=['ano', 'marca', 'preco'])
    
    if df['preco'].dtype == 'object':
        df['preco'] = df['preco'].astype(str).str.replace(r'[^\d.]', '', regex=True).astype(float)
        
    return df

data = carregar_novos_dados('bmwdataset_tratado.csv') 

if data.empty:
    print("O treinamento não pode ser realizado devido à falta de dados.")
    exit()

# Separar variáveis independentes (X) e dependente (y)
X = data[['ano', 'marca']]
y = data['preco']

# Pré-processamento com pipeline
numeric_features = ['ano']
categorical_features = ['marca']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Modelo (RandomForestRegressor)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Divisão e Treinamento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar modelo
model.fit(X_train, y_train)

# Avaliar modelo
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\nTreinamento concluído. RMSE: {rmse}")

# Salvar modelo
joblib.dump(model, 'modelo_previsao_veiculos_com_pipeline.pkl')
print("Modelo salvo com sucesso como 'modelo_previsao_veiculos_com_pipeline.pkl'!")

