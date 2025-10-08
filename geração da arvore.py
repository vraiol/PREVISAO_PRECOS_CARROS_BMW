import joblib
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
from sklearn import tree
import pandas as pd
import os # Garante que o 'os' está importado no topo

# --- 1. Carregar o modelo e os dados de teste ---
try:
    # Carrega o pipeline completo que foi salvo
    model_pipeline = joblib.load('modelo_previsao_veiculos_com_pipeline.pkl')
    print("Modelo 'modelo_previsao_veiculos_com_pipeline.pkl' carregado com sucesso.")
except FileNotFoundError:
    print("ERRO: O arquivo do modelo não foi encontrado. Execute o script de treinamento primeiro.")
    exit()

# --- CÓDIGO CORRIGIDO AQUI ---
# Garante que a pasta de resultados exista ANTES de qualquer tentativa de salvar arquivos.
pasta_resultados = "resultados"
if not os.path.exists(pasta_resultados):
    os.makedirs(pasta_resultados)
    print(f"Pasta '{pasta_resultados}' criada com sucesso.")

# Para visualizar a árvore, precisamos dos nomes das features DEPOIS do pré-processamento.
# (O resto do código para extrair os nomes das features continua igual)
preprocessor = model_pipeline.named_steps['preprocessor']
categorical_features_raw = preprocessor.transformers_[1][2]
one_hot_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
numeric_features = preprocessor.transformers_[0][2]
categorical_features_encoded = one_hot_encoder.get_feature_names_out(categorical_features_raw)
feature_names = list(numeric_features) + list(categorical_features_encoded)

# --- 2. Extrair e Visualizar UMA Árvore da Floresta ---
random_forest_regressor = model_pipeline.named_steps['regressor']
single_tree = random_forest_regressor.estimators_[0]
print(f"\nExtraindo a primeira árvore de um total de {len(random_forest_regressor.estimators_)} árvores na floresta.")

# --- MÉTODO 1: Visualização com Matplotlib (Recomendado, mais simples) ---
print("\nGerando visualização da árvore com Matplotlib...")
plt.figure(figsize=(20, 15))
tree.plot_tree(
    single_tree,
    feature_names=feature_names,
    filled=True,
    rounded=True,
    max_depth=4,
    fontsize=8
)
plt.title("Visualização de uma Árvore de Decisão do RandomForest (profundidade limitada a 4)")
# Agora esta linha vai funcionar, pois a pasta 'resultados' já foi criada
plt.savefig(os.path.join(pasta_resultados, "arvore_de_decisao.png"))
print(f"Imagem 'arvore_de_decisao.png' salva na pasta '{pasta_resultados}/'.")


# --- MÉTODO 2: Visualização com Graphviz (Qualidade maior, um pouco mais complexo) ---
print("\nGerando arquivo .dot para visualização com Graphviz...")
# Não precisamos mais criar a pasta aqui, pois já foi feito no início.
dot_data = export_graphviz(
    single_tree,
    out_file=os.path.join(pasta_resultados, 'arvore_completa.dot'),
    feature_names=feature_names,
    filled=True,
    rounded=True,
    special_characters=True,
    max_depth=4
)
print(f"Arquivo 'arvore_completa.dot' salvo na pasta '{pasta_resultados}/'.")
print("Para converter para PNG, use o comando no terminal: dot -Tpng resultados/arvore_completa.dot -o resultados/arvore_completa.png")
