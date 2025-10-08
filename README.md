##🤖 Projeto de Previsão de Preços de Carros BMW

Este projeto utiliza Machine Learning para prever o valor de carros usados da marca BMW com base em suas características, como modelo, ano, quilometragem e tipo de combustível.
Para tornar a experiência interativa, foi desenvolvido um dashboard web onde o usuário pode inserir as informações de um veículo e receber uma estimativa de preço em tempo real.

(Sugestão: Tire um print da sua aplicação funcionando e substitua o link acima para exibir uma imagem real do seu dashboard!)

#🎯 Objetivo
O principal objetivo é aplicar técnicas de ciência de dados e aprendizado de máquina para construir um modelo preditivo preciso e, ao mesmo tempo, fornecer uma interface simples e intuitiva para que usuários finais possam interagir com o modelo sem precisar de conhecimentos técnicos.

#✨ Funcionalidades
Análise de Dados: Tratamento e preparação dos dados a partir do arquivo bmwdataset_tratado.csv.

Modelo Preditivo: Treinamento de um modelo de regressão (Random Forest) para estimar preços.

Dashboard Interativo: Interface web criada com Streamlit que permite ao usuário:

Selecionar o modelo do carro.

Definir o ano de fabricação.

Escolher o tipo de transmissão e combustível.

Informar a quilometragem, o consumo (MPG) e o tamanho do motor.

Previsão em Tempo Real: O dashboard utiliza o modelo treinado (.pkl) para gerar a previsão instantaneamente.

#🛠️ Tecnologias Utilizadas
Este projeto foi construído utilizando as seguintes tecnologias e bibliotecas Python:

Pandas: Para manipulação e análise dos dados.

Scikit-learn: Para criar o pipeline de pré-processamento e treinar o modelo de Machine Learning.

Streamlit: Para a construção e execução do dashboard web interativo.

Pickle: Para salvar e carregar o modelo treinado.

#📁 Estrutura do Projeto
PREVISAO_PRECOS_CARROS_BMW-main/
│
├──  Dashboard.py                    # Arquivo principal que executa a aplicação web (dashboard)
├── treinamento.py                    # Script para treinar e salvar o modelo de ML
├── geração da arvore.py              # Script para gerar a visualização da árvore de decisão
│
├── bmwdataset_tratado.csv            # Dataset com os dados limpos e tratados
├── modelo_previsao_veiculos_com_pipeline.pkl # Modelo de ML treinado e salvo
│
└── resultados/
    ├── arvore_de_decisao.png         # Imagem da árvore de decisão gerada
    └── arvore_completa.dot           # Arquivo de dados para a visualização da árvor****

#🧠 O Modelo de Machine Learning
Para a tarefa de predição, foi utilizado um algoritmo Random Forest Regressor.

O que é? É um modelo de conjunto (ensemble) que opera construindo múltiplas árvores de decisão durante o treinamento. Para uma previsão de regressão (como prever um preço), ele calcula a média das previsões de cada árvore individual, resultando em uma estimativa mais robusta e precisa.

Features Utilizadas: As seguintes características do carro foram usadas para treinar o modelo:

model (Modelo)

year (Ano)

transmission (Transmissão)

mileage (Quilometragem)

fuelType (Tipo de Combustível)

mpg (Milhas por Galão - Consumo)

engineSize (Tamanho do Motor)

Pipeline de Pré-processamento: Para garantir que os dados fossem inseridos corretamente no modelo, foi utilizado um Pipeline do Scikit-learn que automatiza as seguintes etapas:

One-Hot Encoding: Converte variáveis categóricas (como model e fuelType) em um formato numérico que o modelo consegue entender.

Standard Scaler: Padroniza as variáveis numéricas (como year e mileage), colocando-as na mesma escala para que nenhuma delas domine o processo de aprendizado.

Treinamento do Modelo: Após o pré-processamento, os dados são enviados ao RandomForestRegressor para treinamento.

Este pipeline é salvo no arquivo modelo_previsao_veiculos_com_pipeline.pkl, garantindo que qualquer nova previsão no dashboard passe exatamente pelas mesmas etapas de transformação.
