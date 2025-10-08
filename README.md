🤖 Projeto de Previsão de Preços de Carros BMW
Este projeto utiliza Machine Learning para prever o valor de carros usados da marca BMW com base em suas características, como modelo, ano, quilometragem e tipo de combustível.

Para tornar a experiência interativa, foi desenvolvido um dashboard web onde o usuário pode inserir as informações de um veículo e receber uma estimativa de preço em tempo real.

(Sugestão: Tire um print da sua aplicação funcionando e substitua o link acima para exibir uma imagem real do seu dashboard!)

🎯 Objetivo
O principal objetivo é aplicar técnicas de ciência de dados e aprendizado de máquina para construir um modelo preditivo preciso e, ao mesmo tempo, fornecer uma interface simples e intuitiva para que usuários finais possam interagir com o modelo sem precisar de conhecimentos técnicos.

✨ Funcionalidades
Análise de Dados: Tratamento e preparação dos dados a partir do arquivo bmwdataset_tratado.csv.

Modelo Preditivo: Treinamento de um modelo de regressão (Random Forest) para estimar preços.

Dashboard Interativo: Interface web criada com Streamlit que permite ao usuário:

Selecionar o modelo do carro.

Definir o ano de fabricação.

Escolher o tipo de transmissão e combustível.

Informar a quilometragem, o consumo (MPG) e o tamanho do motor.

Previsão em Tempo Real: O dashboard utiliza o modelo treinado (.pkl) para gerar a previsão instantaneamente.

🛠️ Tecnologias Utilizadas
Este projeto foi construído utilizando as seguintes tecnologias e bibliotecas Python:

Pandas: Para manipulação e análise dos dados.

Scikit-learn: Para criar o pipeline de pré-processamento e treinar o modelo de Machine Learning.

Streamlit: Para a construção e execução do dashboard web interativo.

Pickle: Para salvar e carregar o modelo treinado.
