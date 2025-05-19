# Previsão de Vendas de Sorvete com Machine Learning

Este projeto utiliza Machine Learning para prever as vendas de sorvetes com base na temperatura ambiente, utilizando um modelo de regressão linear.

## Como Rodar

1. Clone o repositório.
2. Instale as dependências com o comando:
    ```bash
    pip install -r requirements.txt
    ```
3. Para treinar o modelo, execute o script:
    ```bash
    python src/train.py
    ```
4. Para realizar uma previsão, execute:
    ```bash
    python src/predict.py
    ```

## Insights

Durante o treinamento, o modelo apresentou um erro médio quadrático (MSE) de X. Isso indica que a temperatura tem uma boa correlação com as vendas de sorvete.

## Possíveis Melhorias

- Incorporar mais variáveis, como a umidade ou dia da semana.
- Explorar diferentes modelos de Machine Learning.
