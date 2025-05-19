import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

# Função para treinar o modelo
def train_model(data_path):
    # Carregar os dados
    df = pd.read_csv(data_path)
    X = df[['temperatura']]  # Variável independente
    y = df['vendas']  # Variável dependente

    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Criar e treinar o modelo de regressão linear
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Fazer previsões e avaliar o modelo
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # Log do modelo com o MLflow
    mlflow.start_run()
    mlflow.log_param("model", "Linear Regression")
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model")
    mlflow.end_run()

    return model

# Função para salvar o modelo
def save_model(model, model_path):
    mlflow.sklearn.save_model(model, model_path)
