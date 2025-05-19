import mlflow
import mlflow.sklearn
import numpy as np

# Função para realizar previsões
def predict_temperature(temp):
    # Carregar o modelo
    model = mlflow.sklearn.load_model("model_logs/model")

    # Realizar a previsão
    prediction = model.predict(np.array([[temp]]))
    return prediction[0]

if __name__ == "__main__":
    temperature = float(input("Digite a temperatura atual: "))
    predicted_sales = predict_temperature(temperature)
    print(f"As vendas previstas para uma temperatura de {temperature}°C são: {predicted_sales:.0f} sorvetes.")
