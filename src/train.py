from model import train_model

if __name__ == "__main__":
    data_path = "../inputs/dados_temperatura.csv"  # Caminho do arquivo de dados
    model = train_model(data_path)
    print("Modelo treinado e salvo com sucesso!")
