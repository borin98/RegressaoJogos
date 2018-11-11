import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def montaRede ( previsao, vendaNa, vendaEu, vendaJp ) :
    """

    Função que monta
    a rede neural
    mult layer

    :return: aNN
    """

    tam = len ( previsao[0] )

    camadaEntrada = Input ( shape = (tam, ) )

    nNeuronios = int( ( len ( previsao ) + 3 )/2 )

    hiddenLayer1 = Dense(
        units = nNeuronios,
        activation = "relu",
    ) ( camadaEntrada )

    Dropout1 = Dropout( 0.1 ) ( hiddenLayer1 )

    hiddenLayer2 = Dense(
        units = nNeuronios,
        activation = "relu"
    ) ( hiddenLayer1 )

    Dropout2 = Dropout ( 0.15 ) ( hiddenLayer2 )

    saidaPrevisao1 = Dense(
        units = 1,
        activation = "linear"
    ) ( hiddenLayer2 )

    saidaPrevisao2 = Dense(
        units = 1,
        activation = "linear"
    ) ( hiddenLayer2 )

    saidaPrevisao3 = Dense(
        units = 1,
        activation = "linear"
    ) ( hiddenLayer2 )

    Regressor = Model (
        inputs = camadaEntrada,
        outputs = [ saidaPrevisao1, saidaPrevisao2, saidaPrevisao3 ]
    )

    Regressor.compile(
        optimizer = "adam",
        loss = "mse"
    )

    Regressor.fit ( previsao,
                    [vendaNa, vendaEu, vendaJp],
                    epochs = 5000000,
                    batch_size = 100)
    previsaoNa, previsaoEu, previsaoJṕ = Regressor.predict ( previsao )

    #   gráficos de análise

    #  predição na américa do norte
    plt.plot ( vendaNa, "r-", label = "VendasNaReal" )
    plt.plot ( previsaoNa, "b-", label = "PrevisaoVendasNa" )
    plt.title("Previsão de vendas na América do norte")
    plt.legend(loc = "upper right")
    plt.grid(True)
    plt.show()

    # predição na europa
    plt.plot(vendaEu, "r-", label = "VendasEuReal")
    plt.plot(previsaoEu, "b-", label = "PrevisaoVendasEu")
    plt.title("Previsão de vendas na Europa")
    plt.legend(loc = "upper right")
    plt.grid(True)
    plt.show()

    # predição no japão
    plt.plot(vendaNa, "r-", label = "VendasJpReal")
    plt.plot(previsaoNa, "b-", label = "PrevisaoVendasJp")
    plt.legend(loc = "upper left")
    plt.title("Previsão de vendas no Japão")
    plt.grid(True)
    plt.show()

def previsoesProcessig ( previsao ) :
    """
    Função que faz a o processamento dos
    parâmetros de previsores

    :param previsores:
    :return:
    """
    labelEnconder = LabelEncoder()

    previsao[:, 0] = labelEnconder.fit_transform ( previsao [:, 0] )
    previsao[:, 2] = labelEnconder.fit_transform ( previsao [:, 2] )
    previsao[:, 3] = labelEnconder.fit_transform ( previsao [:, 3].astype(str) )
    previsao[:, 8] = labelEnconder.fit_transform ( previsao [:, 8] )

    oneHotEncoder = OneHotEncoder ( categorical_features = [0, 2, 3, 8] )
    previsao = oneHotEncoder.fit_transform ( previsao ).toarray()

    return previsao

def dataProcessing (  ) :
    """
    Função que faz o pre-
    processing dos dados
    e retorna o dataset
    e a coluna de nomes
    dos jogos.

    :return: df
    :return: nome
    """

    df = pd.read_csv ("games.csv")
    df = df.replace("tbd",np.nan)
    df = df.replace("nan", np.zeros)

    df = df.drop("Other_Sales", axis = 1)
    df = df.drop("Global_Sales", axis = 1)
    df = df.drop("Developer", axis = 1)

    media = int(df["Critic_Score"].mean())
    df["Critic_Score"].fillna (media , inplace = True )
    df["User_Score"].fillna(5, inplace = True )
    df["Critic_Count"].fillna(50, inplace = True)
    df["User_Count"].fillna(500, inplace = True)
    df["Rating"].fillna("M", inplace = True)
    df["Year_of_Release"].fillna(2000, inplace = True)

    df = df.loc[df["NA_Sales"] > 1]
    df = df.loc[df["EU_Sales"] > 1]

    #print(df["Name"].value_counts())

    colJogos = df.Name
    df = df.drop("Name", axis = 1)

    labelEnconder = LabelEncoder()

    return df, colJogos

def main (  ) :
    """
    Modelo não sequencial para
    previsão de venda de jogos
    de video games

    :return:
    """
    df, colJogos = dataProcessing()

    previsao = df.iloc [ :, [0, 1, 2, 3, 7, 8, 9, 10, 11] ].values
    vendasNa = df.iloc [ :, 4 ].values
    vendasEu = df.iloc [ :, 5 ].values
    vendasJp = df.iloc [ :, 6 ].values

    previsao = previsoesProcessig ( previsao )

    montaRede ( previsao, vendasNa, vendasEu, vendasJp )

if __name__ == '__main__':
    main (  )