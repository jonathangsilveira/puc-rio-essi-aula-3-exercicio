import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from pickle import dump
from pickle import load
from numpy import dtypes

# Carrega arquivo csv usando Pandas usando uma URL

# Informa a URL de importação do dataset
url = "https://raw.githubusercontent.com/tatianaesc/datascience/main/diabetes.csv"

# Informa o cabeçalho das colunas
colunas: list[str] = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

# Lê o arquivo utilizando as colunas informadas
dataset: pd.DataFrame = pd.read_csv(url, skiprows=0, delimiter=',')

# Pega apenas os dados do dataset e guardando em um array
data: list[list[any]] = dataset.to_numpy()

# Separa o array em variáveis preditoras (X) e variável target (Y)
X: list[list[any]] = data[:,0:8]
Y: list[any] = data[:,8]

# Divide os dados em treino e teste
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=7)

# Cria o modelo
modelo = LogisticRegression(solver='liblinear')

# Treina o modelo
modelo.fit(X_train, Y_train)

# Salva o modelo no disco
filename = 'model.pkl'
dump(modelo, open(filename, 'wb'))

# Algum tempo depois...
# Carrega o modelo do disco
loaded_model = load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)