import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

# Coleta dos dados de teste
df = pd.read_csv("C:/Users/Leozin/Desktop/Nova pasta/dados_banco_100.csv", encoding='utf-8')

# Separação da coluna de respostas
inputs = df.drop('Aceito', axis = 'columns')
inputs = inputs.dropna()
target = df['Aceito'].dropna()

# Criação de classe responsável por transformar os dados categóricos em dados numéricos
le_salario = LabelEncoder()
le_idade = LabelEncoder()
le_historico_de_credito = LabelEncoder()

# Transformação dos dados utilizando o Label Encoder
inputs['Salario_n'] = le_salario.fit_transform(inputs['Salario'])
inputs['Idade_n'] = le_idade.fit_transform(inputs['Idade'])
inputs['Historico_de_credito_n'] = le_historico_de_credito.fit_transform(inputs['Historico_de_Credito'])

# Remoção das colunas de dados categóricos
inputs_n = inputs.drop(['Salario','Idade','Historico_de_Credito'], axis='columns')

# Criação do modelo de árvore de decisão
model = tree.DecisionTreeClassifier()

# Treinamento do modelo com base nos dados de treinamento e suas devidas saídas esperadas
model.fit(inputs_n, target)

# Coleta dos dados de teste
dft = pd.read_csv("C:/Users/Leozin/Desktop/Nova pasta/dados_banco_teste.csv", encoding='latin1')

# Separação das repostas esperadas
inputst = dft.drop('Aceito', axis = 'columns')
inputst = inputst.dropna()
targett = dft['Aceito'].dropna()

# Transformação dos dados de teste
inputst['Salario_n'] = le_salario.fit_transform(inputst['Salario'])
inputst['Idade_n'] = le_idade.fit_transform(inputst['Idade'])
inputst['Historico_de_credito_n'] = le_historico_de_credito.fit_transform(inputst['Historico_de_Credito'])

# Remoção dos dados categóricos do conjunto de teste
inputs_nt = inputst.drop(['Salario','Idade','Historico_de_Credito'], axis='columns')

# Calcula e imprime a acurácia do modelo
print(model.score(inputs_nt,targett))
