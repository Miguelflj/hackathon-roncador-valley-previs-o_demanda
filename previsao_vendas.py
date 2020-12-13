from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import random
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor

vendas = []
for i in dados['vendas']:
	vendas.append(i)

df = pd.DataFrame(vendas)
print(df.head())

df.drop("preço", inline=True)




df_vendas_diarias = pd.read_csv('vendas.csv')
vendas = pd.DataFrame()

data = df_vendas_diarias['data_venda']
p1 = df_vendas_diarias['vendas_fone']
p2 = df_vendas_diarias['vendas_controle']
p3 = df_vendas_diarias['vendas_capinha']
p4 = df_vendas_diarias['vendas_relogio']

mtz = [p1,p2,p3,p4]

def soma_vendas(mtz):
	
	m = []
	k=0
	p1 = []
	p2 = []
	p3 = []
	p4 = []
	
	for i in range(7,len(mtz[0]),7):
		p1s = 0
		p2s = 0
		p3s = 0
		p4s = 0
		for j in range(k,i):
			p1s += mtz[0][j]
			p2s += mtz[1][j]
			p3s += mtz[2][j]
			p4s += mtz[3][j]
		
		k = i
		p1.append(p1s)
		p2.append(p2s)
		p3.append(p3s)
		p4.append(p4s)
	p1s = 0
	p2s = 0
	p3s = 0
	p4s = 0
	for j in range(i,len(mtz[0])):
		p1s += mtz[0][j]
		p2s += mtz[1][j]
		p3s += mtz[2][j]
		p4s += mtz[3][j]
	p1.append(p1s)
	p2.append(p2s)
	p3.append(p3s)
	p4.append(p4s)

	m.append(p1)
	m.append(p2)
	m.append(p3)
	m.append(p4)
	return m

mtz = soma_vendas(mtz)

vendas['cod_produto'] = [i for k in range(len(mtz[0])) for i in range(0,4)]	
vendas['semana'] = [k for k in range(len(mtz[0])) for i in range(1,5)]
vendas['vendas'] = [mtz[j][i] for i in range(len(mtz[0])) for j in range(4)]
#vendas['data_vendas'] = [data[i] for i in range(data.shape[0]) for k in range(4) ] 
vendas['vendas_ultima_semana'] = vendas.groupby(['cod_produto'])['vendas'].shift()
vendas['diff_vendas_ultima_semana'] = [(vendas.loc[i,'vendas'] - vendas.loc[i,'vendas_ultima_semana']) for i in range(vendas.shape[0])]

vendas = vendas.dropna()
print(vendas.head(10))


#erro quadratico medio
def rmsle(ytrue,ypred):
	return np.sqrt(mean_squared_log_error(ytrue,ypred))



#baseline
def preve_valor_semana_passada(df):
	mean_error = []
	for w in range(400,500):
		train = df_work[df_work['semana'] < w]
		test = df_work[df_work['semana'] == w]
		#print(train)

		p = test['vendas_ultima_semana'].values


		error = rmsle(test['vendas'].values, p)
		#print("error: %.5f" %(error) )
		mean_error.append(error)
	print("Erro medio: %.5f" %(np.mean(mean_error)))

preve_valor_semana_passada(vendas)

def preve_valor_rfr(df):
	x = vendas.drop(columns='vendas')

	y = vendas['vendas']
	x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2)
	rfr = RandomForestRegressor(n_estimators=300, n_jobs=-1,random_state=0)
	rfr.fit(x_train, y_train)
	p = rfr.predict(x_test)
	for i in range(len(p)):
		p[i] = round(p[i])

	print(rmsle(y_test,p))
	print(accuracy_score(y_test, p))
	return rfr

predict = preve_valor_rfr(vendas)