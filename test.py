from import_data import *
import pandas as pd
import numpy as np
import csv
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import *
from sklearn.model_selection import train_test_split
from keras.models import *
from keras.layers import *

#Création d'un fichier data avec l'échantillon de clients
data = pd.read_csv("/Users/Maghrani/Desktop/5eBD/machine_learning/instacart/instacart-market-basket-analysis/merged-sample.csv")

#vérification d'une order_id
d=data.loc[data['order_id'] == 431534]
print(d)


#initialisation d'un client
Liste=[]

#Boucle pour réaliser une liste des transactions de chaque client
for client in data.groupby("user_id"):
    client_id = client[0] #on retourne le 1er client_id
    client_data = client[1] # on retourne l'ensemble des order_id et produits du 1er client
    #

    client_transactions=[]

    #groupby des order_id d'un client spécifique
    for order in client_data.groupby("order_id"):
        order_id = order[0]#retourne le 1er order du client
        order_data = order[1]#possède l'ensemble des categories de produits de l'order
        department_ids = list(order_data["department_id"]) #on récupère dans une liste les catégories
        
        client_transactions.append(department_ids)
        
    Liste.append(client_transactions)

print("Liste des transactions de chaque client")
print(Liste)

df_products = importation_file("/Users/Maghrani/Desktop/5eBD/machine_learning/instacart/instacart-market-basket-analysis/products.csv")

i=df_products.department_id.unique()
print("unique categorie de produit",i)
print("nombre de categorie unique",len(i))


nb_clients=len(data.user_id.unique())
print("nombre de clients uniques:",nb_clients)
#Création une séquence limite de catégories de produit par transactions
seq_categorie= 8

#nombre de transactions par client
T=3

X1=np.zeros((nb_clients,T,seq_categorie))
X1.shape

for i in range(len(Liste)):
  X_pad =pad_sequences(Liste[i], maxlen=seq_categorie,dtype="int32", padding="post", truncating="pre", value=0)
  t=min(T,X_pad.shape[0])
  X_pad = X_pad [:T]
  X1[i]=X_pad

print(X1)

X = X1[:,:-1,:]
y = X1[:,-1,:]

print(X)

print(y)

print('shape:')
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y)

def LSTM_model(samples,num_feature):
  model = Sequential()
  model.add(LSTM(256,activation='relu', return_sequences=True, input_shape=(samples,num_feature)))
  model.add(LSTM(64, activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(y.shape[1], activation='softmax'))
  model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
  return model

Model=LSTM_model(X.shape[1], X.shape[2])

Model.summary()

callback_1 = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00005, patience=5)
callback_2 = callbacks.ModelCheckpoint(filepath='weights.hdf5', verbose=1, save_best_only=True)

Model.fit(X_train, y_train, epochs=600,batch_size=256,verbose=1, callbacks=[callback_1, callback_2])