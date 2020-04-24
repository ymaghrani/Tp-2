from import_data import *
import pandas as pd
import numpy as np
import csv

# Dataframe train
df_order_products_prior = importation_file("/Users/Maghrani/Desktop/5eBD/machine_learning/instacart/instacart-market-basket-analysis/order_products__prior.csv")
df_orders = importation_file("/Users/Maghrani/Desktop/5eBD/machine_learning/instacart/instacart-market-basket-analysis/orders.csv")
df_products = importation_file("/Users/Maghrani/Desktop/5eBD/machine_learning/instacart/instacart-market-basket-analysis/products.csv")

def read_orders(N,path="/Users/Maghrani/Desktop/5eBD/instacart/instacart-market-basket-analysis/orders.csv"):
    df_orders = pd.read_csv(path)
    df_orders = df_orders[['order_id','user_id']]
    return df_orders[df_orders['user_id'].isin(range(N+1))]

def read_orders_products(path="/Users/Maghrani/Desktop/5eBD/machine_learning/instacart/instacart-market-basket-analysis/order_products__prior.csv"):
    orders_products_df=pd.read_csv(path)
    return orders_products_df[['order_id','product_id']]


#récupérer les order_id pour chaque user_id
#retourne les order_id pour les N premiers clients
n_clients = 100
orders_df = read_orders(n_clients)

# retourne pour chaque client ['user_id','order_id','product_id','department_id']
orders_products_df= pd.merge(orders_df, df_order_products_prior,how='inner', on='order_id') 
orders_products_df= pd.merge(orders_products_df, df_products,how='inner', on='product_id') 
orders_products_df= orders_products_df[['user_id','order_id','department_id']]



#Création fichier data avec l'échantillon de clients
data = orders_products_df.values
