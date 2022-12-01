
import pandas as pd 
import numpy as np  


df_keyword = pd.read_csv('KEYWORD.csv', encoding = 'big5')
df_keyword = df_keyword.drop(['Date'], axis=1)

df_price = pd.read_csv('FUTURE.csv', encoding = 'big5')
df_price = df_price.drop(['Date'], axis=1)

df_target = df_price.loc[:, ['LME_Nickel_close']]

def keyword_list():
    keyword_data = []
    for i in range(188):
        a = df_keyword.iloc[i : i+31]
        a = np.array(a.as_matrix())
        keyword_data.append(a)
    return np.array(keyword_data)

keyword = keyword_list()

def price_list():
    price_data = []
    for i in range(188):
        b = df_price.iloc[i+1 : i+31]
        b = np.array(b.as_matrix())
        price_data.append(b)
    return np.array(price_data)

price = price_list()

def target_list():
    target_data = []
    for i in range(188):
        c = df_target.iloc[i+31 : i+53]
        c = np.array(c.as_matrix())
        target_data.append(c)
    return np.array(target_data)

target = target_list()