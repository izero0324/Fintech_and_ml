
import pandas as pd 
import numpy as np  


df_BTC = pd.read_csv('Bitcoin Historical Data_mod.csv')
df_BTC = df_BTC.drop(['Date'], axis=1)


df_ETH = pd.read_csv('Ethereum Historical Data_mod.csv')
df_ETH = df_ETH.drop(['Date'], axis=1)

df_target = df_ETH.loc[:, ['Price']]

def BTC_list():
    BTC_data = []
    for i in range(1023):
        a = df_BTC.iloc[i : i+31].astype(float)
        #a = np.array(a.values)
        BTC_data.append(a.values)
    return np.array(BTC_data)

BTC = BTC_list()
#print(BTC.dtype)

def ETH_list():
    ETH_data = []
    for i in range(1023):
        b = df_ETH.iloc[i : i+31].astype(float)
        #b = np.array(b.values)
        ETH_data.append(b.values)
    return np.array(ETH_data)

ETH = ETH_list()

def target_list():
    target_data = []
    for i in range(1013):
        c = df_target.iloc[i+31 : i+41].astype(float)
        #c = np.array(c.values)
        target_data.append(c.values)
    return np.array(target_data)

target = target_list()
#print(target)

