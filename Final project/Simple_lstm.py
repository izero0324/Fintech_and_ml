import torch
import os
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt
from optimizer import Ranger
import datetime
import pandas as pd
import torch.nn.functional as F
from torch.autograd import Variable
#from autoencoder import auto_encoder_FFN1
from dataloader import BTC_list, ETH_list, target_list

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)

rnn = nn.LSTM(10, 20, 2).to(device)
price_tensor = torch.from_numpy(ETH_list()).float().to(device)
key_tensor =  torch.from_numpy(BTC_list()).float().to(device)

output = rnn(price_tensor)