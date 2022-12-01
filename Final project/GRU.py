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

class GRUNet(nn.Module):
  def __init__(self,input_length,target_length):
    super(GRUNet, self).__init__()
    self.sent_rnn = nn.GRU(self.input_length,
                            self.target_length,
                            bidirectional=True,
                            batch_first=True)
    self.l1 = nn.Linear(self.hidden_dim*2, self.hidden_dim)
    self.l2 = nn.Linear(self.hidden_dim, 6)

  def forward(self, x):
    x, __ = self.sent_rnn(x)
    # x: (batch,sent*word,hidden_state*2)
    x = x.view(b,s,w,-1)
    # x: (batch,sent,word,hidden_state*2)
    x = torch.max(x,dim=2)[0]
    # x: (batch,sent,hidden_state*2)
    x = torch.relu(self.l1(x))
    x = torch.sigmoid(self.l2(x))
    # x: (batch,sent,6)
    return x

def train(price_tensor, key_tensor, target_tensor, decoder, optimizer, criterion, num_epochs):
    
    price_tensor = torch.from_numpy(price_tensor).float().to(device)
    key_tensor =  torch.from_numpy(key_tensor).float().to(device)
    target_tensor =  torch.from_numpy(target_tensor).float().to(device)
    optimizer.zero_grad()

    input_length = price_tensor.size(0)
    target_length = target_tensor.size(0)

    best = 1e8

    for epoch in range(num_epochs):
        total_loss = 0
        for ei in range(input_length):

            output = GRUNet(price_tensor[ei], key_tensor[ei])
            #decoder_output = simpleNet(price_tensor[ei]+key_tensor[ei]) 

            loss = criterion(output, target_tensor[ei].squeeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(decoder_output, target_tensor[ei].squeeze(1))
        average_loss = total_loss/target_length
        print('epoch:', epoch, 'loss:', average_loss)

if __name__ == "__main__":
    
    model = GRUNet().to(device)
    loss_fn = nn.MSELoss()
    epochs = 1000
    price = ETH_list()
    key = BTC_list()
    target = target_list()
    optimizer = Ranger(model.parameters())
    #resume = 'savemodel/best_999.pth'
    #model, _, _ = load_model(model, optimizer, resume)
    #print(resume,'Loaded!')
    train_loss = train(price, key, target, model, optimizer, loss_fn, epochs)
    #predict = predict(price, key, target, model)