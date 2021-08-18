# if 'google.colab' in str(get_ipython()):
#   from google.colab import drive
#   drive.mount('/content/drive')
#   # !pip install Embedding
#   location = 'drive/My Drive/Colab Notebooks/PDL212/asmn 2/'
# else:
location = './'

import torch
import random
import numpy as np
from matplotlib import pyplot as plt
import torchvision
plt.show()
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
import scipy.io
import numpy as np
import random
import matplotlib.pyplot as plt

from matplotlib import pyplot as plt
plt.show()
from numpy import genfromtxt
np.random.seed(0)

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np


# from sklearn.metrics import mean_squared_error


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# test = sio.loadmat("drive/My Drive/Colab Notebooks/cv202/assignment 8/test_32x32.mat" )
# a=sio.loadmat("drive/My Drive/Colab Notebooks/PeaksData.mat" )

# Define the LSTM-NN

# Fully connected neural network with one hidden layer
class LSTM_autoencoder(nn.Module):
    def __init__(self, input_size, sequence_length, encoded_size, hidden_size, num_layers, num_classes):
        super(LSTM_autoencoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
#         self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # -> x needs to be: (batch_size, seq, input_size)
        self.sequence_length = sequence_length
        
        self.lstm_encoder = nn.LSTM(input_size, encoded_size, num_layers, batch_first=True)
        self.lstm_decoder = nn.LSTM(encoded_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
        
    def forward(self, x):
      
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 

        encoded_sequence, _ = self.lstm_encoder(x, (h0,c0))  
        
        encoded_sequence = encoded_sequence[:, -1, :]
        
        encoded_sequence = torch.unsqueeze(encoded_sequence, 1)
        
        
        decoded_sequence = []
        hidden_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        cell_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        for i in range(self.sequence_length):
            decoded_element, (hidden_state, cell_state) = self.lstm_decoder(encoded_sequence, (hidden_state,cell_state))
            
            decoded_sequence.append(self.fc(decoded_element))
        return torch.cat(decoded_sequence, 1)

## Question 3.3.1

import pandas as pd
from tqdm import tqdm

stocks = pd.read_csv(location+'SP 500 Stock Prices 2014-2017.csv')

price_df = stocks.pivot(index='date',columns='symbol',values='high').dropna(how='any',axis=1)
amzn = price_df['AMZN']
googl = price_df['GOOGL']

# amzn_plot = 
amzn.plot(title='AMZN daily maximum rate')
# amzn_fig = amzn_plot.get_figure()
# amzn_fig.savefig(location+'AMZN daily max.png')

pd_fig = googl.plot(title='daily maximum values')
stocks_plt = pd_fig.get_figure()
stocks_plt.legend()
stocks_plt.savefig(location+'daily maximum values.png')
# stocks_plt.close()

## Question 3.3.2

price_df = stocks.pivot(index='date',columns='symbol',values='close').dropna(how='any',axis=1)

prices_mean = price_df.mean().mean()
prices = price_df - prices_mean

prices_max = prices.max().max()
prices_min = prices.min().min()

prices_normalized = (prices-prices_min)/(prices_max-prices_min)

prices = prices_normalized.to_numpy()

# Fully connected neural network with one hidden layer
class LSTM_autoencoder(nn.Module):
    def __init__(self, input_size, sequence_length, encoded_size, hidden_size, num_layers, num_classes):
        super(LSTM_autoencoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
#         self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # -> x needs to be: (batch_size, seq, input_size)
        self.sequence_length = sequence_length
        
        self.lstm_encoder = nn.LSTM(input_size, encoded_size, num_layers, batch_first=True)
        self.lstm_decoder = nn.LSTM(encoded_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
        
    def forward(self, x):
      
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 

        encoded_sequence, _ = self.lstm_encoder(x, (h0,c0))  
        
        encoded_sequence = encoded_sequence[:, -1, :]
        
        encoded_sequence = torch.unsqueeze(encoded_sequence, 1)
        
        
        decoded_sequence = []
        hidden_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        cell_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        for i in range(self.sequence_length):
            decoded_element, (hidden_state, cell_state) = self.lstm_decoder(encoded_sequence, (hidden_state,cell_state))
            decoded_sequence.append(self.fc(decoded_element))
        return torch.cat(decoded_sequence, 1)

def construct_loaders(X_data, cross_validation_cur_round=None,
                      cross_validation_total_rounds=None):
  batch_size = 32
  test_perc = 0.15

  if cross_validation_cur_round is None:
    idx1 = 0
    idx2 = int(X_data.shape[0]*test_perc)

    test_dataset = X_data[idx1:idx2]
    train_dataset = np.concatenate((X_data[:idx1], X_data[idx2:]))
  else:
    shuffle_indices = np.arange(X_data.shape[0])
    np.random.shuffle(shuffle_indices)
    test_dataset = X_data[shuffle_indices[:int(test_perc*X_data.shape[0])]]
    train_dataset = X_data[shuffle_indices[int(test_perc*X_data.shape[0]):]]
 

  train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True)

  test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=False)
  return train_loader, test_loader

def define_and_train_model(input_size, modified_seq_length,
                           encoded_size, hidden_size, learning_rate,
                           train_loader, num_epochs, grad_clip):
                           
  global model, criterion, optimizer
  model = LSTM_autoencoder(input_size,
                          modified_seq_length,
                          encoded_size,
                          hidden_size,
                          num_layers=1,
                          num_classes=1
                          ).to(device)
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  # Train the model

  n_total_steps = len(train_loader)
  for epoch in range(num_epochs):
      for i, stocks in enumerate(train_loader):          
          # Forward pass
          outputs = model(stocks.to(device).float())
          loss = criterion(outputs, stocks.to(device).float())
          
          # Backward and optimize
          optimizer.zero_grad()
          loss.backward()
          if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
          optimizer.step()
          # if (i+1) % 100 == 0:
          if i == n_total_steps-1:
            if num_epochs<100 or epoch%100==0:
            # if True:
              print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item()}')
  return model, criterion, optimizer

def plot_examples(model, input_size, learning_rate, hidden_size, grad_clip, num_epochs, test_loader, modified_seq_length, with_predictions=False):
    some_batch = next(iter(test_loader))
    some_stocks = some_batch[0:2]
# global some_preds
    some_preds = model(some_stocks.to(device).float())


    some_stocks_np = some_stocks.numpy()
    if not with_predictions:
        some_preds_np = some_preds.detach().cpu().numpy()
    else:
        pred1 = some_preds[0].detach().cpu().numpy()
        pred2 = some_preds[1].detach().cpu().numpy()
        some_preds_np = np.concatenate((pred1, pred2))


    some_stocks_np = some_stocks_np*(prices_max-prices_min) + prices_min + prices_mean
    some_preds_np = some_preds_np*(prices_max-prices_min) + prices_min + prices_mean


    stock1 = some_stocks_np[0].reshape((modified_seq_length*input_size))
    stock2 = some_stocks_np[1].reshape((modified_seq_length*input_size))

    pred1 = some_preds_np[0].reshape((modified_seq_length*input_size))
    pred2 = some_preds_np[1].reshape((modified_seq_length*input_size))



    x_axis = np.arange(1,(modified_seq_length*input_size)+1)


    plt.plot(x_axis,stock1,label=r'stock 1')
    plt.plot(x_axis,stock2,label=r'stock 2')


    plt.plot(x_axis,pred1,label=r'reconstruction 1')
    plt.plot(x_axis,pred2,label=r'reconstruction 2')




    plt.title("Examples for reconstructed stocks")

    # plt.title("input_size={}, lr={}, hidden/encoded size={}, grad_clip={}, epochs={}".format(
    #     input_size,learning_rate, hidden_size,grad_clip, num_epochs))
    plt.xlabel(r'$t$ = time')
    plt.ylabel("value")
    plt.legend()
    plt.show()
    plt.close()

def calc_test_err(model, test_loader, criterion):
  global stocks, outputs
  total_loss = 0
  for _, stocks in enumerate(test_loader):
    outputs = model(stocks.to(device).float())
    loss = criterion(outputs, stocks.to(device).float())
    total_loss += loss.detach().cpu().numpy()
  return total_loss/len(test_loader)

def run_from_scratch(num_epochs, learning_rate, grad_clip, hidden_size,
                     encoded_size, input_size, cross_validation_cur_round=None,
                     cross_validation_total_rounds=None):
  orig_sequence_length = prices.shape[0]
  num_of_expls = prices.shape[1]
  modified_seq_length = int(orig_sequence_length/input_size)

  prices_cut = prices[:input_size*modified_seq_length,:].T
  X_data = prices_cut.reshape((-1, modified_seq_length, input_size))

  # global model, criterion, test_loader

  train_loader, test_loader = construct_loaders(X_data,
                                                cross_validation_cur_round,
                                                cross_validation_total_rounds)
  model, criterion, optimizer = define_and_train_model(input_size, modified_seq_length,
                            encoded_size, hidden_size, learning_rate,
                            train_loader, num_epochs, grad_clip)
  if cross_validation_cur_round is None:
    plot_examples(model, input_size, learning_rate, hidden_size,
                      grad_clip, num_epochs, test_loader, modified_seq_length,
                  with_predictions=False)
    return model, criterion, optimizer, train_loader, test_loader, modified_seq_length
  else:
    test_err = calc_test_err(model, test_loader, criterion)
    return test_err

def cross_validation(num_epochs):

  learning_rate_options = [0.01,0.001]
  grad_clip_options = [8, 14]
  hidden_size_options = [128, 256]
  input_size_options = [8, 16, 32, 64]

  cross_validation_total_rounds = (len(learning_rate_options)
  * len(grad_clip_options)
  * len(hidden_size_options)
  * len(input_size_options))

  best_loss = float('inf')

  best_learning_rate = learning_rate_options[0]
  best_grad_clip = grad_clip_options[0]
  best_hidden_size = hidden_size_options[0]
  best_input_size = input_size_options[0]  

  cross_validation_round = 1

  i=1
  for learning_rate in learning_rate_options:
    for grad_clip in grad_clip_options:
      for hidden_size in hidden_size_options:
        encoded_size = hidden_size
        for input_size in input_size_options:
          cur_loss = run_from_scratch(num_epochs, learning_rate, grad_clip, hidden_size,
                       encoded_size, input_size,
                       cross_validation_round,
                       cross_validation_total_rounds)
          print('{}/{}'.format(i,cross_validation_total_rounds),
                'parameters:',
                'learning_rate= {}'.format(learning_rate),
                'grad_clip= {}'.format(grad_clip),
                'hidden_size= {}'.format(hidden_size),
                'encoded_size= {}'.format(encoded_size),
                'input_size= {}'.format(input_size),
                sep=', ')
          i=i+1
          if cur_loss<best_loss:
            best_loss = cur_loss
            best_learning_rate = learning_rate
            best_grad_clip = grad_clip
            best_hidden_size = hidden_size
            best_input_size = input_size 
  return best_loss, best_learning_rate, best_grad_clip, best_hidden_size, best_input_size

num_epochs = 1
#Here
best_loss, best_learning_rate, best_grad_clip, best_hidden_size, best_input_size = cross_validation(num_epochs)

learning_rate = best_learning_rate
grad_clip = best_grad_clip
hidden_size = best_hidden_size
encoded_size = hidden_size
input_size = best_input_size

print('best parameters:',
      'learning_rate= {}'.format(learning_rate),
      'grad_clip= {}'.format(grad_clip),
      'hidden_size= {}'.format(hidden_size),
      'encoded_size= {}'.format(encoded_size),
      'input_size= {}'.format(input_size),
      sep='\n')

num_epochs = 1
# num_epochs = 1000
#here
model, criterion, optimizer, train_loader, test_loader, modified_seq_length = run_from_scratch(num_epochs,
                                                                                               learning_rate, grad_clip, hidden_size,
                     encoded_size, input_size, cross_validation_cur_round=None,
                     cross_validation_total_rounds=None)

# Question 3.3.3

# LSTM autoencoder
class LSTM_autoencoder(nn.Module):
    def __init__(self, input_size, sequence_length, encoded_size, hidden_size, num_layers):
        super(LSTM_autoencoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.classifier_hidden_size = encoded_size # maybe we should change it 
#         self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # -> x needs to be: (batch_size, seq, input_size)
        
        self.lstm_encoder = nn.LSTM(input_size, encoded_size, num_layers, batch_first=True)
        self.lstm_decoder = nn.LSTM(encoded_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
        
        # define the classifier - takes a hidden state of the encoder and returns a predictor for the next input in the series.
        self.fc_classifier1 = nn.Linear(hidden_size, self.classifier_hidden_size) # The input to the classifier is the last hidden-state of the decoder.
        self.relu = nn.ReLU()
        self.fc_classifier2 = nn.Linear(self.classifier_hidden_size, input_size)
        
    def forward(self, x):
#         print("input shape: ", x.size())
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) # x.size(0) is the batch size.
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # the above two lines with .double in the end
        
        # x: (n, 28, 28), h0: (2, n, 128)
        
        # Forward propagate RNN
        encoded_sequence, _ = self.lstm_encoder(x, (h0,c0))  
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # classifier
#         print("shape of encoded secuence: ", encoded_sequence.size())
#         print("sequence length: ", self.sequence_length)
        predictions = self.fc_classifier1(encoded_sequence) # decoded element is the hidden-state of the top LSTM layer of the decoder.
        predictions = self.relu(predictions)
        predictions = self.fc_classifier2(predictions)
#         print("predictions shape: ", predictions.size())
#         print("predictions without the last prediction shape: ", predictions[:,:-1,:].size())
#         print("first time step shape: ", torch.unsqueeze(x[:,1,:], 1).size())
        predictions = torch.cat((torch.unsqueeze(x[:,1,:], 1), predictions[:,:-1,:]), 1)
#         print("predictions shape: ", predictions.size())
        

        
        
        # Decode the hidden state of the last time step
        encoded_sequence = encoded_sequence[:, -1, :]
        # out: (n, 128)
#         print(encoded_sequence.size())
#         print(torch.unsqueeze(encoded_sequence, 1).size())
        encoded_sequence = torch.unsqueeze(encoded_sequence, 1)
        
        
        decoded_sequence = []
        hidden_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)#.double() 
        cell_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)#.double()
        for i in range(self.sequence_length):
            decoded_element, (hidden_state, cell_state) = self.lstm_decoder(encoded_sequence, (hidden_state,cell_state))
#             print(hidden_state.size())
            decoded_element = decoded_element[:, -1, :]
            decoded_element = self.fc(decoded_element)
            decoded_element = torch.unsqueeze(decoded_element, 1)
            decoded_sequence.append(decoded_element)
#         out = self.fc(out)
        # out: (n, 10)
        return torch.cat(decoded_sequence, 1), predictions # turns a list of tensors to a tensor.

def plot_examples(model, input_size, learning_rate, hidden_size,
                grad_clip, num_epochs, test_loader, modified_seq_length,
                with_predictions=False):
    some_batch = next(iter(test_loader))
    some_stocks = some_batch[0:2]
    global some_preds
    some_reconstructions, some_predictions = model(some_stocks.to(device).float())


    some_stocks_np = some_stocks.numpy()


    reconstruction1 = some_reconstructions[0].detach().cpu().numpy()
    reconstruction2 = some_reconstructions[1].detach().cpu().numpy()

    pred1 = some_predictions[0].detach().cpu().numpy()
    pred2 = some_predictions[1].detach().cpu().numpy()



    some_stocks_np = some_stocks_np*(prices_max-prices_min) + prices_min + prices_mean

    reconstruction1 = reconstruction1*(prices_max-prices_min) + prices_min + prices_mean
    reconstruction2 = reconstruction2*(prices_max-prices_min) + prices_min + prices_mean

    pred1 = pred1*(prices_max-prices_min) + prices_min + prices_mean
    pred2 = pred2*(prices_max-prices_min) + prices_min + prices_mean


    stock1 = some_stocks_np[0].reshape((modified_seq_length*input_size))
    stock2 = some_stocks_np[1].reshape((modified_seq_length*input_size))

    reconstruction1 = reconstruction1.reshape((modified_seq_length*input_size))
    reconstruction2 = reconstruction2.reshape((modified_seq_length*input_size))

    pred1 = pred1.reshape((modified_seq_length*input_size))
    pred2 = pred2.reshape((modified_seq_length*input_size))


    x_axis = np.arange(1,(modified_seq_length*input_size)+1)


    plt.plot(x_axis,stock1,label=r'stock 1')
    plt.plot(x_axis,stock2,label=r'stock 2')


    plt.plot(x_axis,pred1,label=r'prediction 1')
    plt.plot(x_axis,pred2,label=r'prediction 2')


    plt.plot(x_axis,reconstruction1,label=r'reconstruction 1')
    plt.plot(x_axis,reconstruction2,label=r'reconstruction 2')



    # plt.title("Examples for reconstructed stocks")
    plt.title("input_size={}, lr={}, hidden/encoded size={}, grad_clip={}, epochs={}".format(
        input_size,learning_rate, hidden_size,grad_clip, num_epochs))
    plt.xlabel(r'$t$ = time')
    plt.ylabel("value")
    plt.legend()
    plt.show()
    plt.close()

def calc_loss(loader, model):

    criterion = nn.MSELoss()
    with torch.no_grad():
        loss = 0
        for _, stocks in enumerate(loader):          
            reconstructions, predictions = model(stocks.to(device).float())


            loss += (criterion(reconstructions, stocks.to(device).float())
            + criterion(predictions, stocks.to(device).float()))
    return loss/len(loader)

def calc_prediction_loss(loader, model):

    criterion = nn.MSELoss()
    with torch.no_grad():
        loss = 0
        for _, stocks in enumerate(loader):          
            reconstructions, predictions = model(stocks.to(device).float())


            loss += criterion(predictions, stocks.to(device).float())
    return loss/len(loader)

def define_and_train_model(input_size, modified_seq_length,
                           encoded_size, hidden_size, learning_rate,
                           train_loader, num_epochs, grad_clip,
                           epochs_passed=0, criterion=None,
                           optimizer=None, test_loader=None, model=None):
  
    if epochs_passed == 0:
        model = LSTM_autoencoder(input_size,
                            modified_seq_length,
                            encoded_size,
                            hidden_size,
                            num_layers=1
                            # num_classes=1
                            ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Train the model


    training_loss_arr = []
    prediction_loss_arr = []
    time_arr = []

    n_total_steps = len(train_loader)
    for epoch in range(epochs_passed, epochs_passed+num_epochs):
        for i, stocks in enumerate(train_loader):        
      # Forward pass
      # outputs = model()
      # loss = criterion(outputs, stocks.to(device).float())

      
            outputs, predictions = model(stocks.to(device).float())
            loss = criterion(outputs, stocks.to(device).float()) + criterion(predictions, stocks.to(device).float())
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            # if (i+1) % 100 == 0:
            if i == n_total_steps-1:
                if num_epochs<=100 or epoch%100==0:
                    print (f'Epoch [{epoch+1}/{epochs_passed+num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item()}')
          
    time_arr.append(epoch+1)
    training_loss_arr.append(calc_loss(train_loader, model))
    prediction_loss_arr.append(calc_prediction_loss(train_loader, model))
  
    return model, criterion, optimizer, time_arr, training_loss_arr, prediction_loss_arr

def run_from_scratch(num_epochs, learning_rate, grad_clip, hidden_size,
                     encoded_size, input_size, cross_validation_cur_round=None,
                     cross_validation_total_rounds=None):
  orig_sequence_length = prices.shape[0]
  num_of_expls = prices.shape[1]
  modified_seq_length = int(orig_sequence_length/input_size)

  prices_cut = prices[:input_size*modified_seq_length,:].T
  X_data = prices_cut.reshape((-1, modified_seq_length, input_size))

  # global model, criterion, test_loader

  train_loader, test_loader = construct_loaders(X_data,
                                                cross_validation_cur_round,
                                                cross_validation_total_rounds)
  model, criterion, optimizer, time_arr, training_loss_arr, prediction_loss_arr = define_and_train_model(input_size, modified_seq_length,
                            encoded_size, hidden_size, learning_rate,
                            train_loader, num_epochs, grad_clip)
  return time_arr, training_loss_arr, prediction_loss_arr

num_epochs = 1
# num_epochs = 3000
# Here
time_arr, training_loss_arr, prediction_loss_arr = run_from_scratch(num_epochs,
                                                                    learning_rate, grad_clip, hidden_size,
                     encoded_size, input_size, cross_validation_cur_round=None,
                     cross_validation_total_rounds=None)


plt.plot(time_arr,training_loss_arr,label=r'training loss')

plt.legend()
plt.title('trainging loss vs. time')
plt.xlabel('epochs')
plt.ylabel('training loss')

plt.show()
plt.close()


plt.plot(time_arr,prediction_loss_arr,label=r'prediction loss')

plt.legend()
plt.title('prediction loss vs. time')
plt.xlabel('epochs')
plt.ylabel('prediction loss')

plt.show()
plt.close()
