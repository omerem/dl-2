import numpy as np
import sys
import matplotlib.pyplot as plt
from models import LSTM_autoencoder 


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Question 3.1.1

input_size = 5
sequence_length = 10


location = './'
rand_data = np.genfromtxt(location+'synthetic data.csv', delimiter=',') 

arg = None

if len(sys.argv)>1:
    arg = int(sys.argv[1])

if arg == 1:
    fig, ax = plt.subplots()
    X1=rand_data[0]
    X2=rand_data[1]
    X3=rand_data[2]
    x_axis = np.arange(1,X1.size+1)
    ax.plot(x_axis,X1, label=r'signal 1')
    ax.plot(x_axis,X2, label=r'signal 2')
    ax.plot(x_axis,X3, label=r'signal 3')
    plt.xlabel(r'$t$ = time')
    plt.ylabel("signal")
    plt.legend()
    plt.title("Examples for the random data")

    location = './figs/'
    plt.savefig(location+'rand data example plot.png')
    plt.show()
    plt.close() 



# Question 3.1.2


X_rand_data = rand_data.reshape((-1, 10, 5))


data_size = X_rand_data.shape[0]

X_train = X_rand_data[:int(0.6*data_size)]
X_validation = X_rand_data[int(0.6*data_size):int(0.8*data_size)]
X_test = X_rand_data[int(0.8*data_size):]

# adjusting data shape

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],5)
X_validation = X_validation.reshape(X_validation.shape[0],X_validation.shape[1],5)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],5)



batch_size = 32

X_train_tens = torch.from_numpy(X_train)
X_validation_tens = torch.from_numpy(X_validation)
X_test_tens = torch.from_numpy(X_test)


train_dataset = X_train_tens
test_dataset = X_test_tens
validation_dataset = X_test_tens

train_dataset = train_dataset.to(device).float()
test_dataset = test_dataset.to(device).float()
validation_dataset = validation_dataset.to(device).float()

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)


validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

def calc_error(model, loader, criterion):
  total_loss = 0
  for _, stocks in enumerate(loader):
    outputs = model(stocks.to(device).float())
    loss = criterion(outputs, stocks.to(device).float())
    total_loss += loss.detach().cpu().numpy()
  return total_loss/len(loader)

def train_model(input_size, sequence_length, encoded_size, hidden_size,
              learning_rate, grad_clip, num_epochs):
  model = LSTM_autoencoder(input_size,
                          sequence_length,
                          encoded_size,
                          hidden_size,
                          num_layers=1,
                          ).to(device)

                          
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  # Train the model
  n_total_steps = len(train_loader)
  global signals, outputs
  for epoch in range(num_epochs):
      for i, signals in enumerate(train_loader):
          # Forward pass
          outputs = model(signals)
          loss = criterion(outputs, signals)
          
          # Backward and optimize
          optimizer.zero_grad()
          loss.backward()
          if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

          optimizer.step()

          if i == n_total_steps-1:
              print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item()}')
  return model, criterion

num_epochs = 50

learning_rate_options = [0.01, 0.001]
grad_clip_options = [1, 8, 14]
hidden_size_options = [64, 128, 256]


total_num_of_options = (len(learning_rate_options)
* len(grad_clip_options)
* len(hidden_size_options))

best_loss = float('inf')

best_learning_rate = learning_rate_options[0]
best_grad_clip = grad_clip_options[0]
best_hidden_size = hidden_size_options[0]

for learning_rate in learning_rate_options:
  for grad_clip in grad_clip_options:
    for hidden_size in hidden_size_options:
      encoded_size = hidden_size

      model, criterion = train_model(input_size, sequence_length, encoded_size, hidden_size,
              learning_rate, grad_clip, num_epochs)
      cur_loss = calc_error(model, validation_loader, criterion)
      if cur_loss<best_loss:
        best_loss = cur_loss
        best_learning_rate = learning_rate
        best_grad_clip = grad_clip
        best_hidden_size = hidden_size

# best_loss = cur_loss
learning_rate = best_learning_rate
grad_clip = best_grad_clip
hidden_size = best_hidden_size
encoded_size = hidden_size

print('best parameters:\n loss= {}\n learning rate= {}, gradient clipping= {}, hidden size= {}'.
      format(best_loss, learning_rate, grad_clip, hidden_size))

# learning_rate = 0.001
# grad_clip = 14
# hidden_size = 256
# encoded_size = 256

num_epochs = 1000

model, _ = train_model(input_size, sequence_length, encoded_size, hidden_size,
              learning_rate, grad_clip, num_epochs)

signal1 = test_dataset[0,:,:].reshape(1,sequence_length,-1)
signal2 = test_dataset[1,:,:].reshape(1,sequence_length,-1)

pred1 = model(signal1)
pred2 = model(signal2)

x_axis = np.arange(1,(sequence_length*input_size)+1)

plt.plot(x_axis,signal1.cpu().numpy().reshape((sequence_length*input_size)),label=r'signal 1')
plt.plot(x_axis,signal2.cpu().numpy().reshape((sequence_length*input_size)),label=r'signal 2')


plt.plot(x_axis,pred1.cpu().detach().numpy().reshape((sequence_length*input_size)),label=r'pred 1')
plt.plot(x_axis,pred2.cpu().detach().numpy().reshape((sequence_length*input_size)),label=r'pred 2')


plt.title("Examples for reconstructed signals")
plt.xlabel(r'$t$ = time')
plt.ylabel("signal")
plt.legend()
plt.show()
plt.close()