import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Q 3.1


# Fully connected neural network with one hidden layer
class LSTM_autoencoder(nn.Module):
    def __init__(self, input_size, sequence_length,
    encoded_size, hidden_size, num_layers):
        super(LSTM_autoencoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
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




class LSTM_autoencoder_with_classification(nn.Module):
    def __init__(self, input_size, sequence_length, encoded_size, hidden_size, num_layers, num_classes):
        super(LSTM_autoencoder_with_classification, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size # This is the size of the hidden (and cell) state of the decoder-LSTM.
        self.sequence_length = sequence_length
        self.classifier_hidden_size = encoded_size # maybe we should change it 
        # -> x needs to be: (batch_size, seq, input_size)
        
        # define the encoder and the decoder
        self.lstm_encoder = nn.LSTM(input_size, encoded_size, num_layers, batch_first=True)
        self.lstm_decoder = nn.LSTM(encoded_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size) # takes a hiden state of the decoder and returns an approximation of the coresponding input to the encoder
        
        # define the classifier
        self.fc_classifier1 = nn.Linear(hidden_size, self.classifier_hidden_size) # The input to the classifier is the last hidden-state of the decoder.
        self.relu = nn.ReLU()
        self.fc_classifier2 = nn.Linear(self.classifier_hidden_size, num_classes)
        
    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) # x.size(0) is the batch size.
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        
        encoded_sequence, _ = self.lstm_encoder(x, (h0,c0))  
        # encoded_sequence: (batch_size, seq_length, hidden_size)        
        encoded_sequence = encoded_sequence[:, -1, :] # Keep only the hidden state from the last time tik.
        # encoded_sequence: (batch_size, hidden_size)
        encoded_sequence = torch.unsqueeze(encoded_sequence, 1) # Add a second dimension to make pytorch treat it as a sequence of length 1.
        # encoded_sequence: (batch_size, 1, hidden_size)
        
        decoded_sequence = []
        hidden_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        cell_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        decoded_element = 0 # it should be approchable to the classifier so we declare it here.
        for i in range(self.sequence_length):
            decoded_element, (hidden_state, cell_state) = self.lstm_decoder(encoded_sequence, (hidden_state,cell_state))
            decoded_element = decoded_element[:, -1, :]
            decoded_sequence.append(torch.unsqueeze(self.fc(decoded_element), 1))
        decoded_sequence = torch.cat(decoded_sequence, 1) # turns a list of tensors to a tensor.
    
        # perform the classification
#         print("hey",decoded_element.size())
        scores = self.fc_classifier1(decoded_element) # decoded element is the hidden-state of the top LSTM layer of the decoder.
        scores = self.relu(scores)
        scores = self.fc_classifier2(scores)
        
        return decoded_sequence, scores
