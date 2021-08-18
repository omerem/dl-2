from models import LSTM_autoencoder, LSTM_autoencoder_with_classification
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Question 3.2.1


# Hyper-parameters 
# input_size = 784 # 28x28
num_classes = 10
num_epochs = 3
batch_size = 100
learning_rate = 0.001

input_size = 28
sequence_length = 28
hidden_size = 128
encoded_size = 128
num_layers = 2

# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# Creating the model
model = LSTM_autoencoder(input_size, sequence_length, encoded_size, hidden_size, num_layers).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # origin shape: [N, 1, 28, 28]
        # resized: [N, 28, 28]
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, images)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

with torch.no_grad():
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        print(images.size())
        outputs = model(images)
#         print(outputs.size())
        loss = criterion(outputs, images)
        
        for i in range(6):
            plt.subplot(2,3,i+1)
            plt.imshow(images[i].cpu(), cmap='gray')
            if i == 1:
                plt.title('mnist examples - original')
        plt.show()
        
        for i in range(6):
            plt.subplot(2,3,i+1)
            plt.imshow(outputs[i].cpu(), cmap='gray')
            if i == 1:
                plt.title('mnist examples - reconstruction')
        plt.show()
        
        break
plt.close()

# Question 3.2.2

# Creating the model
model = LSTM_autoencoder_with_classification(input_size, sequence_length, encoded_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion_for_encoderdecoder = nn.MSELoss()
criterion_for_classifier = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

def calc_accuracy(loader, model):
  with torch.no_grad():
      n_correct = 0
      n_samples = 0
      for images, labels in loader:
          images = images.reshape(-1, sequence_length, input_size).to(device)
          labels = labels.to(device)
          # print(images.size())
          outputs, scores = model(images)
  #         print(outputs.size())
          loss = 10*criterion_for_encoderdecoder(outputs, images) + criterion_for_classifier(scores, labels) #(#need to convert labels to one-hot)

          
          _, predicted = torch.max(scores.data, 1)

          n_samples += labels.size(0)
          n_correct += (predicted == labels).sum().item()

          # acc = 100.0 * n_correct / n_samples
          # print(f'Accuracy of the network on the 10000 test images: {acc} %')
  return loss, n_correct/n_samples
          

# Train the model

loss_arr = []
acc_arr = []
time_arr = []

# n_train_samples = len(train_dataset)
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
  loss_of_epoch = 0
  for i, (images, labels) in enumerate(train_loader):  
    # origin shape: [N, 1, 28, 28]
    # resized: [N, 28, 28]
    images = images.reshape(-1, sequence_length, input_size).to(device)
    labels = labels.to(device)

    # Forward pass
    outputs, scores = model(images)
    loss = 10*criterion_for_encoderdecoder(outputs, images) + criterion_for_classifier(scores, labels) #(#need to convert labels to one-hot)
    #         print(scores.size())
    #         loss += criterion_for_classifier(scores, labels)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    if (i+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
    loss_of_epoch += loss.item()
  # loss_of_epoch = loss_of_epoch/len(train_loader)
  # _, scores = model(train_dataset.data.to('cuda'))
  # _, predicted = torch.max(scores.data, 1)  
  # n_correct = (predicted == labels).sum().item()
  # accuracy_of_epoch = n_train_samples/n_correct

  loss_of_epoch, accuracy_of_epoch = calc_accuracy(train_loader, model)
  loss_arr.append(loss_of_epoch)
  acc_arr.append(accuracy_of_epoch)
  time_arr.append(epoch+1)

# loss_of_epoch, accuracy_of_epoch = calc_accuracy(train_loader)

acc, loss = calc_accuracy(test_loader, model)

print(f'Accuracy of the network on the 10000 mnist test images: {acc} %')
print(f'Test loss of the network on the 10000 mnist test images: {loss} %')


plt.plot(time_arr,loss_arr,label=r'loss')

plt.legend()
plt.title('loss vs. epochs (mnist dataset)')
plt.xlabel('epochs')
plt.ylabel('loss')

plt.show()
plt.close()

plt.plot(time_arr,acc_arr,label=r'accuracy')

plt.legend()
plt.title('accuracy vs. epochs (mnist dataset)')
plt.xlabel('epochs')
plt.ylabel('accuracy rate')

plt.show()
plt.close()