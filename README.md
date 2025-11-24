# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## THEORY


## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

STEP 1:
Load and normalize data, create sequences

STEP 2:
Convert data to tensors and set up DataLoader.

STEP 3:
Define the RNN model architecture.

STEP 4:
Summarize, compile with loss and optimizer.

STEP 5:
Train the model with loss tracking.

STEP 6:
Predict on test data, plot actual vs. predicted prices.

## PROGRAM

### Name: KESAVAN S

### Register Number: 212224230121

```python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
## Step 1: Load and Preprocess Data
# Load training and test datasets
df_train = pd.read_csv('trainset.csv')
df_test = pd.read_csv('testset.csv')
# Use closing prices
train_prices = df_train['Close'].values.reshape(-1, 1)
test_prices = df_test['Close'].values.reshape(-1, 1)
# Normalize the data based on training set only
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_prices)
scaled_test = scaler.transform(test_prices)
# Create sequences
def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

seq_length = 60
x_train, y_train = create_sequences(scaled_train, seq_length)
x_test, y_test = create_sequences(scaled_test, seq_length)
x_train.shape, y_train.shape, x_test.shape, y_test.shape
# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
# Create dataset and dataloader
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
class RNNModel (nn.Module):
  def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
    super (RNNModel, self). __init__()
    self.rnn = nn. RNN(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn. Linear(hidden_size, output_size)
  def forward(self, x):
    out,_=self.rnn(x)
    out=self.fc (out[:,-1,:])
    return out
model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# input_size = (batch_size, seq_len, input_size)
summary(model, input_size=(64, 60, 1))
criterion = nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
## Step 3: Train the Model
def train_model(model, train_loader, criterion, optimizer, epochs=20):
  train_losses = []
  model.train()
  for epoch in range(epochs):
    total_loss = 0
    for x_batch, y_batch in train_loader:
      x_batch,y_batch=x_batch.to(device),y_batch.to(device)
      optimizer.zero_grad()
      outputs=model(x_batch)
      loss=criterion(outputs, y_batch)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
    train_losses.append(total_loss/len(train_loader))
    print (f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/ len(train_loader):.4f}')
  plt.plot(train_losses, label='Training Loss')
  plt.xlabel('Epoch')
  plt.ylabel('MSE Loss')  
  plt.title('Training Loss Over Epochs')
  plt.legend()
  plt.show()
# Plot training loss
print('Name: KESAVAN S')
print('Register Number: 212224230121')
train_model (model, train_loader, criterion, optimizer)
## Step 4: Make Predictions on Test Set
model.eval()
with torch.no_grad():
    predicted = model(x_test_tensor.to(device)).cpu().numpy()
    actual = y_test_tensor.cpu().numpy()

# Inverse transform the predictions and actual values
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(actual)

# Plot the predictions vs actual prices
print('Name:KESAVAN S ')
print('Register Number: 212224230121')
plt.figure(figsize=(10, 6))
plt.plot(actual_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Stock Price Prediction using RNN')
plt.legend()
plt.show()
print(f'Predicted Price: {predicted_prices[-1]}')
print(f'Actual Price: {actual_prices[-1]}')

```

### OUTPUT
<img width="952" height="356" alt="image" src="https://github.com/user-attachments/assets/65741893-8413-4771-bfb9-1fd69df9acc5" />
<img width="310" height="467" alt="image" src="https://github.com/user-attachments/assets/171291b6-b3ab-4388-998f-4808a30f03ee" />
<img width="758" height="553" alt="image" src="https://github.com/user-attachments/assets/3cdc8b86-e2d7-47b9-8332-ff4c3d6763ac" />
<img width="1118" height="776" alt="image" src="https://github.com/user-attachments/assets/d4187278-c1f2-48fe-b3cb-8f295acd8652" />

## RESULT
Include your result here
