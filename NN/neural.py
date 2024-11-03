import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn
import torch.optim as optim 

from utilities.load_dataset import load_tnt

from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader, TensorDataset


import time

# 80-20 train test split 


datastic = load_tnt(filepath='./data/KDDTrain+.txt')
batch_size = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


X_train = datastic['Xtrain']
X_val = datastic['Xval']
Y_train = (datastic['Ytrain'].T).flatten()
Y_val = (datastic['Yval']).flatten()


tensor_x_train = torch.tensor(X_train, dtype=torch.float32).to(device)
tensor_y_train = torch.tensor(Y_train, dtype=torch.int64).to(device)

tensor_x_cv = torch.tensor(X_val, dtype=torch.float32).to(device)
tensor_y_cv = torch.tensor(Y_val, dtype=torch.int64).to(device)

dataset = TensorDataset(tensor_x_train, tensor_y_train)
gd_batches = DataLoader(dataset, batch_size=batch_size, shuffle=True)



    


class Classifier(nn.Module):
    def __init__(self, input_size=123, hidden_size=256, num_classes=23):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, num_classes)

        self.dropoutL1 = nn.Dropout(0.2)
        self.dropoutL2 = nn.Dropout(0.4)
        self.dropoutL3 = nn.Dropout(0.2)
        
    def forward(self, x, predict=False):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        if(predict):
            return nn.Softmax(x)
        return x
    
model = Classifier().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 40  

for epoch in range(num_epochs):
    model.train()
    for features, labels in gd_batches:
        outputs = model(features)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for features, label in zip(tensor_x_cv, tensor_y_cv):
        outputs = model(features)
        _, predicted = torch.max(outputs, 0)
        total += 1
        correct += (predicted == label).item()
    
    accuracy = (100 * correct )/total
    print(f'Validation Accuracy: {accuracy:.2f}%')