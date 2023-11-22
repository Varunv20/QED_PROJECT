import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np

class CNN(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 10, 5)
        self.fc1 = nn.Linear(610094, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, output_size)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.0000001, momentum=0.9)
        

    def forward(self, x, y):
        x = np.transpose(x, (2,0,1))

        x = torch.from_numpy(x).float().to(self.device)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = x.view(1,-1)# flatten all dimensions except batch
       

        x = torch.cat((x, y) ,1)

        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class LossPredictor(nn.Module):

    def __init__(self, cnn_dim, hidden_dim):
        super(LossPredictor, self).__init__()
        self.hidden_dim = hidden_dim

        self.cnn = CNN(cnn_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(1, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.f0 = nn.Linear(100, 1)
        self.f1 = nn.Linear(1000, 100)
        self.f2 = nn.Linear(hidden_dim**2 + 5,1000)
        self.f3 = nn.Linear(hidden_dim**2 + 5,hidden_dim**2 + 5)


        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.0000001, momentum=0.9)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x,y):
        x = torch.from_numpy(x).float().to(self.device)
        
        x = self.cnn(x)
        x, _ = self.lstm(x.view(len(x), 1, -1))
        x = x.view(1, -1)
    
        x = torch.cat((x, y) ,1)
        x = F.relu(self.f3(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f1(x))

        x = self.f0(x)
        return x

class LSTM(nn.Module):

    def __init__(self, cnn_dim, hidden_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim

        self.cnn = CNN(cnn_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        #self.lstm = nn.LSTM(1, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.f1 = nn.Linear((hidden_dim + 3)*hidden_dim, 1000)
        self.f2 = nn.Linear(1000, 100)
        self.f3 = nn.Linear(100, 1)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.0001, momentum=0.9)
        self.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    def forward(self, x, color):
        x = np.transpose(x, (2,0,1))


        x = torch.from_numpy(x).float().to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        color = torch.from_numpy(color).float().to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

      
        x = self.cnn(x)
        

        x = torch.cat((x, color),0)

        x, _ = self.lstm(x.view(len(x), 1, -1))
        x = x.view(1, -1)
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = self.f3(x)
        x = F.tanh(x)
        return x

