import torch
import torch.nn as nn
import torch.nn.functional as F


# Standard NN with 2 hidden layers of 64 neurons
# Input -> 64 neurons -> 64 neurons -> Output
class StandardModel(nn.Module):
    def __init__(self, input_size=7, num_classes=22):
        super(StandardModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.2)
        self.load_state_dict(torch.load("app/ai_models/StandardModel.pth"))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x



# Simpler NN with just 1 hidden layer of 128 neurons and no dropout
# Input -> 128 neurons -> Output
class SimpleNN(nn.Module):
    def __init__(self, input_size=7, num_classes=22):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.load_state_dict(torch.load("app/ai_models/SimpleModel.pth"))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Deeper NN with 1 hidden layer of 128 and another of 64 neurons
# Input -> 128 -> 64 -> Output
class DeeperNN(nn.Module):
    def __init__(self, input_size=7, num_classes=22):
        super(DeeperNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.load_state_dict(torch.load("app/ai_models/DeepModel.pth"))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Super Deep NN with 4 hidden layers with 256, 128, 64, and 32 respectively
# Input -> 256 -> 128 -> 64 -> 32 -> Output
class SuperDeepNN(nn.Module):
    def __init__(self, input_size=7, num_classes=22):
        super(SuperDeepNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.load_state_dict(torch.load("app/ai_models/SuperDeepModel.pth"))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)
        return x
    

LABEL_MAPPING = {
    0: 'apple', 1: 'banana', 2: 'blackgram', 3: 'chickpea', 
    4: 'coconut', 5: 'coffee', 6: 'cotton', 7: 'grapes', 
    8: 'jute', 9: 'kidneybeans', 10: 'lentil', 11: 'maize', 
    12: 'mango', 13: 'mothbeans', 14: 'mungbean', 15: 'muskmelon', 
    16: 'orange', 17: 'papaya', 18: 'pigeonpeas', 19: 'pomegranate', 
    20: 'rice', 21: 'watermelon'
}