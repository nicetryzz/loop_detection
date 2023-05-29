from torch import nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(1024, 512) 
        self.fc2 = nn.Linear(512, 256) 
        self.fc3 = nn.Linear(256, 128) 
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 1024)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
