from torch import nn

# Define the MLP model with two hidden layers
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(384, 256) # input size is 28*28 pixels
        self.fc2 = nn.Linear(256, 128) # first hidden layer size is 256 units
        self.fc3 = nn.Linear(128, 10) # second hidden layer size is 128 units
        self.relu = nn.ReLU() # activation function is ReLU

    def forward(self, x):
        x = x.view(-1, 384) # flatten the input image
        x = self.relu(self.fc1(x)) # apply first linear layer and activation
        x = self.relu(self.fc2(x)) # apply second linear layer and activation
        x = self.fc3(x) # apply third linear layer (no activation)
        return x
