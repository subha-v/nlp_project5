import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim


    def forward(self,x,hidden): # x refers to the input
        pass

    def init_hidden(self):
        return torch.zeros(1,1,self.hidden_dim)

# the embedding layer is basicallyt he list of one+hot encoding
