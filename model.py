import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        self.lin_f = nn.Linear(input_dim+hidden_dim, hidden_dim) # this outputs to what C_t-1 is
        self.lin_i = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.lin_c = nn.Linear(input_dim + hidden_dim,hidden_dim)
        self.lin_o = nn.Linear(input_dim + hidden_dim, hidden_dim)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self,x,state):
        h,c, = state # unpacking a tuple
        xh = torch.cat([x,h], dim=1) # concatenating x and h
        f = self.sigmoid(self.lin_f(xh)) # input flows through the linear layers, and then passes through a sigmoid function
        # f is the values we are forgetting
        # after this line we get a tensor of weights
        # Update path
        i = self.sigmoid(self.lin_i(xh))
        ct = self.tanh(self.lin_c(xh))
        ot = self.sigmoid(self.lin_o(xh)) # output, its the weights that say what are we multiplying our cell state by
        c = f * c + i * ct

        h = ot * self.tanh(c)  # this is the new tanh


        return h, (h,c)



        # previous cell state is c
        # modifying c in place

        # xh is our input with our previous hidden
        # we need to see what values we are forgetting


        c = f * c + i * ct # we use a * because it preforms the multiplication element wise







class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.lstm = LSTMCell(input_dim, hidden_dim)





    def forward(self,x,hidden): # x refers to the input
        x = self.embedding(x)
        output, state = self.lstm(x, hidden)
        return output, state


    def init_hidden(self):
        return torch.zeros(1,1,self.hidden_dim)


# the embedding layer is basicallyt he list of one+hot encoding


class Decoder(nn.MOdule):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim # we need to access the hidden dim
        self.embedding = nn.Embedding(output_dim, hidden_dim)
        self.lstm = LSTMCell(hidden_dim, hidden_dim) # we want it to output a hidden dim because 
        self.lin_out = nn.Linear(hidden_dim,output_dim)
        self.log_softmax = 


    def init_hidden(self):
        return torch.zeros(1,1,self.hidden_dim)


    def forward(self,x,hidden):
        out = self.embedding(x)
        out = F.relu(out) 
        out, hidden = self.lstm(out, hidden)
        out = self.lin_out(out)
        out = self.log_softmax(out)
        return out, hidden
        


