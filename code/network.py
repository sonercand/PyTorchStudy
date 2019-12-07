import torch.nn.functional as F
from torch import nn
from torch import optim


class Network(nn.Module):
    def __init__(self,input_size,output_size,hidden_layers):
        '''
        args:
            input_size int : input dimension
            output_size int : output dimension
            hidden_layers list : hidden layers contains node sizes for the layers between input and output
        '''
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.layers = nn.ModuleList([nn.Linear(self.input_size, self.hidden_layers[0])]) # initiate input layer
        self.set_layers() # add addtional layers including output layer
                         
                         
    def set_layers(self):
        '''
        append hidden layers and output layer
        '''
        queue = self.hidden_layers.copy()
        num_nodes_in = queue.pop(0)
        while queue:
            num_nodes_out = queue.pop(0)
            self.layers.append(nn.Linear(num_nodes_in,num_nodes_out))
            num_nodes_in = num_nodes_out
        self.layers.append(nn.Linear(num_nodes_in,self.output_size))
    
    def forward(self,x):
        '''
        forward prop.
        '''
        x = x.view(x.shape[0], -1)# flatten the image
        
        for k in range(len(self.layers)-1):
            x = F.relu(self.layers[k](x))
        x = F.log_softmax(self.layers[k+1](x),dim=1) # all the layers passes through relue except this last one 
        
        return x

