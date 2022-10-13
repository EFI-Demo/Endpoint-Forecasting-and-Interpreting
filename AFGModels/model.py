import numpy as np
import sys
import sys
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

# np.set_printoptions(threshold=sys.maxsize)


class CUSTOM_RNN_NODE(torch.nn.Module):
    def __init__(self, input_size, embedding_size=64, hidden_size=32,output_size =None,number_layers=4,name="",len_unique_node_labels=None,len_unique_edge_labels=None):
        super(CUSTOM_RNN_NODE, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.number_layers = number_layers
        self.name = name
        self.len_unique_node_labels = len_unique_node_labels
        self.len_unique_edge_labels = len_unique_edge_labels
        
        self.sequence_embedding_size = embedding_size*input_size + embedding_size*4
        self.input = nn.Embedding(self.len_unique_edge_labels, embedding_size)
        self.input2 = nn.Embedding(self.len_unique_node_labels, embedding_size*4)
        self.rnn = nn.GRU(input_size=self.sequence_embedding_size,hidden_size = self.hidden_size,
                                num_layers=self.number_layers,bias=True,batch_first=True,dropout=0)
        self.hidden_n = None
        self.out = nn.Sequential(nn.Linear(self.hidden_size,self.sequence_embedding_size),nn.ReLU(),nn.Linear(self.sequence_embedding_size,self.output_size))
        self.relu = nn.ReLU()
        
        ###MLP for loss
        self.Linear = nn.Sequential(nn.ReLU(),nn.Linear(self.output_size,self.len_unique_node_labels))
    def forward(self,input,x_node_label, seq_lengths = None,is_packed=True,is_MLP=False):
        
        input = self.input(input)
        input = self.relu(input)
        input = input.reshape(input.shape[0],input.shape[1],-1)
        input2 = self.input2(x_node_label)
        input_concat =torch.cat((input, input2), 2)
        if is_packed:
            input_concat = pack_padded_sequence(input_concat,seq_lengths,batch_first=True,enforce_sorted=False)
        output,self.hidden_n = self.rnn(input_concat,self.hidden_n)
        
        if is_packed:
            output = pad_packed_sequence(output,batch_first=True)[0]
        output = self.out(output)
        if not is_MLP:
            return output
        
        mlp_output= self.Linear(output)
        return output,mlp_output
        
    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.number_layers, batch_size, self.hidden_size))
    
class CUSTOM_RNN_EDGE(torch.nn.Module):
    def __init__(self, input_size, embedding_size=64, hidden_size=32,output_size =None,number_layers=4,name="",len_unique_edge_labels=None):
        super(CUSTOM_RNN_EDGE, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.number_layers = number_layers
        self.name = name
        self.len_unique_edge_labels = len_unique_edge_labels
        
        self.embedding= nn.Embedding(self.len_unique_edge_labels,embedding_size)
        self.linear = nn.Linear(self.input_size,self.embedding_size)
        self.rnn = nn.GRU(input_size=self.embedding_size,hidden_size = self.hidden_size,
                                num_layers=self.number_layers,bias=True,batch_first=True,dropout=0)
        self.hidden_n = None
        self.out = nn.Sequential(nn.Linear(self.hidden_size,self.embedding_size),nn.ReLU(),nn.Linear(self.embedding_size,self.output_size))
        
        self.relu = nn.ReLU()
        self.Linear_mlp = nn.Sequential(nn.ReLU(),nn.Linear(self.output_size,self.len_unique_edge_labels))
    def forward(self,input, seq_lengths = None,is_mlp=False):
        input = self.embedding(input)
        input = self.relu(input)
        input = input.reshape(input.size(0),input.size(1),-1)
        output,self.hidden_n = self.rnn(input,self.hidden_n)
        output = self.out(output)
        
        return output
