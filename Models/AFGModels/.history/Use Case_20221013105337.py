from asyncio import as_completed
from concurrent.futures import thread
import multiprocessing
import os
import copy
import time
import pickle
from unittest import result
import torch 
import random
import networkx as nx
import numpy as np
import argparse
import read_graph
from model import CUSTOM_RNN_EDGE, CUSTOM_RNN_NODE


def encode_M_matrix(adj_graph,M):
    M_matrix = np.zeros((adj_graph.shape[0]-1,M))
    for i in range(1,adj_graph.shape[0]):
        reverse_indexes= list(range(i-1,max(0,i-M)-1,-1))
        M_matrix[i-1,0:len(reverse_indexes)] = adj_graph[i,reverse_indexes]

    return M_matrix

def loadModel(opt):
    data_file = opt.data
    epochs = opt.epochs
    num_graphs_to_be_generated = 1
    model_save_path = "./models/"
    file_name = data_file.split("/")[-1].split(".")[0]
    dir_to_saveMolel = os.path.join(model_save_path, file_name)
    model_parameters = pickle.load(open(dir_to_saveMolel +'/' +  file_name + '_' + str(epochs) + "_parameters_" + 'bfs_' + str(opt.use_bfs) + '.pkl', "rb"))
    
    M = model_parameters['M']
    hidden_size_node_rnn = model_parameters['hidden_size_node_rnn']
    hidden_size_edge_rnn = model_parameters['hidden_size_edge_rnn']
    embedding_size_node_rnn = model_parameters['embedding_size_node_rnn']
    embedding_size_edge_rnn = model_parameters['embedding_size_edge_rnn']
    num_layers = model_parameters['num_layers']
    len_node_labels = model_parameters['len_nodes']
    len_edge_labels = model_parameters['len_edges']
    node_label_dict = model_parameters['node_label_dict']
    edge_label_dict = model_parameters['edge_label_dict']
    node_label_dict = {value:key for key,value in node_label_dict.items()}
    edge_label_dict = {value:key for key,value in edge_label_dict.items()}
    node_rnn = CUSTOM_RNN_NODE(input_size=M, embedding_size=embedding_size_node_rnn,
                    hidden_size=hidden_size_node_rnn, number_layers=num_layers,output_size=hidden_size_edge_rnn,
                name="node",len_unique_node_labels=len_node_labels,len_unique_edge_labels=len_edge_labels)
    edge_rnn = CUSTOM_RNN_EDGE(input_size=1, embedding_size=embedding_size_edge_rnn,
                    hidden_size=hidden_size_edge_rnn, number_layers=num_layers, output_size=len_edge_labels,
                        name="edge",len_unique_edge_labels=len_edge_labels)
    
    fname_node = dir_to_saveMolel +'/' + file_name + '_' + str(epochs) + '_node_' + 'bfs_' + str(opt.use_bfs) +  '.dat'
    fname_edge= dir_to_saveMolel +'/' + file_name + '_' + str(epochs) + '_edge_' + 'bfs_' + str(opt.use_bfs) +  '.dat'
    node_rnn.load_state_dict(torch.load(fname_node))
    edge_rnn.load_state_dict(torch.load(fname_edge))
    node_rnn.hidden_n = node_rnn.init_hidden(num_graphs_to_be_generated)
    return node_rnn, edge_rnn, model_parameters

def PredictNodeEdge(opt, graph, model_parameters, node_rnn, edge_rnn):
    node_rnn.eval() 
    edge_rnn.eval()

    graph = graph
    use_bfs = opt.use_bfs
    adj_graph = [nx.to_numpy_matrix(graph)] 
    if use_bfs == 'True':
        bfs_start_node = np.random.randint(0,graph.number_of_nodes()) 
        bfs_seq = list(nx.bfs_tree(graph,bfs_start_node))
    elif use_bfs == 'False':
        bfs_seq = range(len(graph.nodes)) 

    M = model_parameters['M']
    num_layers = model_parameters['num_layers']
    max_num_nodes = model_parameters['max_num_nodes']
    most_frequent_edge_label = model_parameters['most_frequent_edge_label']
    adj_graph = adj_graph[0][np.ix_(bfs_seq,bfs_seq)]  #### Changing the adjacency matrix in bfs sequence
    M_matrix = encode_M_matrix(adj_graph,M) # [graph.number_of_nodes()-1, max_prev_node(M)]
    X = np.zeros((graph.number_of_nodes(),M))
    Y = np.zeros((graph.number_of_nodes(),M))
    X[0,:] = np.ones((1,X.shape[1]))
    X[1:M_matrix.shape[0]+1,:] = M_matrix
    Y[0:M_matrix.shape[0],:] = M_matrix
    node_labels = np.zeros((graph.number_of_nodes()))
    node_labels[0:len(bfs_seq)] = [graph.nodes[bfs_seq[i]]['node_label'] for i in range(len(bfs_seq))]

    x = torch.tensor(X, dtype=torch.long)
    y = torch.tensor(Y, dtype=torch.long)
    node_labels = torch.tensor(node_labels, dtype=torch.long)
    x[:,0] = x[:,0]*most_frequent_edge_label
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    node_labels = node_labels.unsqueeze(0)
    h,h_mlp = node_rnn(x,node_labels, seq_lengths=[graph.number_of_nodes()],is_MLP=True) 
    h_ce = h_mlp.view(-1,h_mlp.size(2)) # 节点级预测结果
    h = pack_padded_sequence(h,[graph.number_of_nodes()],batch_first=True,enforce_sorted=False).data
    # y_node_labels = node_labels
    # y_node_labels = y_node_labels.reshape(-1)

    h_edge_tmp = torch.zeros(num_layers-1, h.size(0), h.size(1))
    edge_rnn.hidden_n = torch.cat((h.view(1,h.size(0),h.size(1)),h_edge_tmp),dim=0) # 如原作一样，h作为隐藏层输入（先构建成可输入形式）
    y_packed = pack_padded_sequence(y,[graph.number_of_nodes()],batch_first=True,enforce_sorted=False).data
    edge_rnn_y = y_packed.view(y_packed.size(0),y_packed.size(1),1) # [sum(data[2]), max_prev_node, 1]
    edge_rnn_x = torch.cat((torch.ones(edge_rnn_y.size(0),1,1).long()*most_frequent_edge_label,edge_rnn_y[:,0:-1,0:1]),dim=1) # 格式同edge_rnn_y，但是要下移一行

    # edge_rnn_y = edge_rnn_y.reshape(-1)
    edge_rnn_y_pred = edge_rnn(edge_rnn_x)
    edge_rnn_y_pred = edge_rnn_y_pred.view(-1,edge_rnn_y_pred.size(2))
    # torch.argmax(edge_rnn_y_pred, 1).tolist()[edge_rnn_y_pred.size(0)-M:]

    return np.array(torch.argmax(h_ce, 1))[-1], np.array(torch.argmax(edge_rnn_y_pred, 1))[edge_rnn_y_pred.size(0)-M:].tolist()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='./data/Graphs_example.txt')  
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--use_bfs", type=str, default='False')
    opt = parser.parse_args()
    data_file = opt.data

    node_rnn, edge_rnn, model_parameters = loadModel(opt)

    graphs_ASGMulti = read_graph.read_graphs_in_networkx(data_file, True, 1000000) # 用于计算相似度
    graphsMulti = read_graph.read_graphs_in_networkx2(data_file, True, 1000000) # 用于生成模型

    
    for index in range(len(graphs_ASGMulti)):
        graphs = graphsMulti[index]
        M = model_parameters['M']

        nodeNumOri, edgeNumOri = graphs.number_of_nodes(), graphs.number_of_edges()

        for i in range(nodeNumOri, edgeNumOri + M ): 
            nodeType, edgeRelation = PredictNodeEdge(opt, graphs, model_parameters, node_rnn, edge_rnn)
            if nodeType != 0 : 
                graphs.add_node(i, node_label=nodeType) 
                for j in range(max(0, i-M), i): 
                    if edgeRelation[j - max(0, i-M)] ==0 : continue
                    graphs.add_edge(j, i, weight=edgeRelation[j - max(0, i-M)])
                if len(graphs.edges(i)) == 0 : continue 
            else:
                break

        

        
