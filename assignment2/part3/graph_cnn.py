import torch.nn as nn
import torch

import torch.nn.functional as F
from torch_geometric.utils import add_self_loops


class MatrixGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(MatrixGraphConvolution, self).__init__()
        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        self.B = nn.Parameter(torch.Tensor(out_features, in_features))

        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.B)

    def make_adjacency_matrix(self, edge_index, num_nodes):
        """
        Creates adjacency matrix from edge index.

        :param edge_index: [source, destination] pairs defining directed edges nodes. dims: [2, num_edges]
        :param num_nodes: number of nodes in the graph.
        :return: adjacency matrix with shape [num_nodes, num_nodes]

        Hint: A[i,j] -> there is an edge from node j to node i
        """
        src = edge_index[0]    # column index
        dst = edge_index[1]
        #adj matrix
        adjacency_matrix = torch.zeros(num_nodes, num_nodes, device=edge_index.device)

        adjacency_matrix[dst, src] = 1.0 
        #return adj matrix

        return adjacency_matrix

    def make_inverted_degree_matrix(self, edge_index, num_nodes):
        """
        Creates inverted degree matrix from edge index.

        :param edge_index: [source, destination] pairs defining directed edges nodes. shape: [2, num_edges]
        :param num_nodes: number of nodes in the graph.
        :return: inverted degree matrix with shape [num_nodes, num_nodes]. Set degree of nodes without an edge to 1.
        """
        dst_nodes = edge_index[1]

        # count how many incoming edges each node has
        degree_vector = torch.bincount(dst_nodes, minlength=n_nodes).float()

        # nodes without neighbours should still behave sensibly,
        
        #so they have degree 1 to avoid division issues
        degree_vector[degree_vector == 0] = 1.0

        # flip the degree values to get the scaling factors for normalization
        inverted_degree_vector = 1.0 / degree_vector

        # embed these inverse-degree scalars onto the diagonal of a matrix

          # this mirrors how D^{-1} is defined in GCN theory..
        inverted_degree_matrix = torch.diag(inverted_degree_vector.to(edge_index.device))

        return inverted_degree_matrix

    def forward(self, x, edge_index):
        """
        Forward propagation for GCNs using efficient matrix multiplication.

        :param x: values of nodes. shape: [num_nodes, num_features]
        :param edge_index: [source, destination] pairs defining directed edges nodes. shape: [2, num_edges]
        :return: activations for the GCN
        """
        A = self.make_adjacency_matrix(edge_index, x.size(0))

        D_inv = self.make_inverted_degree_matrix(edge_index, x.size(0))
       

    
        # apply the GCN normalization first,which is D^{-1} A X
             # this mixes each node with information coming from its neighbours
        propagated = D_inv @ (A @ x)

        # combine neighbour information,left term, with the node's own features
            # both contributions use their own learnable linear maps
        out = propagated @ self.W.t() + x @ self.B.t()

        return out


class MessageGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(MessageGraphConvolution, self).__init__()
        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        self.B = nn.Parameter(torch.Tensor(out_features, in_features))

        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.B)

    @staticmethod
    def message(x, edge_index):
        """
        message step of the message passing algorithm for GCNs.

        :param x: values of nodes. shape: [num_nodes, num_features]
        :param edge_index: [source, destination] pairs defining directed edges nodes. shape: [2, num_edges]
        :return: message vector with shape [num_nodes, num_in_features]. Messages correspond to the old node values.

        Hint: check out torch.Tensor.index_add function
        """
        num_nodes = x.size(0)
        sources, destinations = edge_index  # u → v

        #  messages along edges are feats of source nod
        messages = x[sources]                           # [E, F]

         #  sum messages for each destin node
        aggregated_messages = torch.zeros_like(x)       # [N, F]
        aggregated_messages.index_add_(0, destinations, messages)

        #  compute degree number of incoming edg.
        sum_weight = torch.bincount(destinations, minlength=num_nodes).float()
        sum_weight[sum_weight == 0] = 1.0               # avoiding /0

        #  avg over neighbors
        aggregated_messages = aggregated_messages / sum_weight.unsqueeze(1)
        return aggregated_messages


    def update(self, x, messages):
        """
        update step of the message passing algorithm for GCNs.

        :param x: values of nodes. shape: [num_nodes, num_features]
        :param messages: messages vector with shape [num_nodes, num_in_features]
        :return: updated values of nodes. shape: [num_nodes, num_out_features]
        """
        x = messages @ self.W.t() + x @ self.B.t()
        return x

    def forward(self, x, edge_index):
        #message 
        message = self.message(x, edge_index)
        x = self.update(x, message)
        #return x
        return x


class GraphAttention(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphAttention, self).__init__()
        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        self.a = nn.Parameter(torch.Tensor(out_features * 2))

        nn.init.xavier_uniform_(self.W)
        nn.init.uniform_(self.a, 0, 1)

    def forward(self, x, edge_index, debug=False):
        """
        Forward propagation for GATs.
        Follow the implementation of Graph attention networks (Veličković et al. 2018).

        :param x: values of nodes. shape: [num_nodes, num_features]
        :param edge_index: [source, destination] pairs defining directed edges nodes. shape: [2, num_edges]
        :param debug: used for tests
        :return: updated values of nodes. shape: [num_nodes, num_out_features]
        :return: debug data for tests:
                 messages -> messages vector with shape [num_nodes, num_out_features], i.e. Wh from Veličković et al.
                 edge_weights_numerator -> unnormalized edge weightsm i.e. exp(e_ij) from Veličković et al.
                 softmax_denominator -> per destination softmax normalizer

        Hint: the GAT implementation uses only 1 parameter vector and edge index with self loops
        Hint: It is easier to use/calculate only the numerator of the softmax
              and weight with the denominator at the end.

        Hint: check out torch.Tensor.index_add function
        """
        
        edge_index, _ = add_self_loops(edge_index)

        # unpack which nodes send and receive information on each edge
        sources, destinations = edge_index

        # linear projection of node features (this is the "Wh" part in the paper)
        activations = x @ self.W.t()

        # for every edge j → i, we take the transformed feature of the source node j
        messages = activations[sources]

        # build the input for the attention mechanism: concat(Wh_i, Wh_j)
        # this is how the model decides how important each neighbour is
        attention_inputs = torch.cat(
            [activations[destinations], activations[sources]], dim=1
        )

        # compute the unnormalized attention scores α_ij before softmax
        # we apply LeakyReLU just like in the original GAT formulation
        edge_weights_numerator = torch.exp(
            F.leaky_relu((attention_inputs * self.a).sum(dim=1))
        )

        # scale each source message by its attention weight
        weighted_messages = messages * edge_weights_numerator.unsqueeze(1)

        # accumulate attention weights for the softmax denominator (one value per node)
        softmax_denominator = torch.zeros(x.size(0), device=x.device)
        softmax_denominator.index_add_(0, destinations, edge_weights_numerator)

        # nodes with no incoming edges should not cause division problems
        softmax_denominator[softmax_denominator == 0] = 1.0

        # sum weighted messages from all neighbours of each node
        aggregated_messages = torch.zeros_like(activations)
        aggregated_messages.index_add_(0, destinations, weighted_messages)

        # final attention output: normalize weighted sum by the denominator
        aggregated_messages = aggregated_messages / softmax_denominator.unsqueeze(1)

        return aggregated_messages, {'edge_weights': edge_weights_numerator, 'softmax_weights': softmax_denominator,
                                     'messages': messages}

