import torch
from torch import nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree

class LightGCN(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add') 

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)
        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]
        return norm.view(-1, 1) * x_j
    
class LightGCNStack(torch.nn.Module):
    def __init__(self, num_nodes, no_user_features, no_movie_features, embedding_dim, num_layers):
        super().__init__()
        self.users_latent = nn.Linear(no_user_features, embedding_dim)
        self.movies_latent = nn.Linear(no_movie_features, embedding_dim)
        self.convs = torch.nn.ModuleList([LightGCN(embedding_dim, embedding_dim) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, user_feature, movie_feature, edge_index):
        movie_embedding = self.movies_latent(movie_feature)
        user_embedding = self.users_latent(user_feature)
        x = torch.cat([user_embedding, movie_embedding], dim=0)
        all_embeddings = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            all_embeddings.append(x)
        
        # Aggregate embeddings with factors a_k = 1/(k+1)
        out = sum((1.0 / (k + 1)) * emb for k, emb in enumerate(all_embeddings))
        return out
    
