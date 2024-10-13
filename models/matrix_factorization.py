import torch.nn as nn
import torch

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, latent_dim, rating_matrix):
        super(MatrixFactorization, self).__init__()
        # User and item latent factors
        self.user_factors = nn.Parameter(torch.randn(num_users, latent_dim) * 0.01)
        self.item_factors = nn.Parameter(torch.randn(num_items, latent_dim) * 0.01)
        self.embeddings = None
        
        # User and item biases
        self.user_bias = nn.Parameter(torch.zeros(num_users, 1))
        self.item_bias = nn.Parameter(torch.zeros(1, num_items))
        
        # Global average rating
        self.global_bias = nn.Parameter(torch.tensor([rating_matrix[rating_matrix != 0].mean()]))
    
    def forward(self):
        # Compute the predicted rating matrix
        interaction = torch.matmul(self.user_factors, self.item_factors.t())
        pred_ratings = interaction + self.user_bias + self.item_bias + self.global_bias
        
        self.embeddings = pred_ratings
        return pred_ratings