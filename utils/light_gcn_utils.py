import torch
import torch.nn.functional as F
from collections import defaultdict

def build_user_movie_interactions(ratings_df, user_column_name, item_column_name, rating_column_name):

    user_movie_dict = defaultdict(list)
    for user_id, movie_id, rating in zip(ratings_df[user_column_name], ratings_df[item_column_name], ratings_df[rating_column_name]):
        user_movie_dict[user_id].append((movie_id, rating))
    return user_movie_dict

def get_positive_negative_ratings(user_item_dict, positive_threshold, negative_threshold,):

    user_ratings = []

    for user_id, movies in user_item_dict.items():
        pos_movies = [movie_id for movie_id, rating in movies if rating >= positive_threshold]
        neg_movies = [movie_id for movie_id, rating in movies if rating <= negative_threshold]
        
        if len(pos_movies) == 0 or len(neg_movies) == 0:
            continue
        
        user_ratings.append((user_id, pos_movies, neg_movies))
        
    return user_ratings

def bpr_loss(embeddings, users, pos_items, neg_items):
    user_emb = embeddings[users]
    pos_emb = embeddings[pos_items]
    neg_emb = embeddings[neg_items]
    
    pos_scores = (user_emb * pos_emb).sum(dim=1)
    neg_scores = (user_emb * neg_emb).sum(dim=1)
    
    loss = -F.logsigmoid(pos_scores - neg_scores).mean()
    return loss


def recall_at_k(user_ratings, embeddings, k=10, device='cpu'):
    hits = 0
    total = 0
    
    for user_id, pos_movies, neg_movies in user_ratings:    
        user_emb = embeddings[user_id]
        pos_emb = embeddings[pos_movies]
        neg_emb = embeddings[neg_movies]
        
        pos_scores = (user_emb * pos_emb).sum(dim=1)
        neg_scores = (user_emb * neg_emb).sum(dim=1)
        
        scores = torch.cat([pos_scores, neg_scores])
        if len(scores) < k:
            continue

        curr_k = min(k, len(scores))
        _, indices = torch.topk(scores, curr_k)
        hits += torch.sum(indices < k).item()
        total += len(pos_movies)
        
    return hits / total

def precision_at_k(user_ratings, embeddings, k=10, device='cpu'):
    hits = 0
    total = 0
    
    for user_id, pos_movies, neg_movies in user_ratings:
        user_emb = embeddings[user_id]
        pos_emb = embeddings[pos_movies]
        neg_emb = embeddings[neg_movies]
        
        pos_scores = (user_emb * pos_emb).sum(dim=1)
        neg_scores = (user_emb * neg_emb).sum(dim=1)
        
        scores = torch.cat([pos_scores, neg_scores])
        if len(scores) < k:
            continue
    
        curr_k = min(k, len(scores))
        _, indices = torch.topk(scores, curr_k)
        hits += torch.sum(indices < k).item()
        total += k
        
    return hits / total

def evaluate(model, user_features_tensor, item_features_tensor, edge_index, users, pos_items, neg_items):
    model.eval()
    
    with torch.no_grad():
        user_features = user_features_tensor[users]
        movie_features = item_features_tensor[pos_items]
        embeddings = model(user_features, movie_features, edge_index)
        loss = bpr_loss(embeddings, users, pos_items, neg_items)
    
    return loss.item()

