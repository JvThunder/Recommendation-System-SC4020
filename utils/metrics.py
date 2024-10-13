import torch

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