from utils.data_utils import python_stratified_split
import numpy as np
import torch 

def recall_at_k(user_ratings, embeddings, k=10):
    hits = 0
    total = 0
    
    for user_id, pos_movies, neg_movies in user_ratings:   
        scores = embeddings[user_id][pos_movies+neg_movies]
        # len(pos_movies) = 3
        # len(neg_movies) = 3
        # k = 3
        # scores = [2, 1, 5, 3, 4, 2]
        # indices = [2, 5, 4, 0, 3, 1]
    

        curr_k = min(k, len(scores))
        _, indices = torch.topk(scores, curr_k)
        print(indices)
        print(indices < len(pos_movies))
        hits += torch.sum(indices < len(pos_movies)).item()
        total += len(pos_movies)

        print("no. correct:", torch.sum(indices < k).item())
        print("total positive:", len(pos_movies))
        
    return hits / total

def precision_at_k(user_ratings, embeddings, k=10):
    hits = 0
    total = 0
    
    for user_id, pos_movies, neg_movies in user_ratings:
        scores = embeddings[user_id][pos_movies+neg_movies]
        
        curr_k = min(k, len(scores))
        _, indices = torch.topk(scores, curr_k)
        hits += torch.sum(indices < k).item()
        total += k
        
    return hits / total


def mask_datasets(device, train_ratings, test_ratings, val_ratio=0.1):
    col_user = 'userid'
    col_item = 'itemid'
    col_timestamp = 'timestamp'
    train_ratings, val_ratings = python_stratified_split(train_ratings, ratio=1-val_ratio, col_user=col_user, col_item=col_item, col_timestamp=col_timestamp)
    
    num_users = train_ratings[col_user].max() + 1
    num_items = train_ratings[col_item].max() + 1
    
    # Create rating and mask matrices for train, val and test
    rating_matrix_train = np.zeros(shape=(num_users, num_items))
    mask_matrix_train = np.zeros(shape=(num_users, num_items))
    rating_matrix_val = np.zeros(shape=(num_users, num_items))
    mask_matrix_val = np.zeros(shape=(num_users, num_items))
    rating_matrix_test = np.zeros(shape=(num_users, num_items))
    mask_matrix_test = np.zeros(shape=(num_users, num_items))
    
    for _, r in train_ratings.iterrows():
        rating_matrix_train[int(int(r.iloc[0])), int(int(r.iloc[1]))] = int(r.iloc[2])
        mask_matrix_train[int(r.iloc[0]), int(r.iloc[1])] = 1
    
    for _, r in val_ratings.iterrows():
        rating_matrix_val[int(r.iloc[0]), int(r.iloc[1])] = int(r.iloc[2])
        mask_matrix_val[int(r.iloc[0]), int(r.iloc[1])] = 1
    
    for _, r in test_ratings.iterrows():
        rating_matrix_test[int(r.iloc[0]), int(r.iloc[1])] = int(r.iloc[2])
        mask_matrix_test[int(r.iloc[0]), int(r.iloc[1])] = 1
    
    rating_matrix_train = torch.tensor(rating_matrix_train).to(device)
    mask_matrix_train = torch.tensor(mask_matrix_train).to(device)
    rating_matrix_val = torch.tensor(rating_matrix_val).to(device)
    mask_matrix_val = torch.tensor(mask_matrix_val).to(device)
    rating_matrix_test = torch.tensor(rating_matrix_test).to(device)
    mask_matrix_test = torch.tensor(mask_matrix_test).to(device)
    
    return rating_matrix_train, mask_matrix_train, rating_matrix_val, mask_matrix_val, rating_matrix_test, mask_matrix_test

def loss_function(pred_ratings, true_ratings, mask, model, alpha, beta):
    # Compute the squared error only on observed entries
    diff = mask * (true_ratings - pred_ratings)
    mse_loss = (diff ** 2).sum()
    
    # Regularization terms
    reg_loss = alpha * torch.norm(model.user_factors, p=2) ** 2 + \
               beta * torch.norm(model.item_factors, p=2) ** 2
    
    # Total loss
    total_loss = mse_loss + reg_loss
    return total_loss

