import torch
import random
import math
import pandas as pd
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset

random.seed(0)


class UserItemRatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""
    def __init__(self, user_tensor, item_tensor, target_tensor):
        """
        args:

            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)


class SampleGenerator(object):
    """Construct dataset for NCF"""

    def __init__(self, ratings):
        """
        args:
            ratings: pd.DataFrame, which contains 4 columns = ['userId', 'itemId', 'rating', 'timestamp']
        """
        assert 'userId' in ratings.columns
        assert 'itemId' in ratings.columns
        assert 'rating' in ratings.columns
        self.original_ratings = ratings.copy()
        self.ratings = ratings
        # explicit feedback using _normalize and implicit using _binarize
        # self.preprocess_ratings = self._normalize(ratings)
        self.preprocess_ratings = self._binarize(ratings)
        self.user_pool = set(self.ratings['userId'].unique())
        self.item_pool = set(self.ratings['itemId'].unique())
        # create negative item samples for NCF learning
        self.negatives = self._sample_negative(ratings)
        self.train_ratings, self.test_ratings = self._split_loo(self.preprocess_ratings)
        self.test_ratings = self._add_real_score(self.test_ratings, self.original_ratings)
        

    def _add_real_score(self, test_ratings, original_ratings):
        """
        Add the real score (original rating) to the test ratings.
        """
        # Merge the test_ratings with the original_ratings on userId and itemId to add the real rating values
        test_ratings = pd.merge(test_ratings, original_ratings[['userId', 'itemId', 'rating']], 
                                on=['userId', 'itemId'], 
                                suffixes=('', '_real'))
        
        # Rename the real rating column to 'real_score'
        test_ratings.rename(columns={'rating_real': 'real_score'}, inplace=True)
        
        return test_ratings

    def _normalize(self, ratings):
        """normalize into [0, 1] from [0, max_rating], explicit feedback"""
        ratings = deepcopy(ratings)
        max_rating = ratings.rating.max()
        ratings['rating'] = ratings.rating * 1.0 / max_rating
        return ratings
    
    def _binarize(self, ratings):
        """binarize into 0 or 1, imlicit feedback"""
        ratings = deepcopy(ratings)
        ratings.loc[ratings['rating'] > 0, 'rating'] = 1.0 # replace ratings['rating'][ratings['rating'] > 0] = 1.0
        return ratings

    def _split_loo(self, ratings):
        # Rank interactions per user by timestamp in descending order
        ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
        
        # Calculate the number of interactions per user
        user_interaction_counts = ratings.groupby('userId').size().reset_index(name='interaction_count')
        
        # Merge interaction counts back into the ratings DataFrame
        ratings = ratings.merge(user_interaction_counts, on='userId')
        
        # Calculate the cutoff rank for test data per user (25% of interactions)
        ratings['cutoff_rank'] = ratings['interaction_count'].apply(lambda x: math.ceil(x * 0.25))
        
        # Mark interactions as test or train based on the cutoff rank
        ratings['is_test'] = ratings['rank_latest'] <= ratings['cutoff_rank']
        
        # Split the data into training and testing sets
        test = ratings[ratings['is_test']]
        train = ratings[~ratings['is_test']]
        
        # Ensure that every user is represented in both sets
        assert train['userId'].nunique() == test['userId'].nunique()
        
        return train[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]
    
    def _sample_negative(self, ratings):
        """return all negative items & 100 sampled negative items"""
        interact_status = ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(
            columns={'itemId': 'interacted_items'})
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
        interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(list(x), 99))
        return interact_status[['userId', 'negative_items', 'negative_samples']]

    def instance_a_train_loader(self, num_negatives, batch_size):
        """instance train loader for one training epoch"""
        users, items, ratings = [], [], []
        train_ratings = pd.merge(self.train_ratings, self.negatives[['userId', 'negative_items']], on='userId')
        train_ratings['negatives'] = train_ratings['negative_items'].apply(lambda x: random.sample(list(x), num_negatives))
        for row in train_ratings.itertuples():
            users.append(int(row.userId))
            items.append(int(row.itemId))
            ratings.append(float(row.rating))
            for i in range(num_negatives):
                users.append(int(row.userId))
                items.append(int(row.negatives[i]))
                ratings.append(float(0))  # negative samples get 0 rating
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(users),
                                        item_tensor=torch.LongTensor(items),
                                        target_tensor=torch.FloatTensor(ratings))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    @property
    def evaluate_data(self):
        """create evaluate data"""
        test_ratings = pd.merge(self.test_ratings, self.negatives[['userId', 'negative_samples']], on='userId')
        test_users, test_items, test_realscore, negative_users, negative_items = [], [], [], [], []
        print(test_ratings.columns)
        for row in test_ratings.itertuples():
            test_users.append(int(row.userId)) 
            test_items.append(int(row.itemId)) 
            test_realscore.append(row.real_score)
            for i in range(len(row.negative_samples)):
                negative_users.append(int(row.userId))
                negative_items.append(int(row.negative_samples[i]))
        
        # Correct print statement
        
        return [torch.LongTensor(test_users), torch.LongTensor(test_items), torch.LongTensor(test_realscore), torch.LongTensor(negative_users),
                torch.LongTensor(negative_items)]

