import math
import pandas as pd


class MetronAtK(object):
    def __init__(self, top_k):
        self._top_k = top_k
        self._subjects = None  # Subjects which we ran evaluation on
        self._test = None  # The golden set

    @property
    def top_k(self):
        return self._top_k

    @top_k.setter
    def top_k(self, top_k):
        self._top_k = top_k

    @property
    def subjects(self):
        return self._subjects

    @subjects.setter
    def subjects(self, subjects):
        """
        args:
            subjects: list, [test_users, test_items, test_scores, negative users, negative items, negative scores]
        """
        assert isinstance(subjects, list)
        test_users, test_items, test_scores, test_realscore = subjects[0], subjects[1], subjects[2], subjects[3]
        neg_users, neg_items, neg_scores = subjects[4], subjects[5], subjects[6]

        print(f"Length of test_users: {len(test_users)}")
        print(f"Length of test_items: {len(test_items)}")
        print(f"Length of test_preds: {len(test_scores)}")
        # the golden set
        test = pd.DataFrame({'user': test_users,
                             'test_item': test_items,
                             'test_score': test_scores,
                             'real_score': test_realscore})
        # the full set, without real scores
        full = pd.DataFrame({'user': neg_users + test_users,
                            'item': neg_items + test_items,
                            'score': neg_scores + test_scores
                            })
        full = pd.merge(full, test, on=['user'], how='left')
        # rank the items according to the scores for each user
        full['rank'] = full.groupby('user')['score'].rank(method='first', ascending=False)
        full.sort_values(['user', 'rank'], inplace=True)
        self._test = test
        self._subjects = full

    def cal_hit_ratio(self):
        """Hit Ratio @ top_K"""
        full, top_k = self._subjects, self._top_k
        top_k = full[full['rank']<=top_k]
        test_in_top_k = top_k[top_k['test_item'] == top_k['item']]  # golden items hit in the top_K items
        return len(test_in_top_k) * 1.0 / full['user'].nunique()

    def cal_ndcg(self):
        full, top_k = self._subjects, self._top_k
        top_k = full[full['rank']<=top_k]
        test_in_top_k = top_k[top_k['test_item'] == top_k['item']].copy()
        test_in_top_k.loc[:, 'ndcg'] = test_in_top_k['rank'].apply(lambda x: math.log(2) / math.log(1 + x)) # the rank starts from 1
        return test_in_top_k['ndcg'].sum() * 1.0 / full['user'].nunique()
    
    def cal_precision(self):
        """Precision @ top_K
        Implement Precision @ top_k metrics, which is the proportion of the true positive
        Precision is defined as the number of relevant items found in the top K items divided by the total number of items in the top K
        Only do precision on the self._test set
        """
        top_k = self._top_k
        test = self._test.copy()
        test['rank'] = test.groupby('user')['test_score'].rank(method='first', ascending=False)
        test['relevant'] = test['real_score'] >= 5  
        users = test['user'].unique()
        precisions = []
        for user in users:
            user_data = test[test['user'] == user]
            top_k_items = user_data[user_data['rank'] <= top_k]
            num_relevant_in_top_k = top_k_items['relevant'].sum()
            total_items_in_top_k = min(top_k, user_data.shape[0])
            precision = num_relevant_in_top_k / total_items_in_top_k
            precisions.append(precision)
        return sum(precisions) / len(precisions)

    def cal_recall(self):
        """Recall @ top_K 
        Implement Recall @ top_k metrics, which is the proportion of the true positive
        Recall is defined as the number of relevant items found in the top K items divided by the total number of relevant items
        Only do recall on the self._test set
        """
        top_k = self._top_k
        test = self._test.copy()
        test['rank'] = test.groupby('user')['test_score'].rank(method='first', ascending=False)
        test['relevant'] = test['real_score'] >= 5  
        users = test['user'].unique()
        recalls = []
        for user in users:
            user_data = test[test['user'] == user]
            num_relevant = user_data['relevant'].sum()
            if num_relevant == 0:
                continue  # Skip users with no relevant items
            top_k_items = user_data[user_data['rank'] <= top_k]
            num_relevant_in_top_k = top_k_items['relevant'].sum()
            recall = num_relevant_in_top_k / num_relevant
            recalls.append(recall)
        return sum(recalls) / len(recalls)

        
        
