import torch
import random
import pandas as pd
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset

cut_off = 4

class UIRatingLabel(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""
    def __init__(self, user_tensor, item_tensor, rating_tensor, label_tensor):
        """
        args:
            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.rating_tensor = rating_tensor
        self.label_tensor = label_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.rating_tensor[index], self.label_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)


class SingleGenerator(object):
    """Construct dataset for NCF"""

    def __init__(self, ratings, prob):
        """
        args:
            ratings: pd.DataFrame, which contains 3 columns = ['uid', 'iid', 'rating']
        """
        assert 'userID' in ratings.columns
        assert 'itemID' in ratings.columns
        assert 'rating' in ratings.columns

        self.ratings = ratings
        # explicit feedback using _normalize and implicit using _binarize
        self.preprocess_ratings = self._binarize(ratings)
        self.user_pool = set(self.ratings['userID'].unique())  # 用户池：所有用户id
        self.item_pool = set(self.ratings['itemID'].unique())  # 物品池：所有物品id
        self.user_count, self.item_count = max(self.user_pool), max(self.item_pool)
        self.train_ratings, self.evaluate_ratings, self.test_ratings = self._split_loo(self.preprocess_ratings, prob)

    def __len__(self):
        return len(self.ratings)

    def _binarize(self, ratings):
        """binarize into 0 or 1, imlicit feedback"""
        ratings = deepcopy(ratings)
        ratings['label'] = ratings['rating']
        ratings['label'][ratings['rating'] >= cut_off] = 1.0
        ratings['label'][ratings['rating'] < cut_off] = 0.0
        return ratings

    def _split_loo(self, ratings, prob):
        ratings = deepcopy(ratings)
        ratings = ratings.sample(frac=1.0)
        ratings = ratings.reset_index(drop=True)
        test = ratings[:int(len(ratings)*prob)]
        evaluate = ratings[int(len(ratings)*prob):int(2*len(ratings)*prob)]
        train = ratings[int(2*len(ratings)*prob):]
        return train, evaluate, test

    def instance_general_dataset(self):
        """instance only rating train loader for one training epoch"""
        users, items, ratings, labels = [], [], [], []
        for row in self.train_ratings.itertuples():
            users.append(int(row.userID))
            items.append(int(row.itemID))
            ratings.append(float(row.rating))
            labels.append(float(row.label))
        train_dataset = UIRatingLabel(user_tensor=torch.LongTensor(users),
                                        item_tensor=torch.LongTensor(items),
                                        rating_tensor=torch.FloatTensor(ratings),
                                        label_tensor=torch.FloatTensor(labels))
        users, items, ratings, labels = [], [], [], []
        for row in self.evaluate_ratings.itertuples():
            users.append(int(row.userID))
            items.append(int(row.itemID))
            ratings.append(float(row.rating))
            labels.append(float(row.label))
        evaluate_dataset = UIRatingLabel(user_tensor=torch.LongTensor(users),
                                        item_tensor=torch.LongTensor(items),
                                        rating_tensor=torch.FloatTensor(ratings),
                                        label_tensor=torch.FloatTensor(labels))
        users, items, ratings, labels = [], [], [], []
        for row in self.test_ratings.itertuples():
            users.append(int(row.userID))
            items.append(int(row.itemID))
            ratings.append(float(row.rating))
            labels.append(float(row.label))
        test_dataset = UIRatingLabel(user_tensor=torch.LongTensor(users),
                                        item_tensor=torch.LongTensor(items),
                                        rating_tensor=torch.FloatTensor(ratings),
                                        label_tensor=torch.FloatTensor(labels))
        return [train_dataset, evaluate_dataset, test_dataset, self.user_count, self.item_count]


