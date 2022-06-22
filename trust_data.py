from torch.utils.data import Dataset
from copy import deepcopy
import pandas as pd

class GLREC_UUIRatingTrust(Dataset):
    def __init__(self, data, all_avg, avg_user, avg_item, user_interacted_diff, item_interacted_diff, u_users_similar, u_users_items_diff):
        self.data = data
        self.all_avg = all_avg
        self.avg_user = avg_user
        self.avg_item = avg_item
        self.user_interacted_diff = user_interacted_diff
        self.item_interacted_diff = item_interacted_diff
        self.u_users_similar = u_users_similar
        self.u_users_items_diff = u_users_items_diff

    def __getitem__(self, index):
        uid = self.data[index][0]
        iid = self.data[index][1]
        rating = self.data[index][2]
        label = self.data[index][3]
        u_avg = self._seach_avg(self.avg_user, uid)
        i_avg = self._seach_avg(self.avg_item, iid)
        u_items_diff = self._search_usersitems(self.user_interacted_diff, uid)
        i_users_diff = self._search_usersitems(self.item_interacted_diff, iid)
        u_users_similar = self._search_u_users_similar(self.u_users_similar, uid)
        u_users_items_diff = self._search_u_users_items(self.u_users_items_diff, uid)
        return (uid, iid, rating, label), u_avg, i_avg, u_items_diff, i_users_diff, u_users_similar, u_users_items_diff

    def __len__(self):
        return len(self.data)

    def _seach_avg(self, avg_list, key):
        if key in avg_list.keys():
            return avg_list[key]
        else:
            return self.all_avg

    def _search_usersitems(self, users_items, key):
        if key in users_items.keys():
            return users_items[key]
        else:
            return [[0, 0.0]]

    def _search_u_users_similar(self, u_users_similar, key):
        if key in u_users_similar.keys():
            return u_users_similar[key]
        else:
            return [[0, 0.0]]

    def _search_u_users_items(self, u_users_items, key):
        if key in u_users_items.keys():
            return u_users_items[key]
        else:
            return [[[0, 0.0]]]


class SocialGenerator(object):
    """Construct dataset for GraphRec and GLRec"""
    def __init__(self, singlegenerator, trust):
        """
        args:
            singlegenerator: rating_data.SingleGenerator, which contains ratings, train, test, etc. datasets
            trust: pd.DataFrame, which contains 2 columns = ['uid', 'fid']
        """
        self.singlegenerator = singlegenerator
        self.trust = trust
        self.all_avg = self.singlegenerator.train_ratings['rating'].mean()
        self.avg_item, self.avg_user = self._average_data(self.singlegenerator.train_ratings)
        self.user_interacted_origin, self.user_interacted_diff = self._u_items(self.singlegenerator.train_ratings)
        self.item_interacted_origin, self.item_interacted_diff = self._i_users(self.singlegenerator.train_ratings)
        self.u_users_similar, self.u_users_items_diff, self.u_users_items_origin = self._u_users_similar_freitems()

    def _average_data(self, ratings):
        avg_item = ratings.groupby(['itemID'])['rating'].mean()
        avg_user = ratings.groupby(['userID'])['rating'].mean()
        return avg_item, avg_user

    def _u_items(self, ratings):
        ratings = deepcopy(ratings)
        avg_item = self.avg_item.reset_index().rename(columns={'rating':'avg_rating'})
        ratings = pd.merge(ratings, avg_item, on='itemID')
        ratings['difference'] = ratings.apply(lambda x: round(abs(x['rating'] - x['avg_rating'])), axis=1)
        ratings['interacted_origin'] = ratings.apply(lambda x: [x['itemID'], x['rating']], axis=1)
        ratings['interacted_difference'] = ratings.apply(lambda x: [x['itemID'], x['difference']], axis=1)
        user_items_originrating = ratings.groupby('userID')['interacted_origin'].apply(list)
        user_items_difference = ratings.groupby('userID')['interacted_difference'].apply(list)
        return user_items_originrating, user_items_difference

    def _i_users(self, ratings):
        ratings = deepcopy(ratings)
        avg_user = self.avg_item.reset_index().rename(columns={'rating':'avg_rating'})
        ratings = pd.merge(ratings, avg_user, on='itemID')
        ratings['difference'] = ratings.apply(lambda x: round(abs(x['rating'] - x['avg_rating'])), axis=1)
        ratings['interacted_origin'] = ratings.apply(lambda x: [x['userID'], x['rating']], axis=1)
        ratings['interacted_difference'] = ratings.apply(lambda x: [x['userID'], x['difference']], axis=1)
        item_users_oringrating = ratings.groupby('itemID')['interacted_origin'].apply(list)
        item_users_difference = ratings.groupby('itemID')['interacted_difference'].apply(list)
        return item_users_oringrating, item_users_difference

    def _u_users_similar_freitems(self):
        trust = deepcopy(self.trust)
        user_items = deepcopy(self.user_interacted_origin).reset_index().rename(columns={'interacted_origin':'user_items'})
        trust_user = pd.merge(trust, user_items, on='userID')
        fre_items = deepcopy(self.user_interacted_origin).reset_index().rename(columns={'userID':'freID', 'interacted_origin':'fre_items_origin'})
        trust_u_users = pd.merge(trust_user, fre_items, on='freID')
        trust_u_users['similar'] = trust_u_users.apply(lambda x: self._similar_value(x['user_items'], x['fre_items_origin']), axis=1)
        u_users_items = pd.merge(trust_u_users, self.user_interacted_diff.reset_index().rename(columns={'userID': 'freID', 'interacted_difference':'fre_items_diff'}), on='freID')
        u_users_items.drop('user_items', axis=1, inplace=True)  # |userID|freID|fre_items_origin|similar|fre_items_diff|
        u_users_items['fres_similar'] = u_users_items.apply(lambda x: [x.freID, x.similar], axis=1)  # |userID|freID|fre_items_origin|similar|fre_items_diff|fres_similar|
        u_users_similar = u_users_items.groupby('userID')['fres_similar'].apply(list)
        u_users_items_diff = u_users_items.groupby('userID')['fre_items_diff'].apply(list)
        u_users_items_origin = u_users_items.groupby('userID')['fre_items_origin'].apply(list)
        return u_users_similar, u_users_items_diff, u_users_items_origin

    def _similar_value(self, la, lb):
        aa, bb = dict(la), dict(lb)
        count = 0
        for i in aa.keys():
            if i in bb.keys():
                if abs(aa[i] - bb[i]) <= 1:
                    count = count + 1
        return count

    def instance_GLRec_dataset(self):
        train_data = []
        for row in self.singlegenerator.train_ratings.itertuples():
            train_data.append((int(row.userID), int(row.itemID), float(row.rating), float(row.label)))
        train_dataset = GLREC_UUIRatingTrust(train_data, self.all_avg, self.avg_user, self.avg_item, self.user_interacted_diff,
                                 self.item_interacted_diff, self.u_users_similar, self.u_users_items_diff)
        evaluate_data = []
        for row in self.singlegenerator.evaluate_ratings.itertuples():
            evaluate_data.append((int(row.userID), int(row.itemID), float(row.rating), float(row.label)))
        evaluate_dataset = GLREC_UUIRatingTrust(evaluate_data, self.all_avg, self.avg_user, self.avg_item, self.user_interacted_diff,
                                 self.item_interacted_diff, self.u_users_similar, self.u_users_items_diff)
        test_data = []
        for row in self.singlegenerator.test_ratings.itertuples():
            test_data.append((int(row.userID), int(row.itemID), float(row.rating), float(row.label)))
        test_dataset = GLREC_UUIRatingTrust(test_data, self.all_avg, self.avg_user, self.avg_item, self.user_interacted_diff,
                                 self.item_interacted_diff, self.u_users_similar, self.u_users_items_diff)
        return [train_dataset, evaluate_dataset, test_dataset, self.singlegenerator.user_count, self.singlegenerator.item_count]

