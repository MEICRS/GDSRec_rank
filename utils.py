import torch
import random


"""
Ciao dataset info:
Avg number of items rated per user: 38.3
Avg number of users interacted per user: 16.4
Avg number of users connected per item: 2.7
"""


def collate_fn(batch_data, node_dropout=25):
    """This function will be used to pad the graph to max length in the batch
       It will be used in the Dataloader
    """
    uids, iids, ratings, labels, u_avgs, i_avgs = [], [], [], [], [], []
    u_items, u_users, u_users_items, i_users = [], [], [], []
    u_items_len, u_users_len, i_users_len, u_users_items_len = [], [], [], []

    for data, u_avg_u, i_avg_i, u_items_u, i_users_i, u_users_similar_u, u_users_items_u in batch_data:

        (uid, iid, rating, label) = data
        uids.append(uid)
        iids.append(iid)
        ratings.append(rating)
        labels.append(label)
        u_avgs.append(u_avg_u)
        i_avgs.append(i_avg_i)

        # user-items
        if len(u_items_u) <= node_dropout:
            u_items.append(u_items_u)
        else:
            u_items.append(random.sample(u_items_u, node_dropout))
        u_items_len.append(min(len(u_items_u), node_dropout))

        # user-users and user-users-items
        if len(u_users_similar_u) <= node_dropout:
            u_users.append(u_users_similar_u)
            u_u_items = []
            for uui in u_users_items_u:
                if len(uui) < node_dropout:
                    u_u_items.append(uui)
                else:
                    u_u_items.append(random.sample(uui, node_dropout))
            u_users_items.append(u_u_items)
        else:
            sample_index = random.sample(list(range(len(u_users_similar_u))), node_dropout)
            u_users.append([u_users_similar_u[si] for si in sample_index])

            u_users_items_u_tr = [u_users_items_u[si] for si in sample_index]
            u_u_items = []
            for uui in u_users_items_u_tr:
                if len(uui) < node_dropout:
                    u_u_items.append(uui)
                else:
                    u_u_items.append(random.sample(uui, node_dropout))
            u_users_items.append(u_u_items)

        u_users_len.append(min(len(u_users_similar_u), node_dropout))

        # item-users
        if len(i_users_i) <= node_dropout:
            i_users.append(i_users_i)
        else:
            i_users.append(random.sample(i_users_i, node_dropout))
        i_users_len.append(min(len(i_users_i), node_dropout))

    batch_size = len(batch_data)

    # padding
    u_items_maxlen = max(u_items_len)
    u_users_maxlen = max(u_users_len)
    i_users_maxlen = max(i_users_len)

    u_item_pad = torch.zeros([batch_size, u_items_maxlen, 2], dtype=torch.long)
    for i, uid in enumerate(u_items):
        u_item_pad[i, :len(uid), :] = torch.LongTensor(uid)


    u_user_pad = torch.zeros([batch_size, u_users_maxlen, 2], dtype=torch.long)
    for i, uu in enumerate(u_users):
        u_user_pad[i, :len(uu), :] = torch.LongTensor(uu)

    u_user_item_pad = torch.zeros([batch_size, u_users_maxlen, node_dropout, 2], dtype=torch.long)
    for i, uu_items in enumerate(u_users_items):
        for j, ui in enumerate(uu_items):
            u_user_item_pad[i, j, :len(ui), :] = torch.LongTensor(ui)

    i_user_pad = torch.zeros([batch_size, i_users_maxlen, 2], dtype=torch.long)
    for i, iu in enumerate(i_users):
        i_user_pad[i, :len(iu), :] = torch.LongTensor(iu)

    return [torch.LongTensor(uids), torch.LongTensor(iids), torch.FloatTensor(ratings), torch.FloatTensor(labels), \
           torch.FloatTensor(u_avgs),torch.FloatTensor(i_avgs), u_item_pad,i_user_pad, u_user_pad, u_user_item_pad]


"""
    Some handy functions for pytroch model training ...
"""
# Checkpoints

def save_checkpoint(model, model_dir):
    torch.save(model.state_dict(), model_dir)


def resume_checkpoint(model, model_dir):
    state_dict = torch.load(model_dir)
    model.load_state_dict(state_dict)


# Hyper params

def use_optimizer(network, params):
    if params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(),
                                    lr=params['lr'],
                                    momentum=params['sgd_momentum'],
                                    weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(),
                                                          lr=params['lr'],
                                                          weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(network.parameters(),
                                        lr=params['lr'],
                                        alpha=params['rmsprop_alpha'],
                                        momentum=params['rmsprop_momentum'])
    return optimizer
