"""
@author: Jiajia Chan
@date: 20 June, 2020
"""
import random
import argparse
import pickle
import pandas as pd
from scipy.io import loadmat
from rating_data import SingleGenerator
from trust_data import SocialGenerator

random.seed(1234)


if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--dataset', default='Ciao', help='dataset name: Ciao/Epinions')
    parser.add_argument('--test_prop', default=0.2, help='the proportion of data used for test')
    args = parser.parse_args() 
    workdir = 'data/'

    click_f = loadmat(workdir + args.dataset + '/rating.mat')['rating']
    trust_f = loadmat(workdir + args.dataset + '/trustnetwork.mat')['trustnetwork']
    click_dt = pd.DataFrame(click_f)
    trust_dt = pd.DataFrame(trust_f, columns=['userID', 'freID'])
    click_dt = click_dt[[0, 1, 3]]
    click_dt.dropna(inplace=True)
    click_dt.drop_duplicates(inplace=True)
    click_dt.columns = ['userID', 'itemID', 'rating']
    trust_dt.dropna(inplace=True)
    trust_dt.drop_duplicates(inplace=True)

    single_generator = SingleGenerator(ratings=click_dt, prob=args.test_prop)
    social_generator = SocialGenerator(singlegenerator=single_generator, trust=trust_dt)
    GDSRec_dataset = social_generator.instance_GDSRec_dataset()
    general_dataset = single_generator.instance_general_dataset()
    with open(workdir + args.dataset +'/' + 'GDSRec' + '_dataset.pkl', 'wb') as f:
        str1 = pickle.dumps(GDSRec_dataset)
        f.write(str1)
        f.close()



