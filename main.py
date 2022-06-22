import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import pickle
import torch
from torch.utils.data import DataLoader
from utils import collate_fn
from model.GDSRec_model import GDSRec_Engine


config = {
    'model': 'GDSRec', 
    'dataset': 'Ciao',  # Ciao/Epinions
    'optimizer': 'adam',
    'l2_regularization': 0.01,
    'embed_size': 64,
    'batch_size': 128,
    'layers': [256,128,128,64,64],
    'epoch': 20,
    'lr': 0.0005,  # 0.01, 0.001, 0.0001
    'lr_dc': 0.1,  # learning rate decay
    'lr_dc_step': 100,  # the number steps for decay
    'test':False,
    'model_dir': 'checkpoints/{}_{}_best_checkpoint.model'
}
print(config)
here = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
config['device'] = device
workdir = 'C:/Users/meicr/Desktop/GDSRec_rank/data/'
with open(workdir + config['dataset'] + '/' + config['model'] + '_dataset.pkl', 'rb') as f:
    train_dataset, evaluate_dataset, test_dataset, user_count, item_count = pickle.loads(f.read())
config['num_users'] = user_count
config['num_items'] = item_count
config['num_rates'] = 5

if config['test'] is False:
    engine = GDSRec_Engine(config)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    evaluate_loader = DataLoader(evaluate_dataset, batch_size=config['batch_size'], shuffle=False,
                                    collate_fn=collate_fn)
    index_sum = []
    for epoch in range(config['epoch']):
        print('Epoch {} starts !'.format(epoch))
        engine.train_an_epoch(train_loader, epoch)
        recall, ndcg = engine.evaluate(evaluate_loader, epoch)
        if epoch == 0:
            pre_sum = recall + ndcg
            index_sum.append(0)
        else:
            if recall + ndcg < pre_sum:
                index_sum.append(1)
            else:
                pre_sum = recall + ndcg
                index_sum.append(0)
        if sum(index_sum[-10:]) == 10:
            break
        if epoch == 0:
            best_sum = recall + ndcg
            engine.save()
        elif recall + ndcg > best_sum:
            best_sum = recall + ndcg
            engine.save()
else:
    engine = GDSRec_Engine(config)
    print('Load checkpoint and testing...')
    engine.resume()
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    recall, ndcg = engine.evaluate(test_loader, epoch_id=0)
