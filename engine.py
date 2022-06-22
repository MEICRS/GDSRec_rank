import torch
from torch.autograd import Variable

from utils import save_checkpoint, use_optimizer, resume_checkpoint
from metrics import MetronAtK


class Engine(object):
    def __init__(self, config):
        self.config = config  # model configuration
        self._metron = MetronAtK(top_k=5)
        self.opt = use_optimizer(self.model, config)
        self.crit = torch.nn.BCELoss()

    def train_single_batch(self):
        pass

    def train_an_epoch(self, train_loader, epoch_id):
        pass

    def evaluate(self, evaluate_data, epoch_id):
        pass

    def save(self):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        model_dir = self.config['model_dir'].format(self.config['model'],self.config['dataset'])
        save_checkpoint(self.model, model_dir)

    def resume(self):
        model_dir = self.config['model_dir'].format(self.config['model'],self.config['dataset'])
        resume_checkpoint(self.model, model_dir)
