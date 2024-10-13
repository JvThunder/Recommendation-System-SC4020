import torch
from torch.autograd import Variable
from tqdm import tqdm
from tensorboardX import SummaryWriter
from .utils import save_checkpoint, use_optimizer
from .metrics import MetronAtK


class Engine(object):
    """Meta Engine for training & evaluating NCF model

    Note: Subclass should implement self.model !
    """

    def __init__(self, config):
        self.config = config  # model configuration
        self._metron = MetronAtK(top_k=10)
        self._writer = SummaryWriter(log_dir='runs/{}'.format(config['alias']))  # tensorboard writer
        self._writer.add_text('config', str(config), 0)
        self.opt = use_optimizer(self.model, config)
        # explicit feedback
        # self.crit = torch.nn.MSELoss()
        # implicit feedback
        self.crit = torch.nn.BCELoss()

    def train_single_batch(self, users, items, ratings):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        if self.config['use_cuda'] is True:
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
        self.opt.zero_grad()
        ratings_pred = self.model(users, items)
        loss = self.crit(ratings_pred.view(-1), ratings)
        loss.backward()
        self.opt.step()
        loss = loss.item()
        return loss

    def train_an_epoch(self, train_loader, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.train()
        total_loss = 0
        for batch_id, batch in enumerate(train_loader):
            assert isinstance(batch[0], torch.LongTensor)
            user, item, rating = batch[0], batch[1], batch[2]
            rating = rating.float()
            loss = self.train_single_batch(user, item, rating)
            print('[Training Epoch {}] Batch {}, Loss {}'.format(epoch_id, batch_id, loss))
            total_loss += loss
        self._writer.add_scalar('model/loss', total_loss, epoch_id)

    def evaluate(self, evaluate_data, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.eval()
        with torch.no_grad():
            test_users, test_items, test_realscore = evaluate_data[0], evaluate_data[1], evaluate_data[2]
            negative_users, negative_items = evaluate_data[3], evaluate_data[4]
            if self.config['use_cuda'] is True:
                test_users = test_users.cuda()
                test_items = test_items.cuda()
                negative_users = negative_users.cuda()
                negative_items = negative_items.cuda()

        if self.config['use_bachify_eval'] == False:    
            test_scores = self.model(test_users, test_items)
            negative_scores = self.model(negative_users, negative_items)
        else: 
            test_scores = []
            negative_scores = []
            bs = self.config['batch_size']
            for start_idx in range(0, len(test_users), bs):
                end_idx = min(start_idx + bs, len(test_users))
                batch_test_users = test_users[start_idx:end_idx]
                batch_test_items = test_items[start_idx:end_idx]
                test_scores.append(self.model(batch_test_users, batch_test_items))
            for start_idx in tqdm(range(0, len(negative_users), bs)):
                end_idx = min(start_idx + bs, len(negative_users))
                batch_negative_users = negative_users[start_idx:end_idx]
                batch_negative_items = negative_items[start_idx:end_idx]
                negative_scores.append(self.model(batch_negative_users, batch_negative_items))
            test_scores = torch.concatenate(test_scores, dim=0) 
            negative_scores = torch.concatenate(negative_scores, dim=0)


            if self.config['use_cuda'] is True:
                test_users = test_users.cpu()
                test_items = test_items.cpu()
                test_scores = test_scores.cpu()
                test_realscore = test_realscore.cpu()
                negative_users = negative_users.cpu()
                negative_items = negative_items.cpu()
                negative_scores = negative_scores.cpu()
            self._metron.subjects = [test_users.data.view(-1).tolist(),
                                 test_items.data.view(-1).tolist(),
                                 test_scores.data.view(-1).tolist(),
                                 test_realscore.data.view(-1).tolist(), 
                                 negative_users.data.view(-1).tolist(),
                                 negative_items.data.view(-1).tolist(),
                                 negative_scores.data.view(-1).tolist()]

        precision, recall = self._metron.cal_precision(), self._metron.cal_recall()
        self._writer.add_scalar('performance/Precision', precision, epoch_id)
        self._writer.add_scalar('performance/Recall', recall, epoch_id)
        
        print('[Evaluating Epoch {}] Precision = {:.4f}, Recall = {:.4f}'.format(epoch_id, precision, recall))
        return precision, recall

    def save(self, alias, epoch_id, precision, recall):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        model_dir = self.config['model_dir'].format(alias, epoch_id, precision, recall)
        save_checkpoint(self.model, model_dir)