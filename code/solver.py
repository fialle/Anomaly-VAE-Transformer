import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time

from Transformer import TransformerModel
from data_loader import get_loader_segment
import matplotlib.pyplot as plt

def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        import torch

        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)
        torch.cuda.empty_cache()
 

        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train', dataset=self.dataset)
        self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='val', dataset=self.dataset)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test', dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='thre', dataset=self.dataset)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.build_model()

    def build_model(self):

        self.model = TransformerModel(n_feature=self.input_c, d_model=512, nhead=8, d_hid=512, nlayers=3, dropout=0.2).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        if torch.cuda.is_available():
            self.model.cuda()

    def vali(self, vali_loader):
        self.model.eval()

        crit = nn.MSELoss()
        loss_1 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            output = self.model(input, src_mask=None)
            rec_loss = crit(output, input)
            loss_1.append(rec_loss.item())  

        return np.average(loss_1)

    def train(self):

        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)

        train_losses = []
        val_losses = []

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []

            epoch_time = time.time()
            self.model.train()

            for i, (input_data, labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                iter_count += 1

                input = input_data.float().to(self.device)

                output = self.model(input, src_mask=None)

                loss = self.criterion(output, input)

                loss1_list.append(loss.item())
                
                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)

            vali_loss = self.vali(self.vali_loader)
            
            train_losses.append(train_loss)
            val_losses.append(vali_loss)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

        np.save('tf_train_loss.npy', train_losses)
        np.save('tf_val_loss.npy', val_losses)
        plt.plot(train_losses)
        plt.plot(val_losses)
        plt.title('Training/Validation Loss of Transformer')   
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['train loss', 'validation loss'])
        plt.grid(True)
        plt.savefig('tf_loss.png', dpi=500)


    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')))
        self.model.eval()

        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduce=False)

        losses = []
        for i, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            output = self.model(input, src_mask=None)
            loss = torch.mean(criterion(input, output), dim=-1)
            loss = loss.detach().cpu().numpy()
            losses.append(loss)

        losses = np.concatenate(losses, axis=0)
        train_loss = np.array(losses)
        np.save('loss_train_data.npy', train_loss)


        test_labels = []
        losses = []
        outputs = []
        for i, (input_data, labels) in enumerate(self.test_loader):
            input = input_data.float().to(self.device)
            output = self.model(input, src_mask=None)
            outputs.append(output.detach().cpu().numpy())

            loss = torch.mean(criterion(input, output), dim=-1)

            loss = loss.detach().cpu().numpy()
            losses.append(loss)
            test_labels.append(labels)

        outputs = np.concatenate(outputs, axis=0)
        np.save('tf_output.npy', outputs)

        losses = np.concatenate(losses, axis=0)
        test_loss = np.array(losses)
        np.save('loss_test_data.npy', test_loss)

        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)

