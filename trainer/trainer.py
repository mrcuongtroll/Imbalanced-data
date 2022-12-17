import random
import numpy as np
import torch
import torch.nn as nn
import logging
import copy
import os
import dill
from model import losses
from definitions import *


# Checkpoint path
if not os.path.exists(os.path.abspath(CHECKPOINT_DIR)):
    os.makedirs(os.path.abspath(CHECKPOINT_DIR))


# Logging
logger = logging.getLogger(name=__name__)


# Classes
class Trainer:
    def __init__(self, model, optimizer: type(torch.optim.Optimizer), learning_rate=0.0001, device='cuda',
                 checkpoint_name=None):
        super(Trainer, self).__init__()
        self.model = model.to(device)
        self.device = device
        self.optimizer_class = optimizer
        self.learning_rate = learning_rate
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)
        self.train_loss_history = []
        self.dev_loss_history = []
        self.iter_milestones = []
        self.checkpoint_name = checkpoint_name
        if not os.path.exists(os.path.join(CHECKPOINT_DIR, self.checkpoint_name)):
            os.makedirs(os.path.join(CHECKPOINT_DIR, self.checkpoint_name))
        return

    def train(self, train_loader, criterion, dev_loader, epochs=100, num_prints=10,
              gen_rate=0.2, high=0.85, low=0.15, generation_threshold=0.3, switched_train_loader=False,
              freeze_neurons=True):
        best_dev_loss = np.inf
        best_model = None
        if switched_train_loader:
            self.iter_milestones.append(len(self.train_loss_history))
        samples_per_print = len(train_loader.dataset) / num_prints
        samples_per_print_dev = len(dev_loader.dataset) / num_prints
        for epoch in range(epochs):
            # optimizer = self.optimizer(self.model.parameters(), lr=learning_rate)
            print_num = 1
            # Training
            train_correct = 0
            train_loss = 0
            logger.info(f'Training epoch: {epoch + 1}...')
            self.model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                # optimizer.zero_grad()
                self.optimizer.zero_grad()
                output = self.model(data)
                if isinstance(criterion, losses.DRLoss):
                    output_feature = self.model.hidden_table[self.model.prev_layer_look_up['output_layer']]
                    output_feature = output_feature.reshape(output_feature.size(0), -1)
                    with torch.no_grad():
                        output_feature_nograd = output_feature.detach()
                        feature_length = torch.clamp(torch.sqrt(torch.sum(output_feature_nograd ** 2, dim=1, keepdims=False)),
                                                     1e-8)
                    learned_norm = losses.produce_Ew(target, 2)
                    cur_M = self.model.output_layer.ori_M * learned_norm
                    loss = criterion(output_feature, target, cur_M, feature_length, reg_lam=0)
                else:
                    loss = criterion(output, target)
                with torch.no_grad():
                    predicted = output.argmax(dim=1, keepdim=True)
                    train_correct += predicted.eq(target.view_as(predicted)).sum().item()
                loss.backward()
                # optimizer.step()
                self.optimizer.step()
                train_loss += loss.item()
                if batch_idx * train_loader.batch_size >= print_num * samples_per_print:
                    logger.info(
                        f'Training | Epoch: {epoch + 1} [{batch_idx * train_loader.batch_size}/{len(train_loader.dataset)}'
                        f' ({(100. * batch_idx / len(train_loader)):.0f}%)] | Training Loss: {loss.item():.5f}')
                    print_num += 1
                self.train_loss_history.append(loss.item())
                del data, target
            train_acc = 100 * train_correct / len(train_loader.dataset)
            train_loss /= len(train_loader)

            # Growing
            self.model.eval()
            if freeze_neurons:
                num_frozen = self.model.freeze(dev_loader, high, low, labels=(1, 0))

                if num_frozen > 0:
                    logger.info(f'Freezed {num_frozen} neurons')
                    # if self.model.num_unfrozen_neurons / self.model.num_neurons <= generation_threshold:
                    #     self.model.generate_neurons(gen_rate)
                    #     gen_rate -= 0.00025
                else:
                    logger.info('Nothing has been frozen for this epoch')
                if (epoch + 1) % 5 == 0 or self.model.num_unfrozen_neurons / self.model.num_neurons <= generation_threshold:
                    # Was 25 epochs
                    dev_iter = iter(dev_loader)
                    data, target = next(dev_iter)
                    data, target = data.to(self.device), target.to(self.device)
                    pred = self.model(data)
                    if isinstance(criterion, losses.DRLoss):
                        output_feature = self.model.hidden_table[self.model.prev_layer_look_up['output_layer']]
                        output_feature = output_feature.view(output_feature.size(0), -1)
                        with torch.no_grad():
                            output_feature_nograd = output_feature.detach()
                            feature_length = torch.clamp(torch.sqrt(torch.sum(output_feature_nograd ** 2, dim=1, keepdims=False)),
                                                      1e-8)
                        learned_norm = losses.produce_Ew(target, 2)
                        cur_M = self.model.output_layer.ori_M * learned_norm
                        loss = criterion(output_feature, target, cur_M, feature_length, reg_lam=0)
                    else:
                        loss = criterion(pred, target)
                    loss.backward()
                    self.model.generate_neurons(gen_rate, data, target)
                    gen_rate *= 0.9
                    self.optimizer = self.optimizer_class(self.model.parameters(), lr=self.learning_rate)
                    self.optimizer.zero_grad()
                    del data, target, dev_iter
            # TODO: Prune neurons when necessary (Perhaps for the future)

            logger.info(f'Finished training epoch {epoch + 1} '
                        f'| TRAIN_LOSS = {train_loss:.5f} | TRAIN_ACC = {train_acc:.2f}%')

            # Validation
            print_num_dev = 1
            dev_correct = 0
            dev_loss = 0
            self.model.eval()
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(dev_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    predicted = output.argmax(dim=1, keepdim=True)
                    dev_correct += predicted.eq(target.view_as(predicted)).sum().item()
                    if isinstance(criterion, losses.DRLoss):
                        output_feature = self.model.hidden_table[self.model.prev_layer_look_up['output_layer']]
                        output_feature = output_feature.reshape(output_feature.size(0), -1)
                        with torch.no_grad():
                            output_feature_nograd = output_feature.detach()
                            feature_length = torch.clamp(torch.sqrt(torch.sum(output_feature_nograd ** 2, dim=1, keepdims=False)),
                                                      1e-8)
                        learned_norm = losses.produce_Ew(target, 2)
                        cur_M = self.model.output_layer.ori_M * learned_norm
                        loss = criterion(output_feature, target, cur_M, feature_length, reg_lam=0)
                    else:
                        loss = criterion(output, target)
                    dev_loss += loss.item()
                    if batch_idx * dev_loader.batch_size >= print_num_dev * samples_per_print_dev:
                        logger.info(
                            f'Validating | Epoch: {epoch + 1} [{batch_idx * dev_loader.batch_size}/{len(dev_loader.dataset)}'
                            f' ({(100. * batch_idx / len(dev_loader)):.0f}%)] | Validation Loss: {loss.item():.5f}')
                        print_num_dev += 1
                    self.dev_loss_history.append(loss.item())
                    del data, target
            dev_acc = 100 * dev_correct / len(dev_loader.dataset)
            dev_loss /= len(dev_loader)
            logger.info(f'Validation results for epoch {epoch + 1} '
                        f'| VALID LOSS = {dev_loss:.5f} | VALID ACC = {dev_acc:.2f}%')
            if dev_loss <= best_dev_loss:
                best_dev_loss = dev_loss
                logger.info(f"New best validation loss: {best_dev_loss:.5f}.")
                # best_model = copy.deepcopy(self.model)
                # Save checkpoint
                best_model_path = os.path.join(CHECKPOINT_DIR, self.checkpoint_name, "best.th")
                torch.save(self.model, best_model_path, pickle_module=dill)
            # Saving
            checkpoint_path = os.path.join(CHECKPOINT_DIR, self.checkpoint_name, "checkpoint.th")
            torch.save(self.model, checkpoint_path, pickle_module=dill)
            torch.save({'optimizer': self.optimizer.state_dict(),
                        'train_loss': self.train_loss_history,
                        'dev_loss': self.dev_loss_history},
                       os.path.join(CHECKPOINT_DIR, self.checkpoint_name, "trainer.th"),
                       pickle_module=dill)
            logger.info(f'Summary for epoch {epoch + 1} '
                        f'| TRAIN LOSS = {train_loss:.5f} | VALID LOSS = {dev_loss:.5f} '
                        f'| BEST VALID LOSS: {best_dev_loss:.5f}\n')
        return


# Test
if __name__ == '__main__':
    print(CHECKPOINT_DIR)
