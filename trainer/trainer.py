import random
import numpy as np
import torch
import torch.nn as nn
import logging

# Logging
logger = logging.getLogger(name=__name__)


# logger.propagate = False
# stream_handler = logging.StreamHandler()
# file_handler = logging.FileHandler('./logs/log_old.log')
# formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
# stream_handler.setFormatter(formatter)
# stream_handler.setLevel(logging.INFO)
# file_handler.setFormatter(formatter)
# file_handler.setLevel(logging.DEBUG)
# logger.addHandler(stream_handler)
# logger.addHandler(file_handler)
# logger.setLevel(logging.NOTSET)


# Classes
class Trainer:
    def __init__(self, model, optimizer: type(torch.optim.Optimizer), learning_rate=0.0001, device='cuda'):
        super(Trainer, self).__init__()
        self.model = model.to(device)
        self.device = device
        self.optimizer_class = optimizer
        self.learning_rate = learning_rate
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)
        self.loss_history = []
        self.iter_milestones = []
        return

    def train(self, train_loader, criterion, dev_loader, epochs=100, num_prints=10,
              gen_rate=0.2, high=0.85, low=0.15, generation_threshold=0.15, switched_train_loader=False,
              freeze_neurons=True):
        if switched_train_loader:
            self.iter_milestones.append(len(self.loss_history))
        for epoch in range(epochs):
            # optimizer = self.optimizer(self.model.parameters(), lr=learning_rate)
            samples_per_print = len(train_loader.dataset)/num_prints
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
                with torch.no_grad():
                    predicted = output.argmax(dim=1, keepdim=True)
                    train_correct += predicted.eq(target.view_as(predicted)).sum().item()
                loss = criterion(output, target)
                loss.backward()
                # optimizer.step()
                self.optimizer.step()
                train_loss += loss.item()
                if batch_idx * train_loader.batch_size >= print_num * samples_per_print:
                    logger.info(
                        f'Training | Epoch: {epoch + 1} [{batch_idx * train_loader.batch_size}/{len(train_loader.dataset)}'
                        f' ({(100. * batch_idx / len(train_loader)):.0f}%)] | Training Loss: {loss.item():.5f}')
                    print_num += 1
                self.loss_history.append(loss.item())
                del data, target
            train_acc = 100 * train_correct / len(train_loader.dataset)
            train_loss /= len(train_loader)

            self.model.eval()
            if freeze_neurons:
                num_frozen = self.model.freeze(dev_loader, high, low, labels=(1, 0))

                if num_frozen > 0:
                    logger.info(f'Freezed {num_frozen} neurons')
                    if self.model.num_unfrozen_neurons / self.model.num_neurons <= generation_threshold:
                        self.model.generate_neurons(gen_rate)
                        gen_rate -= 0.00025
                else:
                    logger.info('Nothing has been frozen for this epoch')
                if (epoch + 1) % 25 == 0:
                    dev_iter = iter(dev_loader)
                    data, target = next(dev_iter)
                    data, target = data.to(self.device), target.to(self.device)
                    pred = self.model(data)
                    loss = criterion(pred, target)
                    loss.backward()
                    self.model.generate_neurons(gen_rate, data, target)
                    gen_rate *= 0.9
                    self.optimizer = self.optimizer_class(self.model.parameters(), lr=self.learning_rate)
                    self.optimizer.zero_grad()
                    del data, target, dev_iter
            # TODO: Prune neurons when necessary (Perhaps for the future)

            logger.info(f'Finished training epoch {epoch + 1} '
                        f'| TRAIN_LOSS = {train_loss:.5f} | TRAIN_ACC = {train_acc:.2f}%\n')
        return
