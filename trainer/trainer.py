import random
import numpy as np
import torch
import torch.nn as nn


class Trainer:
  def __init__(self, model, device='cuda'):
    super(Trainer, self).__init__()
    self.model = model.to(device)
    self.device = device
    self.optimizer = None
    self.loss_line = []
    self.iter_milestones = []
    return

  def train(self, train_loader, criterion, dev_loader, epochs=100, learning_rate=0.0001, steps=20,
            divide_rate=0.2, high=0.85, low=0.15, generation_threshold=0.15, switched_train_loader=False, freeze_neurons=True):
    if switched_train_loader:
      self.iter_milestones.append(len(self.loss_line))
    for epoch in range(epochs):
      self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
      # Training
      train_correct = 0
      train_loss = 0
      print(f'Training epoch: {epoch+1}...')
      self.model.train()
      for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(self.device), target.to(self.device)
        self.optimizer.zero_grad()
        output = self.model(data)
        with torch.no_grad():
          predicted = output.argmax(dim=1, keepdim=True)
          train_correct += predicted.eq(target.view_as(predicted)).sum().item()
        loss = criterion(output, target)
        loss.backward()
        self.optimizer.step()
        train_loss += loss.item()
        if batch_idx % steps == 0:
          print(f'Epoch: {epoch+1} [{batch_idx*train_loader.batch_size}/{len(train_loader.dataset)} ({(100.*batch_idx/len(train_loader)):.0f}%)]\tTrain Loss: {loss.item():.5f}')
        self.loss_line.append(loss.item())      
      train_acc = 100 * train_correct / len(train_loader.dataset)
      train_loss /= len(train_loader)
      print(f'********** TRAIN_LOSS = {train_loss:.5f} **********')
      print(f'********** TRAIN_ACC = {train_acc:.2f}% ************')

      if freeze_neurons:
        num_frozen = self.model.freeze(dev_loader, high, low, labels=(1, 0))

        if num_frozen > 0:
          print(f'Freezed {num_frozen} neurons')
          # if self.model.num_unfrozen_neurons/self.model.num_neurons <= generation_threshold:
          #   self.model.generate_neurons(divide_rate)
          #   divide_rate -= 0.00025
        else:
          print('Nothing has been frozen')
        if (epoch+1) % 25 == 0:
          self.model.generate_neurons(divide_rate)
          divide_rate *= 0.9
      # TODO: Prune neurons when necessary (Perhaps for the future)

      print(f'Finished training epoch {epoch+1}\n')
    return
