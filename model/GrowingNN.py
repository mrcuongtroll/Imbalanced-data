import random
import numpy as np
import torch
import torch.nn as nn
import logging
import model.modules as modules


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
class GrowingMLP(nn.Module):

    def __init__(self, input_size, output_size, device='cuda'):
        super(GrowingMLP, self).__init__()
        self.device = device
        self.activation_table = {}
        self.pre_act_table = {} # Used for GradMax
        self.hidden_table = {} # Used for GradMax
        self.hook_table = {}
        self.linear1 = nn.Linear(input_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 512)
        self.linear4 = nn.Linear(512, 256)
        self.linear5 = nn.Linear(256, 128)
        self.output_layer = nn.Linear(128, output_size)
        self.num_neurons = 128 + 256 + 512 + 256 + 128
        self.num_unfrozen_neurons = 128 + 256 + 512 + 256 + 128
        # self.act = nn.ReLU()
        # self.act = nn.Sigmoid()  # BECAUSE REASONS
        self.act = modules.CustomReLU()
        self.softmax = nn.LogSoftmax(dim=1)
        # Store the number of original neurons for each layers.
        # Only these original neurons can be cloned
        self.nsrc = {'linear1': 128,
                     'linear2': 256,
                     'linear3': 512,
                     'linear4': 256,
                     'linear5': 128}
        self.layers_LUT = {'linear1': self.linear1,
                           'linear2': self.linear2,
                           'linear3': self.linear3,
                           'linear4': self.linear4,
                           'linear5': self.linear5,
                           'output_layer': self.output_layer}
        self.params_state = {}  # 0 if frozen, 1 otherwise
        # Store names of layers that can divide and prune
        self.layer_names_full = ['linear1', 'linear2', 'linear3', 'linear4', 'linear5']
        self.layer_names = ['linear3', 'linear4', 'linear5']
        self.next_layer_look_up = {'linear1': 'linear2',
                                   'linear2': 'linear3',
                                   'linear3': 'linear4',
                                   'linear4': 'linear5',
                                   'linear5': 'output_layer'}
        self.prev_layer_look_up = {'linear1': None,
                                   'linear2': 'linear1',
                                   'linear3': 'linear2',
                                   'linear4': 'linear3',
                                   'linear5': 'linear4'}
        return

    def forward(self, x):
        z1 = self.linear1(x)
        # self.pre_act_table['linear1'] = z1
        z1.requires_grad_().register_hook(lambda grad: self.pre_activation_hook(grad, 'linear1'))
        h1 = self.act(z1)
        self.activation_table['linear1'] = h1.detach().cpu().numpy()
        self.hidden_table['linear1'] = h1
        z2 = self.linear2(h1)
        # self.pre_act_table['linear2'] = z2
        z2.requires_grad_().register_hook(lambda grad: self.pre_activation_hook(grad, 'linear2'))
        h2 = self.act(z2)
        self.activation_table['linear2'] = h2.detach().cpu().numpy()
        self.hidden_table['linear2'] = h2
        z3 = self.linear3(h2)
        # self.pre_act_table['linear3'] = z3
        z3.requires_grad_().register_hook(lambda grad: self.pre_activation_hook(grad, 'linear3'))
        h3 = self.act(z3)
        self.activation_table['linear3'] = h3.detach().cpu().numpy()
        self.hidden_table['linear3'] = h3
        z4 = self.linear4(h3)
        # self.pre_act_table['linear4'] = z4
        z4.requires_grad_().register_hook(lambda grad: self.pre_activation_hook(grad, 'linear4'))
        h4 = self.act(z4)
        self.activation_table['linear4'] = h4.detach().cpu().numpy()
        self.hidden_table['linear4'] = h4
        z5 = self.linear5(h4)
        # self.pre_act_table['linear5'] = z5
        z5.requires_grad_().register_hook(lambda grad: self.pre_activation_hook(grad, 'linear5'))
        h5 = self.act(z5)
        self.activation_table['linear5'] = h5.detach().cpu().numpy()
        self.hidden_table['linear5'] = h5
        out = self.output_layer(h5)
        # self.pre_act_table['output_layer'] = out
        out.requires_grad_().register_hook(lambda grad: self.pre_activation_hook(grad, 'output_layer'))
        out = self.softmax(out)
        return out

    def init_params_state(self):
        # Please call this method as soon as you create the model
        for name, param in self.named_parameters():
            self.params_state[name] = torch.ones(param.shape, requires_grad=False)
        for layer_name in self.layer_names:
            # Only hook layers listed in layer_names
            self.register_freeze_hook(layer_name)
        return

    def freeze_backward_hook(self, grad, param_name):
        # logger.debug(f"Modified gradient of {param_name} by {self.params_state[param_name]}")
        return grad * self.params_state[param_name].to(self.device)

    def register_freeze_hook(self, layer_name: str):
        self.hook_table[f'{layer_name}.bias'] = self.layers_LUT[layer_name].bias.register_hook(
            lambda grad: self.freeze_backward_hook(grad, f'{layer_name}.bias'))
        self.hook_table[f'{layer_name}.weight'] = self.layers_LUT[layer_name].weight.register_hook(
            lambda grad: self.freeze_backward_hook(grad, f'{layer_name}.weight'))
        return

    def freeze_neuron(self, layer_name: str, neuron_id: int):
        self.params_state[layer_name + '.weight'][neuron_id, :] = 0
        self.params_state[layer_name + '.bias'][neuron_id] = 0
        return

    def pre_activation_hook(self, grad, layer_name):
        self.pre_act_table[layer_name] = grad
        return grad

    def generate_neurons(self, rate, data=None, target=None):
        # Divide phase: Randomly clone some neurons.
        # Shall we say the neurons "divide" in this case?
        # Goal: Add new row to the weight of the current layer
        # and add new column to the weight of the next layer.
        # old_params = []
        # new_params = []
        logger.info("Generating new neurons...")
        for layer in self.layer_names_full:
            logger.debug(f'Layer name: {layer}...')
            current_layer = self.layers_LUT[layer]
            prev_layer_name = self.prev_layer_look_up[layer]
            next_layer_name = self.next_layer_look_up[layer]
            next_layer = self.layers_LUT[next_layer_name]
            num_new_neurons = round(self.nsrc[layer] * rate)
            self.num_neurons += num_new_neurons
            self.num_unfrozen_neurons += num_new_neurons
            old_w = current_layer.weight.detach().cpu().numpy()
            old_b = current_layer.bias.detach().cpu().numpy()
            old_next_w = next_layer.weight.detach().cpu().numpy()
            # old_params.append(current_layer.weight)
            # old_params.append(current_layer.bias)
            # old_params.append(next_layer.weight)
            # Randomly generate num_new_neurons neurons
            """
            # ***Random initialization***
            # Append row to new_w
            new_w_rows = np.random.rand(num_new_neurons, old_w.shape[1]) * 0.01
            if prev_layer_name is not None:
                new_w_rows = new_w_rows * self.params_state[prev_layer_name + '.bias'].detach().cpu().numpy()
            new_w = np.append(old_w, new_w_rows, axis=0)
            # Append scalar to new_b
            new_b = np.append(old_b, np.random.rand(num_new_neurons) * 0.01)
            # new_b = np.append(old_b, np.zeros(num_new_neurons))
            # Append column to new_next_w
            # Dude, I suggest making new connections in params_state table 0
            # for frozen neurons.
            # Once a neuron is frozen, no gradient should flow through it
            new_next_w_cols = np.random.rand(old_next_w.shape[0], num_new_neurons) * 0.01
            new_next_w_cols = new_next_w_cols * self.params_state[next_layer_name + '.bias'].detach().cpu().numpy()[:,
                                                None]  # transpose
            new_next_w = np.append(old_next_w, new_next_w_cols, axis=1)
            """
            # ***GradMax initialization***
            new_w_rows = np.zeros((num_new_neurons, old_w.shape[1]))
            if prev_layer_name is not None:
                new_w_rows = new_w_rows * self.params_state[prev_layer_name + '.bias'].detach().cpu().numpy()
            new_w = np.append(old_w, new_w_rows, axis=0)
            new_b = np.append(old_b, np.zeros(num_new_neurons))
            # Requirement: A backward pass using a batch of data from dev set, keep the gradients.
            if prev_layer_name is None:
                prev_h = data
            else:
                prev_h = self.hidden_table[prev_layer_name]
            next_pre_act_grad = self.pre_act_table[next_layer_name]
            objective = next_pre_act_grad.T @ prev_h
            u, s, vh = torch.linalg.svd(objective)
            new_next_w_cols = (u[:, :num_new_neurons] / torch.linalg.norm(s[:num_new_neurons]) * 1).detach().cpu().numpy()   # c = 1 for now
            # logger.debug(f"Layer: {next_layer_name} | New cols: {new_next_w_cols}")
            # In case the number of left-singular vectors is smaller than the number of neurons to be generated
            if new_next_w_cols.shape[1] < num_new_neurons:
                new_next_w_cols = np.tile(new_next_w_cols, (1, num_new_neurons // new_next_w_cols.shape[1]))
                new_next_w_cols = np.append(new_next_w_cols, new_next_w_cols[:, :num_new_neurons-new_next_w_cols.shape[1]], axis=1)
            new_next_w_cols = new_next_w_cols * self.params_state[next_layer_name + '.bias'].detach().cpu().numpy()[:,
                                                None]  # transpose
            new_next_w = np.append(old_next_w, new_next_w_cols, axis=1)
            # Create new parameters
            current_layer.weight = nn.Parameter(torch.tensor(new_w, dtype=torch.float, device=self.device),
                                                requires_grad=True)
            logger.debug(f"Layer name: {layer} --> New weight generated.")
            current_layer.bias = nn.Parameter(torch.tensor(new_b, dtype=torch.float, device=self.device),
                                              requires_grad=True)
            logger.debug(f'Layer name: {layer} --> New bias generated.')
            next_layer.weight = nn.Parameter(torch.tensor(new_next_w, dtype=torch.float, device=self.device),
                                             requires_grad=True)
            logger.debug(f'Layer name: {next_layer_name} --> New columns added to weight.')
            # new_params.append(current_layer.weight)
            # new_params.append(current_layer.weight)
            # new_params.append(next_layer.weight)
            # Update params_state table. Make new neurons trainable by default
            # Zero out gradient flow through frozen neurons
            logger.debug('Updating parameters states...')
            old_w_state = self.params_state[layer + '.weight'].detach().cpu().numpy()
            new_w_state_rows = np.ones((num_new_neurons, old_w_state.shape[1]), dtype=old_w_state.dtype)
            if prev_layer_name is not None:
                # frozen neurons of the previous layer will have state set to 0
                new_w_state_rows = new_w_state_rows * self.params_state[
                    prev_layer_name + '.bias'].detach().cpu().numpy()
            new_w_state = np.append(old_w_state, new_w_state_rows, axis=0)
            old_b_state = self.params_state[layer + '.bias'].detach().cpu().numpy()
            new_b_state = np.append(old_b_state, np.ones((num_new_neurons,)))
            old_next_w_state = self.params_state[next_layer_name + '.weight'].detach().cpu().numpy()
            new_next_w_state_cols = np.ones((old_next_w_state.shape[0], num_new_neurons), dtype=old_next_w_state.dtype)
            new_next_w_state_cols = new_next_w_state_cols * self.params_state[
                                                                next_layer_name + '.bias'].detach().cpu().numpy()[:,
                                                            None]  # transpose
            new_next_w_state = np.append(old_next_w_state, new_next_w_state_cols, axis=1)
            self.params_state[layer + '.weight'] = torch.tensor(new_w_state, dtype=torch.float, requires_grad=False)
            self.params_state[layer + '.bias'] = torch.tensor(new_b_state, dtype=torch.float, requires_grad=False)
            self.params_state[next_layer_name + '.weight'] = torch.tensor(new_next_w_state, dtype=torch.float,
                                                                          requires_grad=False)
            # Register new freeze hook
            logger.debug('Registering backward hook...')
            if layer in self.layer_names:
                self.register_freeze_hook(layer)
            logger.info(f'Layer name: {layer} --> Done.')
        return #{'old': old_params, 'new': new_params}

    def freeze(self, dev_loader, high, low, labels=(0, 1)):
        # Freeze any neuron that has taken a specific role
        logger.info('Freezing neurons...')
        self.eval()
        num_frozen = 0
        with torch.no_grad():
            # TODO: check which neurons have very high activation for only a given
            # class, freeze them
            # Suppose we have a dev set
            label_count = {}
            for label in labels:
                label_count[label] = 0
            activation_by_label = {}
            for layer in self.layer_names:
                activation_by_label[layer] = {}
                for index in range(self.activation_table[layer].shape[1]):
                    activation_by_label[layer][index] = {}
                    for label in labels:
                        activation_by_label[layer][index][label] = 0
            for batch_idx, (data, target) in enumerate(dev_loader):
                data, target = data.to(self.device), target.to(self.device)
                latest_label = target.detach().cpu().numpy()
                output = self.forward(data)
                for layer in self.layer_names:
                    activation = self.activation_table[layer]
                    for label in np.unique(latest_label):
                        label_mask = latest_label == label
                        label_count[label] += np.sum(label_mask)
                        for index in range(activation.shape[1]):
                            activation_by_label[layer][index][label] += np.sum(activation[:, index] * label_mask)
            for layer in self.layer_names:
                for index in range(len(self.activation_table[layer][0])):
                    # Check for every label
                    for label in label_count.keys():
                        if label_count[label] > 0:
                            activation_by_label[layer][index][label] /= label_count[label]
                    for label in label_count.keys():
                        other_act = 0
                        num_other_labels = 0
                        for other_label in [i for i in label_count.keys() if i != label]:
                            other_act += activation_by_label[layer][index][other_label]
                            num_other_labels += 1
                        other_act /= num_other_labels
                        if activation_by_label[layer][index][label] >= high and other_act <= low:
                            if self.params_state[layer + '.bias'][index] == 1:
                                # If the neuron is already frozen, don't freeze it
                                # TODO: only freeze if that neuron contributes to a good prediction
                                self.freeze_neuron(layer, index)
                                self.num_unfrozen_neurons -= 1
                                num_frozen += 1
        return num_frozen
