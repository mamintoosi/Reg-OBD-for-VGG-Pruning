import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class sparse_regularization(object):

    def __init__(self, model: nn.Module, device):
        self.model = model
        self.device = device

    # L2 regularization, M.Amintoosi
    def l2_regularization(self, _lambda):
        x = 0
        for n, _module in self.model.named_modules():
            if isinstance(_module, nn.Conv2d) and (not 'downsample' in n):
                x += _lambda*torch.norm(torch.flatten(_module.weight), 2)
        return x

    # L1 regularization
    def l1_regularization(self, _lambda):
        x = 0
        for n, _module in self.model.named_modules():
            if isinstance(_module, nn.Conv2d) and (not 'downsample' in n):
                x += _lambda*torch.norm(torch.flatten(_module.weight), 1)
        return x

    # group lasso regularization
    def group_lasso_regularization(self, _lambda):
        x = 0
        for n, _module in self.model.named_modules():
            if isinstance(_module, nn.Conv2d) and (not 'downsample' in n):
                p = _module.weight
                p = p.reshape(p.shape[0], p.shape[1], p.shape[2]*p.shape[3])

                # group lasso regularization based on the feature wise grouping
                x += _lambda*torch.sum(torch.sqrt(torch.sum(torch.sum(p**2, 0), 1)))

                # group lasso regularization based on the neuron wise grouping
                #x += _lambda*torch.sum(torch.sqrt(torch.sum(torch.sum(p**2,1),1)))
        return x

    # hierarchical square rooted group lasso regularization
    def hierarchical_square_rooted_group_lasso_regularization(self, _lambda):
        x = 0
        for n, _module in self.model.named_modules():
            if isinstance(_module, nn.Conv2d) and (not 'downsample' in n):
                p = _module.weight
                number_of_out_channels = p.shape[0]
                number_of_in_channels = p.shape[1]
                p = p.reshape(p.shape[0]*p.shape[1], p.shape[2]*p.shape[3])
                p = torch.sum(p**2, 1)
                p = p.reshape(number_of_out_channels, number_of_in_channels)

                # hierarchical square rooted group lasso regularization based on the feature wise grouping
                x += _lambda*torch.sum(torch.sqrt(torch.sum(torch.sqrt(p), 0)))

                # hierarchical square rooted group lasso regularization based on the neuron wise grouping
                #x+= _lambda*torch.sum(torch.sqrt(torch.sum(torch.sqrt(p),1)))
        return x

    # hierarchical squared group lasso regularization
    def hierarchical_squared_group_lasso_regularization(self, _lambda):
        x = 0
        for n, _module in self.model.named_modules():
            if isinstance(_module, nn.Conv2d) and (not 'downsample' in n):
                p = _module.weight
                number_of_out_channels = p.shape[0]
                number_of_in_channels = p.shape[1]
                p = p.reshape(p.shape[0]*p.shape[1], p.shape[2]*p.shape[3])
                p = torch.sum(p**2, 1)
                p = p.reshape(number_of_out_channels, number_of_in_channels)

                # hierarchical squared group lasso regularization based on the feature wise grouping
                x += _lambda*torch.sum((torch.sum(torch.sqrt(p), 0))**2)

                # hierarchical squared group lasso regularization based on the neuron wise grouping
                #x+= _lambda*torch.sum((torch.sum(torch.sqrt(p),1))**2)
        return x

    # exclusive sparsity regularization
    def exclusive_sparsity_regularization(self, _lambda):
        x = 0
        for n, _module in self.model.named_modules():
            if isinstance(_module, nn.Conv2d) and (not 'downsample' in n):
                p = _module.weight
                p = p.reshape(p.shape[0], p.shape[1], p.shape[2]*p.shape[3])

                # exclusive sparsity regularization based on the feature wise grouping
                x += _lambda*torch.sum((torch.sum(torch.sum(torch.abs(p), 0), 1))**2)

                # exclusive sparsity regularization based on the feature wise grouping
                #x += _lambda*torch.sum((torch.sum(torch.sum(torch.abs(p),1),1))**2)
        return x

    # hierarchical square rooted exclusive sparsity regularization
    def hierarchical_square_rooted_exclusive_sparsity_regularization(self, _lambda):
        x = 0
        for n, _module in self.model.named_modules():
            if isinstance(_module, nn.Conv2d) and (not 'downsample' in n):
                p = _module.weight
                number_of_out_channels = p.shape[0]
                number_of_in_channels = p.shape[1]
                p = p.reshape(p.shape[0]*p.shape[1], p.shape[2]*p.shape[3])
                p = torch.sum(torch.abs(p), 1)
                p = p.reshape(number_of_out_channels, number_of_in_channels)

                # hierarchical square rooted exclusive sparsity regularization based on the feature wise grouping
                x += _lambda*torch.sum(torch.sqrt(torch.sum(p**2, 0)))

                # hierarchical square rooted exclusive sparsity regularization based on the neuron wise grouping
                #x+= _lambda*torch.sum(torch.sqrt(torch.sum(p**2,1)))
        return x

    # hierarchical squared exclusive sparsity regularization
    def hierarchical_squared_exclusive_sparsity_regularization(self, _lambda):
        x = 0
        for n, _module in self.model.named_modules():
            if isinstance(_module, nn.Conv2d) and (not 'downsample' in n):
                p = _module.weight
                number_of_out_channels = p.shape[0]
                number_of_in_channels = p.shape[1]
                p = p.reshape(p.shape[0]*p.shape[1], p.shape[2]*p.shape[3])
                p = torch.sum(torch.abs(p), 1)
                p = p.reshape(number_of_out_channels, number_of_in_channels)

                # hierarchical squared exclusive sparsity regularization based on the feature wise grouping
                x += _lambda*torch.sum((torch.sum(p**2, 0))**2)

                # hierarchical squared exclusive sparsity regularization based on the neuron wise grouping
                #x+= _lambda*torch.sum((torch.sum(p**2,1))**2)
        return x

    # group l1/2 regularization
    def group_l12_regularization(self, _lambda):
        x = 0
        for n, _module in self.model.named_modules():
            if isinstance(_module, nn.Conv2d) and (not 'downsample' in n):
                p = _module.weight
                p = p.reshape(p.shape[0], p.shape[1], p.shape[2]*p.shape[3])

                # group l1/2 regularization based on the feature wise grouping
                x += _lambda*torch.sum(torch.sqrt(
                    torch.sum(torch.sum(torch.abs(p), 0), 1)))

                # group l1/2 regularization based on the feature wise grouping
                #x += _lambda*torch.sum(torch.sqrt(torch.sum(torch.sum(torch.abs(p),1),1)))
        return x

    # hierarchical square rooted group l1/2 regularization
    def hierarchical_square_rooted_group_l12_regularization(self, _lambda):
        x = 0
        for n, _module in self.model.named_modules():
            if isinstance(_module, nn.Conv2d) and (not 'downsample' in n):
                p = _module.weight
                number_of_out_channels = p.shape[0]
                number_of_in_channels = p.shape[1]
                p = p.reshape(p.shape[0]*p.shape[1], p.shape[2]*p.shape[3])
                p = torch.sum(torch.abs(p), 1)
                p = p.reshape(number_of_out_channels, number_of_in_channels)

                # hierarchical square rooted group l1/2 regularization based on the feature wise grouping
                x += _lambda*torch.sum(torch.sqrt(torch.sum(torch.sqrt(p), 0)))

                # hierarchical square rooted group l1/2 regularization based on the neuron wise grouping
                #x+= _lambda*torch.sum(torch.sqrt(torch.sum(torch.sqrt(p),1)))
        return x

    # hierarchical squared group l1/2 regularization
    def hierarchical_squared_group_l12_regularization(self, _lambda):
        x = 0
        for n, _module in self.model.named_modules():
            if isinstance(_module, nn.Conv2d) and (not 'downsample' in n):
                p = _module.weight
                number_of_out_channels = p.shape[0]
                number_of_in_channels = p.shape[1]
                p = p.reshape(p.shape[0]*p.shape[1], p.shape[2]*p.shape[3])
                p = torch.sum(torch.abs(p), 1)
                p = p.reshape(number_of_out_channels, number_of_in_channels)

                # hierarchical squared group l1/2 regularization based on the feature wise grouping
                x += _lambda*torch.sum((torch.sum(torch.sqrt(p), 0))**2)

                # hierarchical squared group l1/2 regularization based on the neuron wise grouping
                #x+= _lambda*torch.sum((torch.sum(torch.sqrt(p),1))**2)
        return x

