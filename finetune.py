import torch
from torch.autograd import Variable
from torchvision import models
import cv2
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dataset
from prune import *
import argparse
from operator import itemgetter
from heapq import nsmallest
import time
import matplotlib.pyplot as plt

class ModifiedVGG16Model(torch.nn.Module):
    def __init__(self):
        super(ModifiedVGG16Model, self).__init__()

        model = models.vgg16(pretrained=True)
        self.features = model.features

        for param in self.features.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class FilterPrunner:
    def __init__(self, model):
        self.model = model
        self.reset()
    
    def reset(self):
        self.filter_ranks = {}

    def forward(self, x):
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}

        activation_index = 0
        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            x = module(x)
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                x.register_hook(self.compute_rank)
                self.activations.append(x)
                self.activation_to_layer[activation_index] = layer
                activation_index += 1

        return self.model.classifier(x.view(x.size(0), -1))

    def compute_rank(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]

        taylor = activation * grad
        # Get the average value for every filter, 
        # accross all the other dimensions
        taylor = taylor.mean(dim=(0, 2, 3)).data


        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = \
                torch.FloatTensor(activation.size(1)).zero_()

            if args.use_cuda:
                self.filter_ranks[activation_index] = self.filter_ranks[activation_index].cuda()

        self.filter_ranks[activation_index] += taylor
        self.grad_index += 1

    def lowest_ranking_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))

        return nsmallest(num, data, itemgetter(2))

    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i])
            v = v.cpu() / np.sqrt(torch.sum(v.cpu() * v.cpu())) #M.Amintoosi
            self.filter_ranks[i] = v#.cpu()

    def get_prunning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)

        # After each of the k filters are prunned,
        # the filter index of the next filters change since the model is smaller.
        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)

        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
            for i in range(len(filters_to_prune_per_layer[l])):
                filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

        filters_to_prune = []
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))

        return filters_to_prune             

class PrunningFineTuner_VGG16:
    def __init__(self, train_path, test_path, model):
        self.train_data_loader = dataset.loader(train_path)
        self.test_data_loader = dataset.test_loader(test_path)
        self.eval_data_loader = dataset.eval_loader(test_path,batch_size=1,num_workers=0)

        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.prunner = FilterPrunner(self.model) 
        self.model.train()

    def test(self):
        # return
        # print("Test starts...")
        self.model.eval()
        correct = 0
        total = 0

        for i, (batch, label) in enumerate(self.test_data_loader):
            if args.use_cuda:
                batch = batch.cuda()
            output = model(Variable(batch))
            pred = output.data.max(1)[1]
            correct += pred.cpu().eq(label).sum()
            total += label.size(0)
        
        print("Accuracy :", float(correct) / total)
        
        self.model.train()

    def eval_test_results(self):
        class_names = ['COVID','NonCOVID']
        plt.figure(figsize=(25,10))
        count = np.zeros(2)

        self.model.eval()
        correct = 0
        total = 0

        for i, (batch, label) in enumerate(self.eval_data_loader):
            if args.use_cuda:
                batch = batch.cuda()
            # print(pred.cpu().eq(label).sum())
            if count[label]>=5:
                continue
            output = model(Variable(batch))
            pred = output.data.max(1)[1]
            correct += pred.cpu().eq(label).sum()
            isPredCorrect = pred.cpu().eq(label)
            total += label.size(0)
            count[label]+=1
            plt.subplot(2,5,5*label+count[label])
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(np.transpose(batch[0].numpy(), (1, 2, 0)))#, cmap=plt.cm.binary)
            if isPredCorrect==True:
                color = 'blue'
            else:
                color = 'red'
            # prob = yHat[0][0] if predicted_label==1 else (1-yHat[0][0])
            # plt.xlabel("({:.2f}){}".format(prob,make_farsi_text(class_names[predicted_label])),color=color, fontsize=35)
            plt.xlabel("{}".format(class_names[pred]),color=color, fontsize=25)
        plt.show()        
        print("Accuracy :", float(correct) / total)
        
        self.model.train()

    def train(self, optimizer = None, epoches=10):
        if optimizer is None:
            optimizer = optim.SGD(model.classifier.parameters(), lr=0.0001, momentum=0.9)

        for i in range(epoches):
            print("Epoch: ", i, '/', epoches)
            self.train_epoch(optimizer)
            self.test()
        print("Finished fine tuning.")
        

    def train_batch(self, optimizer, batch, label, rank_filters):

        if args.use_cuda:
            batch = batch.cuda()
            label = label.cuda()

        self.model.zero_grad()
        input = Variable(batch)

        if rank_filters:
            output = self.prunner.forward(input)
            self.criterion(output, Variable(label)).backward()
        else:
            self.criterion(self.model(input), Variable(label)).backward()
            optimizer.step()

    def train_epoch(self, optimizer = None, rank_filters = False):
        for i, (batch, label) in enumerate(self.train_data_loader):
            self.train_batch(optimizer, batch, label, rank_filters)

    def get_candidates_to_prune(self, num_filters_to_prune):
        self.prunner.reset()
        self.train_epoch(rank_filters = True)
        self.prunner.normalize_ranks_per_layer()
        return self.prunner.get_prunning_plan(num_filters_to_prune)
        
    def total_num_filters(self):
        filters = 0
        for name, module in self.model.features._modules.items():
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                filters = filters + module.out_channels
        return filters

    def prune(self):
        #Get the accuracy before prunning
        self.test()
        self.model.train()

        #Make sure all the layers are trainable
        for param in self.model.features.parameters():
            param.requires_grad = True

        number_of_filters = self.total_num_filters()
        num_filters_to_prune_per_iteration = 512
        iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)

        iterations = int(iterations * 9.5 / 10) # M.Amintoosi

        print("Number of prunning iterations to reduce 95% filters", iterations)

        for i in range(iterations):
            print("Iter: ", i, '/', iterations)
            print("Ranking filters.. ")
            prune_targets = self.get_candidates_to_prune(num_filters_to_prune_per_iteration)
            layers_prunned = {}
            for layer_index, filter_index in prune_targets:
                if layer_index not in layers_prunned:
                    layers_prunned[layer_index] = 0
                layers_prunned[layer_index] = layers_prunned[layer_index] + 1 

            print("Layers that will be prunned", layers_prunned)
            print("Prunning filters.. ")
            model = self.model.cpu()
            for layer_index, filter_index in prune_targets:
                model = prune_vgg16_conv_layer(model, layer_index, filter_index, use_cuda=args.use_cuda)

            self.model = model
            if args.use_cuda:
                self.model = self.model.cuda()

            # total_number_of_filters += 
            message = str(100 - 100*float(self.total_num_filters()) / number_of_filters) + "%"
            print("Filters prunned", str(message))
            self.test()
            print("Fine tuning to recover from prunning iteration.")
            optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
            self.train(optimizer, epoches = 10)


        # print("Finished. Going to fine tune the model a bit more")
        # self.train(optimizer, epoches=5) # M.Amintoosi 15->3
        models_dir = 'models/'
        # torch.save(model.state_dict(), models_dir+"painting_model_prunned.pt")
        torch.save(model, models_dir+"VGG_model_COVID19_prunned.pt")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--prune", dest="prune", action="store_true")
    parser.add_argument("--test", dest="test", action="store_true")
    parser.add_argument("--train_path", type = str, default = "train")
    parser.add_argument("--test_path", type = str, default = "test")
    parser.add_argument('--use-cuda', action='store_true', default=False, help='Use NVIDIA GPU acceleration')    
    parser.set_defaults(train=False)
    parser.set_defaults(prune=False)
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()

    return args

if __name__ == '__main__':
    args = get_args()

    if args.train:
        model = ModifiedVGG16Model()
    elif args.prune:
        models_dir = 'models/'
        model = torch.load(models_dir+"VGG_model_COVID19.pt", map_location=lambda storage, loc: storage)
    elif args.test:
        models_dir = 'models/'
        model = torch.load(models_dir+"VGG_model_COVID19_prunned.pt", map_location=lambda storage, loc: storage)
    
    if args.use_cuda:
        model = model.cuda()
        print('Using CUDA...')

    fine_tuner = PrunningFineTuner_VGG16(args.train_path, args.test_path, model)

    if args.train:
        fine_tuner.train(epoches=40)
        models_dir = 'models/'
        torch.save(model, models_dir+"VGG_model_COVID19.pt")
    elif args.prune:
        fine_tuner.prune()
    elif args.test:
        fine_tuner.eval_test_results()

    # print(model)
