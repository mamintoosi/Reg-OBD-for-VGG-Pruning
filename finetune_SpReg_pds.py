import torch
from torch.autograd import Variable
from torchvision import models
# import cv2
# import sys
import numpy as np
# import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import dataset
from prune import *
import argparse
from operator import itemgetter
from heapq import nsmallest
import time
import matplotlib.pyplot as plt
from sparse_regularization import sparse_regularization
from torchvision import datasets, transforms
import random
import pickle as pkl
from pycm import *
# conda install -c sepandhaghighi pycm
import itertools
from google.colab import files
import os
import seaborn as sns
from collections import Counter
import pandas as pd
from tqdm.notebook import tqdm

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
            nn.Linear(4096, 10))

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
    def __init__(self, ds_name, model):
        # self.train_data_loader = dataset.loader(train_path)
        # self.test_data_loader = dataset.test_loader(test_path)
        # self.eval_data_loader = dataset.eval_loader(test_path,batch_size=1,num_workers=0)
        if ds_name == 'CIFAR10':
            trainset = datasets.CIFAR10(root=data_path,train=True,download=True,transform=transform_train)
            testset = datasets.CIFAR10(root=data_path,train=False,download=True,transform=transform_test)
            classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
            frac = 0.1
            batch_size = 8
        elif ds_name == 'MNIST':
            trainset = datasets.MNIST(root=data_path,train=True,download=True,transform=transform_train)
            testset = datasets.MNIST(root=data_path,train=False,download=True,transform=transform_test)
            classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
            frac = 0.1
            batch_size = 32
        elif ds_name == 'FashionMNIST':
            trainset = datasets.FashionMNIST(root=data_path,train=True,download=True,transform=transform_train)
            testset = datasets.FashionMNIST(root=data_path,train=False,download=True,transform=transform_test)
            classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
            frac = 0.1
            batch_size = 32
        elif ds_name == 'STL10':
            trainset = datasets.STL10(root=data_path,split='train',download=True,transform=transform_train)
            testset = datasets.STL10(root=data_path,split='test',download=True,transform=transform_test)
            classes = ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')
            frac = 0.5
            batch_size = 32
        # elif ds_name == 'LSUN':
        #     trainset = datasets.LSUN(root=data_path,classes='train',transform=transform_train) #download=True,
        #     testset = datasets.LSUN(root=data_path,classes='test',transform=transform_test)
        #     classes = ('bedroom', 'bridge', 'church_outdoor', 'classroom', 'conference_room', 'dining_room', 'kitchen', 'living_room', 'restaurant', 'tower')
        #     frac = 0.01
            
        # workers = 0 # for local
        workers = 2 # for colab

        # انتخاب زیرمجموعه‌ای کوچک‌تر برای سرعت بیشتر
        List = range(len(trainset))
        rnd_subset = random.sample(List, int(len(trainset)*frac)) 
        trainset_sub = torch.utils.data.Subset(trainset, rnd_subset)
        # print('Len TrainSet',len(trainset), len(trainset_sub))
        List = range(len(testset))
        rnd_subset = random.sample(List, int(len(testset)*frac)) 
        testset_sub = torch.utils.data.Subset(testset, rnd_subset)

        List = range(len(testset))
        rnd_subset = random.sample(List, int(len(testset)*frac)) 
        valset_sub = torch.utils.data.Subset(testset, rnd_subset)
    
        trainloader = torch.utils.data.DataLoader(trainset_sub,batch_size=batch_size,shuffle=True,num_workers=workers)
        testloader = torch.utils.data.DataLoader(testset_sub,batch_size=batch_size,shuffle=False, num_workers=workers)
        valloader = torch.utils.data.DataLoader(valset_sub,batch_size=batch_size,shuffle=False, num_workers=workers)

        self.train_data_loader = trainloader
        self.test_data_loader = testloader
        self.val_data_loader = valloader
        # self.eval_data_loader = testloader

        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.prunner = FilterPrunner(self.model) 
        self.model.train()

    def test(self, val_test_set = 'val'):
        # return
        # print("Test starts...")
        self.model.eval()
        correct = 0
        # incorrect = 0
        total = 0
        Labels = []
        Preds = []
        epoch_loss= []
        if val_test_set == 'val':
            test_loader = self.val_data_loader
        else: # Test set
            test_loader = self.test_data_loader

        # for i, (batch, label) in enumerate(self.test_data_loader):
        for i, (batch, label) in enumerate(test_loader):
            if args.use_cuda:
                batch = batch.cuda()
                label = label.cuda()

            output = self.model(Variable(batch))
            loss = self.criterion(output, Variable(label))
            epoch_loss.append(loss.item())

            pred = output.data.max(1)[1]
            correct += pred.cpu().eq(label.cpu()).sum()
            # incorrect += pred.cpu().ne(label).sum()
            total += label.size(0)
            # print(pred.cpu(), label)
            Preds.append(pred.cpu().tolist())
            Labels.append(label.cpu().tolist())
        
        acc = float(correct) / total
        print("Accuracy on ",val_test_set, ":{:.3f}".format(acc),"\t Loss:{:.3f}".format(sum(epoch_loss)/len(epoch_loss)))
        
        self.model.train()
        
        Labels = list(itertools.chain.from_iterable(Labels))
        Preds = list(itertools.chain.from_iterable(Preds))

        return Preds, Labels, epoch_loss

    def eval_test_results(self):
        # کدها رو حذف کردم. از کووید قابل برداشت است
        # به این دلیل که در اونجا از دیتالودر تست برای اعتبارسنجی و از 
        # eval
        # برای ارزیابی داده‌های آزمون استفاده کرده بودم. برای اینکه نامها مثل استفاده باشند عوض شدند.
        return

    def train(self, optimizer = None, epoches=10, regularization=None):
        if optimizer is None:
            # فقط پارامترهای طبفه‌بند
            optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001)
            # optimizer = optim.SGD(model.classifier.parameters(), lr=0.0001, momentum=0.9)

        Train_loss=  []
        Val_loss=  []
        min_val_loss = np.inf
        for i in tqdm(range(epoches), desc='Training', unit='Epoch'):
            # print("Epoch: ", i+1, '/', epoches)
            epoch_loss = self.train_epoch(optimizer,regularization=regularization)
            Train_loss.append(sum(epoch_loss)/len(epoch_loss))

            Preds, Labels, epoch_loss = self.test()
            val_loss = sum(epoch_loss)/len(epoch_loss)
            Val_loss.append(val_loss)

            # Save the best model
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                model_file_name = '{}{}.pt'.format(args.models_dir,args.output_model)
                torch.save(self.model, model_file_name)
                # print('Model Saved in epoch: ',i)

        print("Finished fine tuning.")
        return Train_loss, Val_loss
        

    def train_epoch(self, optimizer = None, rank_filters = False, regularization = None):
        epoch_loss= []
        for i, (batch, label) in enumerate(self.train_data_loader):
            loss = self.train_batch(optimizer, batch, label, rank_filters, regularization)
            epoch_loss.append(loss.item())
        return epoch_loss    

    def train_batch(self, optimizer, batch, label, rank_filters, regularization=None):

        if args.use_cuda:
            batch = batch.cuda()
            label = label.cuda()

        self.model.zero_grad()
        input = Variable(batch)

        if rank_filters:
            output = self.prunner.forward(input)
            loss = self.criterion(output, Variable(label))
            loss.backward()
        else:
            loss = self.criterion(self.model(input), Variable(label))
            if regularization is not None:
                # print('Using Regularization: ',reg_name)
                loss += args.landa*regularization(0.5)
            loss.backward()
            optimizer.step()
        return loss    

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
        num_filters_to_prune_per_iteration = args.num_f2ppi
        iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)
        # print(number_of_filters, num_filters_to_prune_per_iteration, iterations)
        iterations = int(iterations * args.prune_percent / 100.0) 
        print(iterations)
        print("Number of prunning iterations to reduce {}% filters: {}".format(args.prune_percent,iterations))

        dics = [] # A list for saving dics
        for i in range(iterations):
            print("Iter: ", i+1, '/', iterations)
            print("Ranking filters.. ")
            prune_targets = self.get_candidates_to_prune(num_filters_to_prune_per_iteration)
            layers_prunned = {}
            for layer_index, filter_index in prune_targets:
                if layer_index not in layers_prunned:
                    layers_prunned[layer_index] = 0
                layers_prunned[layer_index] = layers_prunned[layer_index] + 1 

            print("Layers that will be prunned", layers_prunned)

            # Add to list for future saving
            dics.append(layers_prunned)

            print("Prunning filters.. ")
            model = self.model.cpu()
            for layer_index, filter_index in prune_targets:
                model = prune_vgg16_conv_layer(model, layer_index, filter_index, use_cuda=args.use_cuda)

            self.model = model
            if args.use_cuda:
                self.model = self.model.cuda()

            message = str(100 - 100*float(self.total_num_filters()) / number_of_filters) + "%"
            print("Filters prunned", str(message))
            self.test()
            print("Fine tuning to recover from prunning iteration.")
            # همه پارامترها
            optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
            # optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
            if args.eat == 'dec':
                self.train(optimizer, epoches=iterations-i)
            else:
                self.train(optimizer, epoches=int(args.eat))


        # print("Finished. Going to fine tune the model a bit more")
        # self.train(optimizer, epoches=5) 
        # models_dir = 'models/'
        # torch.save(model.state_dict(), models_dir+"painting_model_prunned.pt")
        # torch.save(model, models_dir+"VGG_model_COVID19_prunned.pt")
        # model_file_name = '{}_prnIn_{}_reg-{}_pruned.pt'.format(args.models_dir, \
        #     args.prune_input,args.ds_name, args.reg_name)

        # model_file_name = '{}{}.pt'.format(args.models_dir,args.output_model)
        # torch.save(model, model_file_name)

        dic_file_name = '{}_{}_dic.pkl'.format(args.ds_name, args.output_model)
        pkl.dump(dics, open(dic_file_name, "wb" ) )


    def prune_reg(self):
        if args.reg_name is not None:
            device = torch.device("cuda" if args.use_cuda else "cpu") #
            regularization = sparse_regularization(self.model,device)
            # reg_name = 'HSQGL12'
            if reg_name == 'L2':
                regularizationFun = regularization.l2_regularization
            elif reg_name == 'L1':
                regularizationFun = regularization.l1_regularization
            elif reg_name == 'HSQGL12':
                regularizationFun = regularization.hierarchical_squared_group_l12_regularization
            print('Using Regularization: ',reg_name)
        else:
            regularizationFun = None

        #Get the accuracy before prunning
        self.test()
        self.model.train()

        #Make sure all the layers are trainable
        for param in self.model.features.parameters():
            param.requires_grad = True
        print("Retraining with regularization ... ")
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        # optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.train(optimizer, epoches = args.train_epoch, regularization = regularizationFun)

        # model_file_name = '{}_prnIn-{}_{}_reg-{}.pt'.format(args.models_dir, \
        #     args.prune_input,args.ds_name, args.reg_name)
 
        # model_file_name = '{}{}.pt'.format(args.models_dir,args.output_model)
        # torch.save(model, model_file_name)

def bar_filters_pruned(dic):
    filters_pruned = dict(Counter(dic[0])+Counter(dic[1])+Counter(dic[2])+\
                          Counter(dic[3])+Counter(dic[4]))
    sum_filters_pruned = 0
    dic_obd = {}
    for k,v in filters_pruned.items():
        dic_obd[map_layer_nums[k]] = v
        sum_filters_pruned += v
    # print(sum_filters_pruned)
    # print(dic_obd)
    # keys_obd = list(dic_obd.keys())
    # vals_obd = [float(dic_obd[k]) for k in keys_obd]
    # sns.barplot(x=keys_obd, y=vals_obd)
    return dic_obd

def num_parameters(model,eps):
    num_el = 0
    num_zeros = 0
    for n, _module in model.named_modules():
#         print(_module)
        if isinstance(_module, nn.Conv2d) or isinstance(_module, nn.Linear) and (not 'downsample' in n):
            w = torch.flatten(_module.weight)
            num_el += w.shape[0]
            num_zeros += torch.sum(torch.abs(w)<eps)
    ze = num_zeros.cpu().detach().numpy()
    return num_el, ze

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--prune", dest="prune", action="store_true")
    parser.add_argument("--test", dest="test", action="store_true")
    parser.add_argument("--train_path", type = str, default = "data/train")
    parser.add_argument("--test_path", type = str, default = "data/test")
    parser.add_argument('--use-cuda', action='store_true', default=False, help='Use NVIDIA GPU acceleration')    
    parser.add_argument('--reg_name', type = str, default = None)
    parser.add_argument('--ds_name', type = str, default = 'COVID-CT')
    parser.add_argument("--train_epoch", type = int, default = 15)
    parser.add_argument('--prune_input', type = str, default = 'vgg')
    parser.add_argument('--input_model', type = str, default = 'vgg')
    parser.add_argument('--output_model', type = str, default = 'taylor')
    parser.add_argument("--num_f2ppi", type = int, default = 512)
    parser.add_argument("--prune_percent", type = int, default = 70)
    parser.add_argument("--landa", type = float, default = 1e-8)
    # landa is used instead of lambda, which is a python keyword
    # Epochs After tuning: constant or decremental, const, dec
    parser.add_argument('--eat', type = str, default = '5') 
    parser.set_defaults(train=False)
    parser.set_defaults(prune=False)
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()

    return args

if __name__ == '__main__':
    # برای اجرای محلی 
    # __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    # در اجرای محلی پارامترهای رشته‌ای ارسالی به مین نباید داخل تک کوتیشن باشند

    map_layer_nums = {0:1,2:2,5:3,7:4,10:5,12:6,14:7,17:8,19:9,21:10,24:11,26:12,28:13}

    # global args 
    args = get_args()

    ds_name = args.ds_name #'cifar10'
    num_classes = 10

    data_path = '/content/data'

    if ds_name in ['CIFAR10']:
        transform_train = transforms.Compose([
                # transforms.RandomCrop(32,padding = 4),
                transforms.Resize([224,224]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2023, 0.1994, 0.2010)),
            ])
        transform_test = transforms.Compose([
                transforms.Resize([224,224]),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2023, 0.1994, 0.2010)),
                ])
    # https://github.com/kuangliu/pytorch-cifar/blob/master/main.py            
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),            

    if ds_name in ['STL10']:
        transform_train = transforms.Compose([
                # transforms.RandomCrop(32,padding = 4),
                transforms.Resize([224,224]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5),
                                    (0.5, 0.5, 0.5)),
            ])
        transform_test = transforms.Compose([
                transforms.Resize([224,224]),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5),
                                    (0.5, 0.5, 0.5)),
                ])

    # MNIST Coef from: https://www.programmersought.com/article/5163444351/
    if ds_name in ['MNIST']:
        transform_train = transforms.Compose([
                # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize([224,224]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307, 0.1307, 0.1307),
                                    (0.3081, 0.3081, 0.3081)),
            ])
        transform_test = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize([224,224]),
                transforms.ToTensor(),
                transforms.Normalize((0.1307, 0.1307, 0.1307),
                                    (0.3081, 0.3081, 0.3081)),
                ])
    if ds_name in ['FashionMNIST']:
        transform_train = transforms.Compose([
                # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize([224,224]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5),
                                    (0.5, 0.5, 0.5)),
            ])
        transform_test = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize([224,224]),
                transforms.ToTensor(),
                transforms.Normalize((0.1307, 0.1307, 0.1307),
                                    (0.3081, 0.3081, 0.3081)),
                ])

    args.models_dir = 'models/'
    reg_name = args.reg_name
    print(ds_name)

    if args.train:
        model = ModifiedVGG16Model()
    elif args.test or args.prune:
        model_file_name = '{}{}.pt'.format(args.models_dir, args.input_model)
        model = torch.load(model_file_name, map_location=lambda storage, loc: storage)
    
    if args.use_cuda:
        model = model.cuda()
        print('Using CUDA...')
    
    fine_tuner = PrunningFineTuner_VGG16(args.ds_name, model)

    if args.train:            
        Train_loss, Val_loss = fine_tuner.train(epoches=args.train_epoch)#, regularization=regularizationFun)
        # model_file_name = '{}{}.pt'.format(args.models_dir,args.output_model)
        # torch.save(model, model_file_name)
        loss_file_name = '{}_loss.pkl'.format(args.ds_name)
        with open(loss_file_name, 'wb') as f:
            pkl.dump((Train_loss, Val_loss), f)
        # files.download(loss_file_name)

    elif args.prune:
        if args.reg_name is None:
            fine_tuner.prune()
        else:
            fine_tuner.prune_reg()

    # Preds ,Labels, epoch_loss =  fine_tuner.test(val_test_set = 'val')
    Preds ,Labels, epoch_loss =  fine_tuner.test(val_test_set = 'test')
    cm = ConfusionMatrix(actual_vector=Labels, predict_vector=Preds) # Create CM From Data
    cm_file_name = '{}_{}_cm.pkl'.format(args.ds_name, args.output_model)
    pkl.dump(cm, open(cm_file_name, "wb" ) )
    # files.download(cm_file_name)

    # dic_file_name = '{}_{}_dic.pkl'.format(args.ds_name, args.output_model)
    # if os.path.exists(dic_file_name):
    #     files.download(dic_file_name)

    if args.test:
        cm.plot(cmap=plt.cm.Greens,normalized=False,number_label=True,plot_lib="seaborn")
        # print(cm)
        print('ACC={:.2f}'.format(cm.Overall_ACC))
        print("\t".join("{}:{:.2f}".format(k, v) for k, v in cm.AUC.items()))
        print('FNR:{:.2f}, FPR:{:.2f}'.format(cm.FNR_Macro, cm.FPR_Macro))
        print('TNR:{:.2f}, TPR:{:.2f}'.format(cm.TNR_Macro, cm.TPR_Macro))
        # cm.classes
        # cm.table

    # print(model)
