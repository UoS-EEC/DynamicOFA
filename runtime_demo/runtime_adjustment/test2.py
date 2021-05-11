# In[ ]:

import sys
import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import numpy as np
import time
import random
import math
import copy
from matplotlib import pyplot as plt
import pickle
from ofa.model_zoo import ofa_net
from ofa.utils import download_url
import psutil
#from subnetfinder import EvolutionFinder
from imagenet_eval_helper import evaluate_ofa_subnet, evaluate_ofa_specialized
#os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
# set random seed
def picknet(level,constraint,accuracy):
    #print(level_list)
    #if int(level_pre_pre) < int(level_pre):
        #pid = os.getpid()
        #label = 2
        #return str(label)+str(pid)
    accu_list = [74.202,  74.968,  76.134,  77.54 ,  78.346,  79.03 ]

    indexs = []
    for accu in accu_list:
        if accu >= int(accuracy):
            indexs.append(accu_list.index(accu))
    #print(indexs)
    if level not in indexs:
        f = open('judge.txt','w')
        f.write('True')
        f.close()
    else:
        f = open('judge.txt','w')
        f.write('False')
        f.close()
    usage1 = psutil.virtual_memory().percent
    level = int(level)
    constraint = int(constraint)
    random_seed = 1
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    #print('Successfully imported all packages and configured random seed to %d!'%random_seed)
    ofa_network = ofa_net('ofa_mbv3_d234_e346_k357_w1.2', pretrained=True)
    #print('The OFA Network is ready.')
    
    cuda_available = torch.cuda.is_available()

    imagenet_data_path = '/home/lewislou/data/testing'
#imagenet_data_path = '/home/lewislou/data/imagenet'
    def build_val_transform(size):
        return transforms.Compose([
            transforms.Resize(int(math.ceil(size / 0.875))),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            root=os.path.join(imagenet_data_path, 'validations'),
            transform=build_val_transform(224)
            ),
        batch_size=1,  # test batch size
        shuffle=False,
        #num_workers=16,  # number of workers for the data loader
        pin_memory=False,
        drop_last=False,
        )
    #print('The ImageNet dataloader is ready.')
    data_loader.dataset
    mylist=[]
    file=open(r"nets_6.pickle","rb")
    mylist=pickle.load(file) #读取文件到list\
    #result=[]
    #file=open('bn'+str(level)+'.pickle',"rb")
    #result=pickle.load(file) #读取文件到list
    bn_list = mylist['bn_lists'][level]
    top1s = []
    latency_list = []
    init_latencys = []
    net_config = mylist['configs'][level]
    point = time.time()
    f = open('latencylog.txt','a')
    f.write('Change to level '+str(level)+'at time '+str(point)+'\n')
    f.close()
    #f = open('usage.txt','a')
    #f.write(str(usage1)+'\n')
    #f.close()
    #print(net_config)
    top1,latency,init_latency= evaluate_ofa_subnet(
        ofa_network,
        imagenet_data_path,
        net_config,
        data_loader,
        batch_size=16,
        bn_list = bn_list[0],
        usage = usage1,
        maximum = constraint)
    if top1 == 0:
        pid = os.getpid()
        label = 0
        return str(label)+str(pid)
    elif top1 == 1:
        pid = os.getpid()
        label = 1
        return str(label)+str(pid)
    else:
        return 1111

