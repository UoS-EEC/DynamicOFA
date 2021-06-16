# In[ ]:


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

from ofa.model_zoo import ofa_net
from ofa.utils import download_url

#from subnetfinder import EvolutionFinder
from imagenet_eval_helper import evaluate_ofa_subnet, evaluate_ofa_specialized
#os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
# set random seed
random_seed = 1
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
print('Successfully imported all packages and configured random seed to %d!'%random_seed)
ofa_network = ofa_net('ofa_mbv3_d234_e346_k357_w1.2', pretrained=True)
print('The OFA Network is ready.')

cuda_available = torch.cuda.is_available()
if cuda_available:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(random_seed)
    print('Using GPU.')
else:
    print('Using CPU.')

imagenet_data_path = os.path.join(os.path.abspath(os.getcwd()), 'data') #You need creat a folder named data in DynamicOFA-master folder
#'/home/~/data/testing' ## change to your data path
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
    batch_size=8,  # test batch size
    shuffle=False,
    #num_workers=16,  # number of workers for the data loader
    pin_memory=False,
    drop_last=False,
    )

print('The ImageNet dataloader is ready.')
data_loader.dataset
import pickle
mylist=[]
file=open(r"GPUflopresult_300_500.pickle","rb") # The subnetwork config file
mylist=pickle.load(file)
top1s = []
latency_list = []
init_latencys = []
bn_lists = []
j = 0 ##init number
for result in mylist:
    j += 1
    if j > 0:
        _, net_config, flop = result
        top1,latency,init_latency,bn_list= evaluate_ofa_subnet(
            ofa_network,
            imagenet_data_path,
            net_config,
            data_loader,
            batch_size=16)
        print('top1',top1)
        print('init_latency and inference latency',init_latency,sum(latency)/len(latency))

        print('Number',j)
        with open('flop_acc'+str(j)+'.pickle','wb') as file:
            pickle.dump(
            {
                'top1':top1,
                'flop':flop,
                'init_latency':init_latency,
                'bn_lists':bn_list,
                'configs':net_config
            },
            file,pickle.HIGHEST_PROTOCOL)
