#!/usr/bin/env python
# coding: utf-8

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

from accuracy_predictor import AccuracyPredictor
from latency_table import LatencyTable
from evolution_finder import EvolutionFinder
#from subnetfinder import EvolutionFinder
from imagenet_eval_helper import evaluate_ofa_subnet, evaluate_ofa_specialized
#os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
# set random seed
random_seed = 1
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
print('Successfully imported all packages and configured random seed to %d!'%random_seed)


cuda_available = torch.cuda.is_available()
if cuda_available:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(random_seed)
    print('Using GPU.')
else:
    print('Using CPU.')
    
# accuracy predictor
accuracy_predictor = AccuracyPredictor(
    pretrained=True,
)

print('The accuracy predictor is ready!')
print(accuracy_predictor.model)

target_hardware = 'jetson' #If using GPU version, set 'jetson_GPU'
latency_table = LatencyTable(device=target_hardware)
print('The Latency lookup table on %s is ready!' % target_hardware)

result_lis = []
for latency in range(820,1000,10):
    P = 100  # The size of population in each generation
    N = 500  # How many generations of population to be searched
    r = 0.25  # The ratio of networks that are used as parents for next generation
    params = {
        'constraint_type': target_hardware, # Let's do FLOPs-constrained search
        'efficiency_constraint': latency,
        'mutate_prob': 0.4, # The probability of mutation in evolutionary search
        'mutation_ratio': 0.5, # The ratio of networks that are generated through mutation in generation n >= 2.
        'efficiency_predictor': latency_table, # To use a predefined efficiency predictor.
        'accuracy_predictor': accuracy_predictor, # To use a predefined accuracy_predictor predictor.
        'population_size': P,
        'max_time_budget': N,
        'parent_ratio': r,
        }
    finder = EvolutionFinder(**params)
    # start searching    
    st = time.time()
    best_valids,best_info = finder.run_evolution_search()
    result_lis.append(best_info)
    ed = time.time()
    print(best_info)
import pickle

file=open(r"CPUresult_820_1000.pickle","wb")
pickle.dump(result_lis,file)
file.close()

