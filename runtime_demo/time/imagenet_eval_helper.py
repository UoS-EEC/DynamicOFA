import os
import os.path as osp
import argparse
import numpy as np
import math
from tqdm import tqdm
import time
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms, datasets

from ofa.utils import accuracy
from ofa.model_zoo import ofa_net
from ofa.model_zoo import ofa_specialized
from ofa.elastic_nn.utils import set_running_statistics
import pickle

def evaluate_ofa_subnet(subnet, path,  net_config, data_loader, batch_size,device='cuda:0'):


    top1,latency = validate(subnet, path, net_config['r'][0], data_loader, batch_size, device='cuda:0')
    return top1,latency




def calib_bn(net,data_loader,bn_mean,bn_var):
    print('Creating dataloader for resetting BN running statistics...')
    set_running_statistics(net, data_loader,bn_mean,bn_var)

def validate(net, path, image_size, data_loader, batch_size=100, device='cuda:0'):


    data_loader.dataset.transform = transforms.Compose([
        transforms.Resize(int(math.ceil(image_size / 0.875))),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss().to(device)

    net.eval()
    #st = time.time()
    net = net.to(device)
    #ed = time.time()
    #print('To device',ed-st)
    #batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(data_loader),
        [losses, top1, top5],
        prefix='Test: ')
    latency = []
    record = []
    print('Start inference')
    
    with torch.no_grad():
        #start = time.time()
        #with tqdm(total=len(data_loader), desc='Validate') as t:
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
                # compute output
                #torch.cuda.synchronize()
            
            #end = time.time()
            #print('Dataloader time: ',end-start)
            st = time.time()
                
            output = net(images)
                #torch.cuda.synchronize()
            ed = time.time()
            
            print('latency: ',ed-st)
            
            '''
            if i == 250:
                return top1.avg,latency/200
            if i > 10:
                record.append(ed-st)
                print('The average latency of last 10 batches',(sum(record[-10:])/10)*1000)
            #print('The latency of this batch',ed-st)
            '''
            output = output.view(-1,1000)
                

            loss = criterion(output, labels)
                # measure accuracy and record loss
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                

            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))
            #latency += time.time() - end
            #print('The latency of this batch',time.time() - end)

            #end = time.time()
            #if i % 10 == 0:
                #progress.display(i)

    
    
    print('Results: loss=%.5f,\t top1=%.1f,\t top5=%.1f' % (losses.avg, top1.avg, top5.avg))
    return top1.avg,np.mean(latency)
    

def evaluate_ofa_specialized(path, data_loader, batch_size=100, device='cpu'):
    def select_platform_name():
        valid_platform_name = [
            'pixel1', 'pixel2', 'note10', 'note8', 's7edge', 'lg-g8', '1080ti', 'v100', 'tx2', 'cpu', 'flops'
        ]
        
        print("Please select a hardware platform from ('pixel1', 'pixel2', 'note10', 'note8', 's7edge', 'lg-g8', '1080ti', 'v100', 'tx2', 'cpu', 'flops')!\n")
        
        while True:
            platform_name = input()
            platform_name = platform_name.lower()
            if platform_name in valid_platform_name:
                return platform_name
            print("Platform name is invalid! Please select in ('pixel1', 'pixel2', 'note10', 'note8', 's7edge', 'lg-g8', '1080ti', 'v100', 'tx2', 'cpu', 'flops')!\n")
    
    def select_netid(platform_name):
        platform_efficiency_map = {
            'pixel1': {
                143: 'pixel1_lat@143ms_top1@80.1_finetune@75',
                132: 'pixel1_lat@132ms_top1@79.8_finetune@75',
                79: 'pixel1_lat@79ms_top1@78.7_finetune@75',
                58: 'pixel1_lat@58ms_top1@76.9_finetune@75',
                40: 'pixel1_lat@40ms_top1@74.9_finetune@25',
                28: 'pixel1_lat@28ms_top1@73.3_finetune@25',
                20: 'pixel1_lat@20ms_top1@71.4_finetune@25',
            },
            
            'pixel2': {
                62: 'pixel2_lat@62ms_top1@75.8_finetune@25',
                50: 'pixel2_lat@50ms_top1@74.7_finetune@25',
                35: 'pixel2_lat@35ms_top1@73.4_finetune@25',
                25: 'pixel2_lat@25ms_top1@71.5_finetune@25',
            },
            
            'note10': {
                64: 'note10_lat@64ms_top1@80.2_finetune@75',
                50: 'note10_lat@50ms_top1@79.7_finetune@75',
                41: 'note10_lat@41ms_top1@79.3_finetune@75',
                30: 'note10_lat@30ms_top1@78.4_finetune@75',
                22: 'note10_lat@22ms_top1@76.6_finetune@25',
                16: 'note10_lat@16ms_top1@75.5_finetune@25',
                11: 'note10_lat@11ms_top1@73.6_finetune@25',
                8: 'note10_lat@8ms_top1@71.4_finetune@25',
            },
            
            'note8': {
                65: 'note8_lat@65ms_top1@76.1_finetune@25',
                49: 'note8_lat@49ms_top1@74.9_finetune@25',
                31: 'note8_lat@31ms_top1@72.8_finetune@25',
                22: 'note8_lat@22ms_top1@70.4_finetune@25',
            },
            
            's7edge': {
                88: 's7edge_lat@88ms_top1@76.3_finetune@25',
                58: 's7edge_lat@58ms_top1@74.7_finetune@25',
                41: 's7edge_lat@41ms_top1@73.1_finetune@25',
                29: 's7edge_lat@29ms_top1@70.5_finetune@25',
            },
            
            'lg-g8': {
                24: 'LG-G8_lat@24ms_top1@76.4_finetune@25',
                16: 'LG-G8_lat@16ms_top1@74.7_finetune@25',
                11: 'LG-G8_lat@11ms_top1@73.0_finetune@25',
                8: 'LG-G8_lat@8ms_top1@71.1_finetune@25',
            },
            
            '1080ti': {
                27: '1080ti_gpu64@27ms_top1@76.4_finetune@25',
                22: '1080ti_gpu64@22ms_top1@75.3_finetune@25',
                15: '1080ti_gpu64@15ms_top1@73.8_finetune@25',
                12: '1080ti_gpu64@12ms_top1@72.6_finetune@25',
            },
            
            'v100': {
                11: 'v100_gpu64@11ms_top1@76.1_finetune@25',
                9: 'v100_gpu64@9ms_top1@75.3_finetune@25',
                6: 'v100_gpu64@6ms_top1@73.0_finetune@25',
                5: 'v100_gpu64@5ms_top1@71.6_finetune@25',
            },
            
            'tx2': {
                96: 'tx2_gpu16@96ms_top1@75.8_finetune@25',
                80: 'tx2_gpu16@80ms_top1@75.4_finetune@25',
                47: 'tx2_gpu16@47ms_top1@72.9_finetune@25',
                35: 'tx2_gpu16@35ms_top1@70.3_finetune@25',
            },
            
            'cpu': {
                17: 'cpu_lat@17ms_top1@75.7_finetune@25',
                15: 'cpu_lat@15ms_top1@74.6_finetune@25',
                11: 'cpu_lat@11ms_top1@72.0_finetune@25',
                10: 'cpu_lat@10ms_top1@71.1_finetune@25',
            },
            
            'flops': {
                595: 'flops@595M_top1@80.0_finetune@75',
                482: 'flops@482M_top1@79.6_finetune@75',
                389: 'flops@389M_top1@79.1_finetune@75',
            }
        }
        
        sub_efficiency_map = platform_efficiency_map[platform_name]
        if not platform_name == 'flops':
            print("Now, please specify a latency constraint for model specialization among", sorted(list(sub_efficiency_map.keys())), 'ms. (Please just input the number.) \n')
        else:
            print("Now, please specify a FLOPs constraint for model specialization among", sorted(list(sub_efficiency_map.keys())), 'MFLOPs. (Please just input the number.) \n')
        
        while True:
            efficiency_constraint = input()
            if not efficiency_constraint.isdigit():
                print('Sorry, please input an integer! \n')
                continue
            efficiency_constraint = int(efficiency_constraint)
            if not efficiency_constraint in sub_efficiency_map.keys():
                print('Sorry, please choose a value from: ', sorted(list(sub_efficiency_map.keys())), '.\n')
                continue
            return sub_efficiency_map[efficiency_constraint]

    platform_name = select_platform_name()
    net_id = select_netid(platform_name)

    net, image_size = ofa_specialized(net_id=net_id, pretrained=True)

    validate(net, path, image_size, data_loader, batch_size, device)

    return net_id
# statistic averaging
#
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
