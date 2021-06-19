# Dynamic-OFA
Offical repo for paper 'Dynamic-OFA: Runtime DNN Architecture Switching for Performance Scaling on Heterogeneous Embedded Platforms'.

> [**Dynamic-OFA: Runtime DNN Architecture Switching for Performance Scaling on Heterogeneous Embedded Platforms**](https://arxiv.org/abs/2105.03596),  
> Wei Lou*, Lei Xun*, Amin Sabet, Jia Bi, Jonathon Hare, Geoff V. Merrett   
> In Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2021  
> *arXiv preprint ([arXiv 2105.03596](https://arxiv.org/abs/2105.03596))*   

![Fig2](Fig2.png)


## Workflow of Dynamic-OFA
Using pre-trained OFA networks that contain 2*10^19 sub-network architectures as the backbone, sub-network architectures are sampled from OFA for both CPU and GPU at the offline stage. These architectures have different performance (e.g. latency, accuracy) and are stored in a look-up table to build a dynamic version of OFA without any additional training required. Then, at runtime, Dynamic-OFA selects and switches to optimal sub-network architectures to fit time-varying available hardware resources.

## Usage
### Optimal Search
The optimal search process aims at searching for optimal sub-networks on the pareto curve from all the sub-networks of OFA model. 
This code can be used for different mobile devices. For different device, the accuracy tables and flop tables are the same which are restored in checkpoints repository the specilaized latency lable need to be built based on each device. And the search can be constrainted either by latency and FLOPs, only with different pre-calculated tables. After searching for certain number of sub-networks, evaluate the real-time latency and accuracy on your device. Then build a latency-accuracy scatter to find those points on the pareto curve.
### Searching
    For latency based search: 
        Search without accuracy constraint: python optimal_search/latency/search.py (The latency value is corresponding to evolution_finder.py)
        Search with accuracy constraint: python optimal_search/latency/search_accu.py (The latency value is corresponding to evolution_finder_accu.py)
    For flop based search:
        python optimal_search/flop/flop_search.py
    
### Evaluation
    python optimal_search/latency/evaluation.py & evaluation_cpu.py (For cpu usage please change the devices of all the function in imagenet_eval_helper.py)
    
### Examples of optimal search results
![Fig3](Fig3.png)

## Q&A
We are constantly improving the readability and useability of our codebase. Any feedback and questions for our paper and code are welcomed, please leave them as GitHub issues.
    
