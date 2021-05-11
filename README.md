# DynamicOFA
This code implementation is for paper 'Dynamic-OFA: Runtime DNN Architecture Switching for Performance Scaling on Heterogeneous Embedded Platforms'.

> [**Dynamic-OFA: Runtime DNN Architecture Switching for Performance Scaling on Heterogeneous Embedded Platforms**](https://arxiv.org/abs/2105.03596),  
> Wei Lou, Lei Xun, Amin Sabet, Jia Bi, Jonathon Hare, Geoff V. Merrett 
> In: Embedded Computer vision (ECV) Workshop, Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR) , 2021  
> *arXiv preprint ([arXiv 2105.03596](https://arxiv.org/abs/2105.03596))*   

![Fig2](Fig2.png)


## Workflow of Dynamic-OFA
Using pre-trained OFA networks that contain 2*10^19 sub-network architectures as the backbone, sub-network architectures are sampled from OFA for both CPU and GPU at the offline stage. These architectures have different performance (\eg latency, accuracy) and are stored in a look-up table to build a dynamic version of OFA without any additional training required. Then, at runtime, Dynamic-OFA selects and switches to optimal sub-network architectures to fit time-varying available hardware resources.
