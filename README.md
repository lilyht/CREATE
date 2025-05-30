# Reducing Class-wise Confusion for Incremental Learning with Disentangled Manifolds
<p align="center">Huitong Chen, Yu Wang, Yan Fan, Guosong Jiang, Qinghua Hu</p>
<p align="center">Tianjin Key Lab of Machine Learning, College of Intelligence and Computing, Tianjin University, China</p>
<p align="center">Haihe Laboratory of Information Technology Application Innovation (Haihe Lab of ITAI), Tianjin, China</p>
<p align="center">CVPR 2025</p>

This is the official PyTorch implementation of [Create](https://arxiv.org/abs/2503.17677).

## Abstract
Class incremental learning (CIL) aims to enable models to continuously learn new classes without catastrophically forgetting old ones. A promising direction is to learn and use prototypes of classes during incremental updates. Despite simplicity and intuition, we find that such methods suffer from inadequate representation capability and unsatisfied feature overlap. These two factors cause class-wise confusion and limited performance. In this paper, we develop a Confusion-REduced AuTo-Encoder classifier (CREATE) for CIL. Specifically, our method employs a lightweight auto-encoder module to learn compact manifold for each class in the latent subspace, constraining samples to be well reconstructed only on the semantically correct auto-encoder. Thus, the representation stability and capability of class distributions are enhanced, alleviating the potential class-wise confusion problem. To further distinguish the overlapped features, we propose a confusion-aware latent space separation loss that ensures samples are closely distributed in their corresponding low-dimensional manifold while keeping away from the distributions of features from other classes. Our method demonstrates stronger representational capacity and discrimination ability by learning disentangled manifolds and reduces class confusion. Extensive experiments on multiple datasets and settings show that CREATE outperforms other state-of-the-art methods up to $5.41$%. 

![](./model.png)

## Dependencies
+ torch 2.0.1
+ torchvision 0.15.2
+ CUDA 12.2
+ Python 3.9.21

## Run experiments
```
python main.py --config=./exps/[MODEL]_[DATASET]_b[BASE]_inc[INC].json
```

## Acknowledgments
We thank the following repos providing helpful components/functions in our work.
+ [PyCIL](https://github.com/G-U-N/PyCIL)
+ [BEEF](https://github.com/G-U-N/ICLR23-BEEF)
+ [CSSR](https://github.com/xyzedd/CSSR)

