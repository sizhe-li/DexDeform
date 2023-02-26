# DexDeform
Code and data for paper [DexDeform: Dexterous Deformable Object Manipulation with Human Demonstrations and Differentiable Physics](https://openreview.net/pdf?id=LIV7-_7pYPl) at ICLR 2023.

# Installation

```bash
conda env create -f environment.yml
pip install -e .
```

# Demo

Demos are provided inside folder `demos`.

- [Environment Loading] `demos/environment_loading.ipynb`
- [Trajectory Optimization] `demos/trajectory_optimization.ipynb`

# Implementation Details

- Our simulation backend supports full differentiability and communications with PyTorch modules.
- For optimal performance, the simulation backend is written in CUDA and implements PlasticineLab. 
- We provide python wrapper for the dexterous hand environment, located inside `hand.py`. 

# Acknowledgements

- Our physics simulation written based on [PlasticineLab](https://github.com/hzaskywalker/PlasticineLab).


# Citation

```bibtex
@inproceedings{
li2023dexdeform,
title={DexDeform: Dexterous Deformable Object Manipulation with Human Demonstrations and Differentiable Physics},
author={Sizhe Li and Zhiao Huang and Tao Chen and Tao Du and Hao Su and Joshua B. Tenenbaum and Chuang Gan},
booktitle={International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=LIV7-_7pYPl}
}
```
