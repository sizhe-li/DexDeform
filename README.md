# DexDeform
Code and data for paper [DexDeform: Dexterous Deformable Object Manipulation with Human Demonstrations and Differentiable Physics](https://openreview.net/pdf?id=LIV7-_7pYPl) at ICLR 2023.

![Alt Text](https://github.com/lester0866/DexDeform/blob/main/misc/flip.gif)

# Installation

```bash
conda env create -f environment.yml
conda activate dexdeform
pip install -e .
```

# Download Demonstrations

Download [here](https://drive.google.com/drive/folders/1xVS9ui5eHVCBFvmIAQ_mRqacEj-0__Hr?usp=sharing). For loading demonstrations, checkout `tutorials/demonstration_loading.ipynb`.

# Tutorials

- [Environment Loading] `tutorials/environment_loading.ipynb`
- [Trajectory Optimization] `tutorials/trajectory_optimization.ipynb`
- [Leap motion tracking module] `leap_motion/`
- [Demonstration Loading] `tutorials/demonstration_loading.ipynb`

# Implementation Details

- Our simulation backend supports full differentiability and communications with PyTorch modules.
- For optimal performance, the simulation backend is written in CUDA and implements PlasticineLab. 
- We provide python wrapper for the dexterous hand environment, located inside `hand.py`. 

# Acknowledgements

- Our physics simulation is written based on [PlasticineLab](https://github.com/hzaskywalker/PlasticineLab).
- Our leap motion tracking module is written based on [this repo](https://github.com/szahlner/shadow-teleop/tree/main/leap_motion).


# TODO
- [x] Support for Human Teleoperation (Leap motion tracking module released, synchronization with simulation coming soon)
- [x] Release demonstrations
- [ ] Support for DexDeform Algorithm (coming soon)

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
