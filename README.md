Wasserstein GAN
===============

Code accompanying the paper ["Wasserstein GAN"](https://arxiv.org/abs/1701.07875)

##Prerequisites

- Computer with Linux or OSX
- [PyTorch](http://pytorch.org)
- For training, an NVIDIA GPU is strongly recommended for speed. CPU is supported but training is very slow.


##Reproducing LSUN experiments

**With DCGAN:**

```python
python main.py --dataset lsun --dataroot [lsun-train-folder] --cuda
```

**With MLP:**

```python
python main.py --mlp_G --ngf 512
```

More improved README in the works.
