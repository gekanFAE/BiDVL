# Official PyTorch implementation of "Bi-level Double Variational Learning for Energy-based Latent Variable Models" [(CVPR 2022 Paper)](https://arxiv.org/abs/2203.14702)

## Requirements:

Make sure the following environments are installed.

```
tensorflow-gpu=1.14.0
torchvision=0.4.0
pytorch=1.2.0
scipy=1.1.0
scikit-learn=0.21.2
Pillow=6.2.0
matplotlib=3.1.1
seaborn=0.9.0
```
We test the code on Unbuntu with RTX 2080ti. Other platforms may/may not have numerical instablities. Notice the FID&MSE here is stable while the EBM here is relatively less, which is similar to the adversarial training of GANs, and we will resort to some other tricks to solve it in future works. 


## Training on CIFAR-10:

```bash
$ python train_cifar_BiDVL.py
```

## Training on CelebA-64:

```bash
$ python train_celeba_BiDVL.py
```

Please refer to the python file for optimal training parameters.


## Bibtex ##
Cite our paper using the following bibtex item:

```
@article{Kan2022BilevelDV,
  title={Bi-level Doubly Variational Learning for Energy-based Latent Variable Models},
  author={Ge Kan and Jinhu Lu and Tian Wang and Baochang Zhang and Aichun Zhu and Lei Huang and Guodong Guo and Hichem Snoussi},
  journal={ArXiv},
  year={2022},
  volume={abs/2203.14702}
}
```