# ARS-Aug
Augmented Random Search for Data Augmentation

**Policy Found on CIFAR-10 and CIFAR-100:**

      [('Solarize', 0.66, 0.34), ('Equalize', 0.56, 0.21)],
      [('Equalize', 0.43, 0.76), ('AutoContrast', 0.66, 0.98)],
      [('Color', 0.72, 0.47), ('Contrast', 0.88, 0.86)],
      [('Brightness', 0.84, 0.71), ('Color', 0.31, 0.74)],
      [('Rotate', 0.68, 0.26), ('TranlateX', 0.38, 0.88)]]
 
      [('TranslateY', 0.88, 0.96), ('TranslateY', 0.53, 0.79)],
      [('AutoContrast', 0.44, 0.76), ('Solarize', 0.22, 0.48)],
      [('AutoContrast', 0.93, 0.62), ('Solarize', 0.85, 0.26)],
      [('Solarize', 0.55, 0.38), ('Equalize', 0.43, 0.68)],
      [('TranslateY', 0.72, 0.93), ('AutoContrast', 0.83, 0.95)]]
 
      [('Solarize', 0.43, 0.58), ('AutoContrast', 0.82, 0.26)],
      [('TranslateY', 0.71, 0.79), ('AutoContrast', 0.81, 0.94)],
      [('AutoContrast', 0.92, 0.18), ('TranslateY', 0.77, 0.85)],
      [('Equalize', 0.71, 0.69), ('Color', 0.23, 0.33)],
      [('Sharpness', 0.36, 0.98), ('Brightness', 0.72, 0.78)]]
  
      [('Equalize', 0.74, 0.49), ('TranslateY', 0.86, 0.91)],
      [('TranslateY', 0.82, 0.91), ('TranslateY', 0.96, 0.79)],
      [('AutoContrast', 0.53, 0.37), ('Solarize', 0.39, 0.47)],
      [('TranslateY', 0.22, 0.78), ('Color', 0.91, 0.65)],
      [('Brightness', 0.82, 0.46), ('Color', 0.23, 0.91)]]
  
      [('Cutout', 0.27, 0.45), ('Equalize', 0.37, 0.21)],
      [('Color', 0.43, 0.23), ('Brightness', 0.65, 0.71)],
      [('ShearX', 0.49, 0.31), ('AutoContrast', 0.92, 0.28)],
      [('Equalize', 0.62, 0.59), ('Equalize', 0.38, 0.91)],
      [('Solarize', 0.57, 0.31), ('Equalize', 0.61, 0.51)]]
      
**Require:**

Tensorflow, Python3

**How to run:** (Follow Autoaugment: Learning Augmentation Policies from Data
https://arxiv.org/abs/1805.09501)
```
curl -o cifar-10-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

curl -o cifar-100-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
```

```
python train_cifar.py --model_name=wrn \
                      --checkpoint_dir=/tmp/training \
                      --data_path=/tmp/data \
                      --dataset='cifar10' \
                      --use_cpu=0
```

**If you find this help your research, please cite:**

```
@article{geng2018learning, title={Learning data augmentation policies using augmented random search}, author={Geng, Mingyang and Xu, Kele and Ding, Bo and Wang, Huaimin and Zhang, Lei}, journal={arXiv preprint arXiv:1811.04768}, year={2018} }
```
