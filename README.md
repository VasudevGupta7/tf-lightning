# tf_lightning

This is `tf_lightning`, a very simple-small light-weight wrapper over `tensorflow2`, aims to train tf2 models with much less boiler-plate code and in a very structurized fashion. 

## Supported Features

It's currently supporting simple features for training models. I will add more features in this project as and when I need them.

- [x] Single GPU training with much less code
- [x] Mixed precision based training
- [ ] Distributed Training over multiple GPU's in single machine
- [ ] Gradient Accumulation
- [x] Wandb integration
- [ ] Tensorboard integration

## Installation

This small package, I designed for my personal use mainly. But if you are interested in using it, feel free to raise an issue; I will make docs regarding using it.

```Python
# install tensorflow-2.3 first
pip install tensorflow==2.3

# Run this command to install tf_lightning
pip install git+https://github.com/VasudevGupta7/tf-lightning.git@master
```

## Contributions

Feel free to fork this repositary and contribute to this project. Make sure you make pull request only in `magik` branch.