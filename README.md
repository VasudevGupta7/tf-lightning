# tf_lightning

`tf_lightning` is a very simple, light-weight wrapper over `tensorflow2`, aims to train tf2 models with much less boiler-plate code and in a very structurized fashion. 

## Supported Features

It's currently supporting simple features for training models. I will add more features in this project as and when I need them.

- [x] Single GPU training with much less boiler-plate code
- [x] Mixed precision based training
- [ ] Distributed Training over multiple GPU's in single machine
- [ ] Gradient Accumulation
- [x] Wandb integration
- [x] Tensorboard integration

## Installation

```Python
# install tensorflow-2.4.0 first
pip install tensorflow==2.4.0

# Run this command to install tf_lightning
pip install git+https://github.com/VasudevGupta7/tf-lightning.git@master
```

## Contributions

**Feel free to fork this repositary and contribute to this project. Make pull request only in `master` branch.**
