# tf_lightning

`tf_lightning` is a very simple light-weight wrapper over `tensorflow2`, aims to train tf2 models with much less boiler-plate code and in a very structurized fashion. Its very flexible and one can train any model from simple RNNs to GANs.

## Supported Features

It's currently supporting simple features for training models. I will add more features in this project in my free time.

- [x] Single GPU training with much less code
- [ ] Distributed Training over multiple GPU's in single machine
- [ ] Gradient Accumulation
- [ ] Mixed precision based training

## Installation

```Python
# install tensorflow-2.3 first
pip install tensorflow==2.3

# Run this command to install tf_lightning
pip install git+https://github.com/VasudevGupta7/tf-lightning.git@master
```

## Contributions

Feel free to fork this repositary and contribute to this project. Make sure you make pull request only in `magik` branch.