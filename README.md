# PyTorch Template

This project is a customizable experimental framework for Pytorch aimed to reduce the amount of code needed to run new experiments and gather measurements.
The goal is to have a general configuration file from where we can control the model to be trained, the data, the loss function, the metrics used during training and select different loggers which will save the data.

The main modules are:

1. Data.
The data module consists of data loaders and datasets, both subclasses of pytorch DataLoader and Dataset, respectively.

These modules allow to easily add new datasets and create data loaders which can be used during training or testing.

In particular, the datasets folder hosts custom datasets, which are not available through pytorch.

2. Model.
The model module consists of different model architectures, loss functions and metrics used to measure performance.
The architectures are based on building blocks, which can be found in the 'blocks' folder.

Defining new architectures is straightforward following the ResNet example in the repo.

Similarly, defining new loss functions or new metrics is trivial, following the examples in the repo.

3. Trainer.
A trainer runs the model for a number of epochs and measures its performance.
It implements the logic for running train, validate and test epochs.
If the performance increases, the trainer will save the model in a designated folder.
The trainer also implements early stopping, which can be configured in the settings (see below).
The generic_trainer in the folder can be used to train most 'standard' model.
For special cases, new trainers can easily be implemented following the implementation of the generic trainer.

4. Logger.
One of the goal of the project is to reduce the code needed to instrument, run the experiments and save the measurements.
The logger classes handle the logic for saving the measurements and the code.
Two options are available at the moment: Tensorboard for pytorch and sacred](https://github.com/IDSIA/sacred).
Both can be configured in the settings (see below).

5. Experiment.
Self explainable.


6. Utils.
Self explainable.


Running an experiment:

In order to run an experiment, you have to add a configuration file in 'configs/runs', specifying the data, the model, the optimizer, the loss functions, the metrics, the trainer and configuring the logger.
The file 'example_config.json' is self explainable.

After configuring this file you can run an experimeng using:
```
python train.py --config=configs/runs/example_config.json
```

Currently, the Sacred logger indexes everything is a MongoDB database.
If you want to use this logger, you can run a MongoDB instance using the docker-compose file in the repo:

```
docker-compose -f mongodb.yml up -d
```




Historical considerations:
This project started with a few changes to the project [pytorch-template](https://github.com/victoresque/pytorch-template), but evolved into a stand-alone framework, with deep structural changes:

* Everything is written in an OOP fashion - each loss, metric, trainer is a class which inherits from a base class.
* New loss functions must implement a 'forwarward' function, similar to nn.Module in pytorch.
*
* New Trainers can easily be written and configured in the .json file.
* Testing can also be done during training.
* Logging is rebuild in order to make it easy to add new loggers. The old project only allowed tensorboard. This project also implements [sacred](https://github.com/IDSIA/sacred).
* Logging is improved so we can either log at the end of one epoch or at the end of each batch.
* New loggers
* More models + configs (with new to come, feel free to add any)
