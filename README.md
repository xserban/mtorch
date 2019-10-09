# PyTorch Template

This project is a customizable experimental framework for Pytorch aimed to reduce the amount of code needed to run new experiments and gather measurements.
The goal is to have a general configuration file from where we can control the model to be trained, the data, the loss function, the metrics used during training and select different loggers which will save the data.

The main modules are:

#### 1. Data.

The data module consists of data loaders and datasets, both subclasses of pytorch DataLoader and Dataset, respectively.
These modules allow to easily add new datasets and create data loaders which can be used during training or testing.
In particular, the datasets folder hosts custom datasets, which are not available through pytorch.
Moreover, in the data module we can define custom transformations for each dataset. The transformations will be saved in the logs so we can keep track of the trans. used during one experiment.

#### 2. Model.

The model module consists of different model architectures, loss functions and metrics used to measure performance.
The architectures are based on building blocks, which can be found in the 'blocks' folder.
Defining new architectures is straightforward following the ResNet example in the repo.
Similarly, defining new loss functions or new metrics is trivial, following the examples in the repo.

#### 3. Trainer.

A trainer runs the model for a number of epochs and measures its performance.
It implements the logic for running train, validate and test epochs.
If the performance increases, the trainer will save the model in a designated folder.
The trainer also implements early stopping, which can be configured in the settings (see below).
The default in the folder can be used to train most 'standard' model.

For special cases, new trainers can easily be implemented by creating a new class and inheriting the BaseTrainer class (see default.py).

#### 4. Logger.

One of the goal of the project is to reduce the code needed to instrument, run the experiments and save the measurements.
The logger classes handle the logic for saving the measurements and the code.
Three options are available at the moment: Tensorboard for pytorch, [sacred](https://github.com/IDSIA/sacred) and [py-elasticinfrastructure](https://github.com/NullConvergence/py-elasticinfrastructure) which gathers hardware metrics in elasticsearch.
All can be configured in the settings (see below).


#### 5. Experiment.

Self explainable.


#### 6. Utils.

Self explainable.

#### 7. Learning rate scheduler.

This project provides a dynamic learning rate scheduler which allows to set the value of the learning rate at different epochs.
For example, you may run the first 40 epochs at 0.01, then up to epoch 60 with learning rate 0.1, etc.
The dynamic learning rate takes priority over the pytorch learning rate scheduler (if used).
This means after the dynamic learning rate scheduler updated all learning rates, the pytorch learning rate scheduler kicks in and can dicrease the learning rate every n-epochs (if the StepLR is used), etc.


### Running an experiment:

In order to run an experiment, you have to add a configuration file in 'configs/runs', specifying the data, the model, the optimizer, the loss functions, the metrics, the trainer and configuring the logger.
The file 'default.json' is self explainable.

After configuring this file you can run an experimeng using:
```
python train.py --config=configs/runs/default.json
```

Currently, the Sacred logger indexes everything is a MongoDB database and uses [Omniboard](https://vivekratnavel.github.io/omniboard/#/) for visualizations.
If you want to use this logger, you can run a MongoDB instance using the docker-compose file in the repo:

```
docker-compose -f mongo-omniboard.yml up -d
```

The [py-elasticinfrastructure](https://github.com/NullConvergence/py-elasticinfrastructure) uses elasticsearch to index metrics the machin you run experiments on (and, optionally, kibana to mine them).
For running an elasticsearch cluster see the project [readme](https://github.com/NullConvergence/py-elasticinfrastructure/blob/master/README.md).


##### Historical considerations:

This project started with a few changes to the project [pytorch-template](https://github.com/victoresque/pytorch-template), but evolved into a stand-alone framework, with deep structural changes:

* Everything is written in an OOP fashion - each loss, metric, trainer is a class which inherits from a base class.
* There is a new 'datasets' module used to load data not included pytorch.
* New loss functions are now defined as classes and must implement a 'forward' function, similar to the nn.Module in pytorch.
* New metrics are now defined as classes and must implement a 'forward' function, similar to the nn.Module in pytorch.
* It is easier to write models using the included building blocks. The models are more modular now.
* New Trainers can easily be written and configured in the .json file.
* Testing can also be done during training.
* Logging is rebuild in order to make it easy to add new loggers. The old project only allowed tensorboard. This project also implements [sacred](https://github.com/IDSIA/sacred).
* Logging is improved so we can either log at the end of one epoch or at the end of each batch.
* The loggers are now defined as classes. Besides tensorboard, sacred and py-elasticinfrastructure were added.
* More models + configs (with new to come, feel free to add any)
