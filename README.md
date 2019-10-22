# A PyTorch Framework for Experiment Management

This project is a customizable experimental management framework for Pytorch aimed to reduce the amount of code needed to run new experiments and gather measurements.
The goal is to have a general configuration file from where we can control the model to be trained, the data, the loss function, the metrics used during training and select different loggers which will save the data.
Sharing experiments should only involve sharing the config file.

The framework also integrates natively with other projects such as [sacred](https://github.com/IDSIA/sacred).

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

#### 7. Learning schedulers.

Several learning schedulers can be configured using priorities. The project also implements a custom, dynamic learnning rate scheduler.


### Running an experiment:

In order to run an experiment, you have to add a configuration file in 'mtorch/configs/runs', specifying the data, the model, the optimizer, the loss functions, the metrics, the trainer and configuring the logger.
The file 'default.json' is self explainable.

After configuring this file you can run an experimeng using:
```
python mtorch/train.py --config=mtorch/configs/runs/default.json
```

Currently, the Sacred logger indexes everything is a MongoDB database and uses [Omniboard](https://vivekratnavel.github.io/omniboard/#/) for visualizations.
If you want to use this logger, you can run a MongoDB instance using the docker-compose file in the repo:

```
docker-compose -f mongo-omniboard.yml up -d
```

The [py-elasticinfrastructure](https://github.com/NullConvergence/py-elasticinfrastructure) uses elasticsearch to index metrics the machin you run experiments on (and, optionally, kibana to mine them).
For running an elasticsearch cluster see the project [readme](https://github.com/NullConvergence/py-elasticinfrastructure/blob/master/README.md).


##### Historical considerations:

This project started with a few changes to the project [pytorch-template](https://github.com/victoresque/pytorch-template), but evolved into a stand-alone framework, with deep structural changes.
