import argparse
import collections
import torch
import data.data_loaders as module_data
import model.loss as module_loss
import model.metrics as module_metric
import model.arch as module_arch
import trainer as module_train

from utils.parse_config import ConfigParser
from experiment.sacred import Sacred


def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    train_data_loader = config.initialize(
        module_data, config['train_data_loader'])
    valid_data_loader = train_data_loader.split_validation()

    if config["test"]["do"]:
        test_data_loader = getattr(module_data, config['train_data_loader']['type'])(
            config['train_data_loader']['args']['data_dir'],
            batch_size=config['test']['test_batch_size'],
            shuffle=False,
            validation_split=0.0,
            training=False,
            num_workers=config['train_data_loader']['args']['num_workers']
        )
    else:
        test_data_loader = None

    # build model architecture, then print to console
    model = config.initialize(module_arch, config['arch'])
    logger.info(model)

    # get function handles of loss and metrics
    loss = config.initialize(module_loss, config['loss_function'])
    metrics = [config.initialize(module_metric, met)
               for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.initialize(
        torch.optim, config['optimizer'], trainable_params)

    lr_scheduler = config.initialize(
        torch.optim.lr_scheduler, config['lr_scheduler'], optimizer)

    trainer_args = {
        'model': model,
        'loss': loss,
        'metrics': metrics,
        'optimizer': optimizer,
        'config': config,
        'train_data_loader': train_data_loader,
        'valid_data_loader': valid_data_loader,
        'test_data_loader': test_data_loader,
        'lr_scheduler': lr_scheduler
    }

    trainer = config.initialize(
        module_train, config['trainer'], **trainer_args)
    trainer.train()


def main_sacred(main, config):
    exp = Sacred(config)
    exp.run(main, config)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float,
                   target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int,
                   target=('data_loader', 'args', 'batch_size'))
    ]
    config = ConfigParser(args, options)

    if config['trainer']['sacred_logs']['do'] is False:
        main(config)
    else:
        main_sacred(main, config)
