from torch_temp.model.blocks.densenet import DenseNet


class DenseNetWrapper(DenseNet):
    def __init__(self, block, nblocks,
                 growth_rate=12, reduction=0.5,
                 num_classes=10):
        super(DenseNetWrapper, self).__init__(block, nblocks,
                                              growth_rate=growth_rate,
                                              reduction=reduction,
                                              num_classes=num_classes)
